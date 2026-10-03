#!/usr/bin/env python3
"""Deterministic P0 document extraction and rule-filter pipeline.

The module keeps pure normalization, ID, split, and filtering functions free of
third-party dependencies. The CLI imports PyArrow only for Parquet I/O so the
core logic can be regression-tested on a minimal Python installation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Iterator
import unicodedata
from urllib.parse import urlsplit, urlunsplit


PIPELINE_VERSION = "p0_document_pipeline_v2"
CODE_DOMAINS = {"code", "code_python", "code_shell", "code_other"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str, max_blank_line_run: int = 3) -> str:
    """Normalize encoding/newlines without flattening code or math layout."""
    text = unicodedata.normalize("NFC", text.replace("\x00", ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    result: list[str] = []
    blank_run = 0
    for line in lines:
        if line:
            blank_run = 0
            result.append(line)
        else:
            blank_run += 1
            if blank_run <= max_blank_line_run:
                result.append("")
    return "\n".join(result).strip()


def normalize_url_for_family(value: str) -> str:
    """Remove fragments/query while preserving a stable page-family locator."""
    value = value.strip()
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if not parts.scheme or not parts.netloc:
        return value
    path = re.sub(r"/+", "/", parts.path or "/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def first_nonempty(row: dict[str, Any], fields: Iterable[str]) -> str | None:
    for name in fields:
        value = row.get(name)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            rendered = canonical_json(value)
        else:
            rendered = str(value).strip()
        if rendered:
            return rendered
    return None


def row_matches_source(row: dict[str, Any], source: dict[str, Any]) -> bool:
    """Apply declarative source partition filters before normalization."""
    for item in source.get("row_filters", []):
        if row.get(item["field"]) not in item["allowed_values"]:
            return False
    return True


def build_source_locator(
    row: dict[str, Any],
    fields: Iterable[str],
    row_index: int,
    source_file: str | None = None,
) -> str:
    values = []
    for name in fields:
        value = row.get(name)
        if value is not None and str(value).strip():
            values.append(f"{name}={value}")
    if values:
        return "|".join(values)
    file_part = f"file={source_file}|" if source_file else ""
    return f"{file_part}row={row_index}"


def build_family_id(
    row: dict[str, Any], fields: Iterable[str], source_id: str, source_locator: str
) -> str:
    # A declared URL/repository/problem family must join across data sources.
    # Prefixing it with source_id would allow the same page to leak across
    # train/test when it appears in two corpora. The field name namespaces
    # unrelated identifiers; only the row fallback remains source-specific.
    for name in fields:
        value = first_nonempty(row, [name])
        if not value:
            continue
        if value.startswith(("http://", "https://")):
            value = normalize_url_for_family(value)
        return sha256_text(f"family\0{name}={value}")
    return sha256_text(f"fallback\0{source_id}\0{source_locator}")


def build_document_id(
    source_id: str, source_revision: str, source_locator: str, normalized_sha256: str
) -> str:
    return sha256_text(
        "\0".join((source_id, source_revision, source_locator, normalized_sha256))
    )


def assign_split(
    family_id: str,
    seed: int,
    train_fraction: float = 0.98,
    validation_fraction: float = 0.01,
) -> str:
    digest = hashlib.sha256(f"{seed}\0{family_id}".encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    if position < train_fraction:
        return "train"
    if position < train_fraction + validation_fraction:
        return "validation"
    return "test"


def repeated_line_fraction(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 1.0
    counts = Counter(lines)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(lines)


def replacement_char_fraction(text: str) -> float:
    return text.count("\ufffd") / max(1, len(text))


def rule_filter(
    text: str,
    domain: str,
    *,
    min_chars: int = 200,
    max_chars: int = 200_000,
    max_replacement_char_fraction: float = 0.001,
    max_repeated_line_fraction: float = 0.6,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if len(text) < min_chars:
        reasons.append("too_short")
    if len(text) > max_chars:
        reasons.append("too_long")
    if replacement_char_fraction(text) > max_replacement_char_fraction:
        reasons.append("replacement_char_ratio")
    if repeated_line_fraction(text) > max_repeated_line_fraction:
        reasons.append("repeated_lines")
    lowered = text.lower()
    if "enable javascript and cookies" in lowered or "access denied" == lowered.strip():
        reasons.append("non_content_page")
    if domain in CODE_DOMAINS or domain.startswith("code_"):
        nonblank_lines = [line for line in text.splitlines() if line.strip()]
        if len(nonblank_lines) < 4:
            reasons.append("code_too_few_lines")
        longest = max((len(line) for line in nonblank_lines), default=0)
        if longest > 20_000:
            reasons.append("minified_or_generated_code")
    return not reasons, sorted(set(reasons))


def make_record(
    row: dict[str, Any],
    row_index: int,
    source: dict[str, Any],
    pipeline: dict[str, Any],
    source_file: str | None = None,
) -> dict[str, Any]:
    raw_text = row.get(source["text_field"])
    if not isinstance(raw_text, str):
        raw_text = "" if raw_text is None else str(raw_text)
    normalized = normalize_text(raw_text, pipeline["rules"]["max_blank_line_run"])
    raw_hash = sha256_text(raw_text)
    normalized_hash = sha256_text(normalized)
    locator = build_source_locator(
        row, source["locator_fields"], row_index, source_file=source_file
    )
    family_id = build_family_id(row, source["family_fields"], source["source_id"], locator)
    document_id = build_document_id(
        source["source_id"], source["revision"], locator, normalized_hash
    )
    keep, reasons = rule_filter(
        normalized,
        source["domain"],
        min_chars=pipeline["rules"]["min_chars"],
        max_chars=pipeline["rules"]["max_chars"],
        max_replacement_char_fraction=pipeline["rules"]["max_replacement_char_fraction"],
        max_repeated_line_fraction=pipeline["rules"]["max_repeated_line_fraction"],
    )
    metadata = {name: row.get(name) for name in source.get("metadata_fields", []) if name in row}
    return {
        "document_id": document_id,
        "source_id": source["source_id"],
        "source_revision": source["revision"],
        "source_locator": locator,
        "family_id": family_id,
        "domain": source["domain"],
        "language": source["language"],
        "text": normalized,
        "raw_sha256": raw_hash,
        "normalized_sha256": normalized_hash,
        "license_status": source["license_status"],
        "pipeline_version": PIPELINE_VERSION,
        "split": assign_split(
            family_id,
            pipeline["pipeline_seed"],
            pipeline["split"]["train"],
            pipeline["split"]["validation"],
        ),
        "rule_keep": keep,
        "rule_reason_codes": reasons,
        "source_metadata_json": canonical_json(metadata),
    }


def representative_text(text: str, max_chars: int = 12_000) -> str:
    if len(text) <= max_chars:
        return text
    head = math.ceil(max_chars * 0.6)
    tail = max_chars - head
    return text[:head] + "\n\n[...中间省略...]\n\n" + text[-tail:]


def make_teacher_request(record: dict[str, Any], max_chars: int = 12_000) -> dict[str, Any]:
    return {
        "request_id": sha256_text(
            "\0".join(("p0_quality_v1", record["document_id"], record["normalized_sha256"]))
        ),
        "document_id": record["document_id"],
        "domain": record["domain"],
        "language": record["language"],
        "text_sha256": record["normalized_sha256"],
        "text": representative_text(record["text"], max_chars),
        "prompt_version": "p0_quality_v1",
        "schema_version": "p0_teacher_quality_v1",
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def validate_pipeline_config(pipeline: dict[str, Any]) -> None:
    required = {"pipeline_seed", "split", "rules"}
    missing = sorted(required - pipeline.keys())
    if missing:
        raise ValueError(f"pipeline config missing fields: {missing}")
    split = pipeline["split"]
    fractions = [split.get(name) for name in ("train", "validation", "test")]
    if any(not isinstance(value, (int, float)) or value < 0 for value in fractions):
        raise ValueError("split fractions must be non-negative numbers")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"split fractions must sum to 1, got {sum(fractions)}")


def validate_source_config(source: dict[str, Any]) -> None:
    required = {
        "adapter_version",
        "source_id",
        "dataset",
        "revision",
        "subset",
        "domain",
        "language",
        "license_status",
        "text_field",
        "locator_fields",
        "family_fields",
        "metadata_fields",
    }
    missing = sorted(required - source.keys())
    if missing:
        raise ValueError(f"source config missing fields: {missing}")
    if source["adapter_version"] != "p0_source_adapter_v1":
        raise ValueError(f"unsupported adapter_version: {source['adapter_version']!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
        raise ValueError("source revision must be a 40-character lowercase commit hash")
    for name in ("locator_fields", "family_fields", "metadata_fields"):
        if not isinstance(source[name], list) or not all(
            isinstance(value, str) and value for value in source[name]
        ):
            raise ValueError(f"{name} must be a list of non-empty field names")
    if not source["locator_fields"]:
        raise ValueError("locator_fields cannot be empty")
    if source["license_status"] not in {"pending", "approved", "rejected"}:
        raise ValueError("license_status must be pending, approved, or rejected")
    filters = source.get("row_filters", [])
    if not isinstance(filters, list):
        raise ValueError("row_filters must be a list")
    for item in filters:
        if (
            not isinstance(item, dict)
            or set(item) != {"field", "allowed_values"}
            or not isinstance(item["field"], str)
            or not item["field"]
            or not isinstance(item["allowed_values"], list)
            or not item["allowed_values"]
        ):
            raise ValueError("each row filter needs field and non-empty allowed_values")


def validate_source_fields(source: dict[str, Any], available_fields: Iterable[str]) -> dict[str, Any]:
    """Fail closed on identity/text fields; report optional metadata drift."""
    available = set(available_fields)
    text_field = source["text_field"]
    present_locators = [name for name in source["locator_fields"] if name in available]
    present_families = [name for name in source["family_fields"] if name in available]
    missing_metadata = [name for name in source["metadata_fields"] if name not in available]
    filter_fields = [item["field"] for item in source.get("row_filters", [])]
    if text_field not in available:
        raise ValueError(f"text_field {text_field!r} is absent from Parquet schema")
    if not present_locators:
        raise ValueError(
            "none of locator_fields are present in Parquet schema: "
            f"{source['locator_fields']}"
        )
    missing_filter_fields = [name for name in filter_fields if name not in available]
    if missing_filter_fields:
        raise ValueError(f"row filter fields absent from Parquet schema: {missing_filter_fields}")
    return {
        "text_field": text_field,
        "locator_fields_present": present_locators,
        "family_fields_present": present_families,
        "metadata_fields_missing": missing_metadata,
        "row_filter_fields": filter_fields,
    }


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def iter_parquet_rows(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit("PyArrow is required for Parquet commands") from error
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def parquet_schema(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit("PyArrow is required for Parquet commands") from error
    parquet = pq.ParquetFile(path)
    return {
        "path": str(path),
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "schema": str(parquet.schema_arrow),
        "fields": parquet.schema_arrow.names,
    }


def command_inspect(args: argparse.Namespace) -> None:
    reports = [parquet_schema(Path(path)) for path in args.input]
    print(json.dumps(reports, ensure_ascii=False, indent=2))


def command_validate_source(args: argparse.Namespace) -> None:
    source = load_json(Path(args.source_config))
    validate_source_config(source)
    reports = []
    for value in args.input:
        path = Path(value)
        schema = parquet_schema(path)
        validation = validate_source_fields(source, schema["fields"])
        reports.append({**schema, "source_validation": validation, "valid": True})
    print(json.dumps(reports, ensure_ascii=False, indent=2))


def command_validate_registry(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry)
    registry = load_json(registry_path)
    if registry.get("registry_version") != "p0_source_registry_v1":
        raise ValueError("unsupported or missing registry_version")
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("registry sources must be a non-empty list")
    repo_root = Path(args.repo_root).resolve()
    raw_root = Path(args.raw_root).resolve() if args.raw_root else None
    seen: set[str] = set()
    reports = []
    for entry in sources:
        source_id = entry.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("each registry entry needs a source_id")
        if source_id in seen:
            raise ValueError(f"duplicate source_id in registry: {source_id}")
        seen.add(source_id)
        adapter_path = repo_root / entry["source_config"]
        adapter = load_json(adapter_path)
        validate_source_config(adapter)
        for key in ("source_id", "dataset", "revision", "subset", "license_status"):
            if adapter.get(key) != entry.get(key):
                raise ValueError(
                    f"{source_id}: registry/adapter mismatch for {key}: "
                    f"{entry.get(key)!r} != {adapter.get(key)!r}"
                )
        report = {
            "source_id": source_id,
            "adapter": str(adapter_path),
            "adapter_status": entry.get("adapter_status"),
            "raw_available": False,
        }
        if raw_root is not None:
            raw_path = raw_root / entry["raw_relpath"]
            report["raw_path"] = str(raw_path)
            report["raw_available"] = raw_path.is_file()
            if raw_path.is_file():
                actual_bytes = raw_path.stat().st_size
                actual_hash = sha256_file(raw_path)
                report["actual_bytes"] = actual_bytes
                report["sha256"] = actual_hash
                if entry.get("actual_bytes") is not None and actual_bytes != entry["actual_bytes"]:
                    raise ValueError(f"{source_id}: actual_bytes mismatch")
                if entry.get("sha256") is not None and actual_hash != entry["sha256"]:
                    raise ValueError(f"{source_id}: sha256 mismatch")
            elif entry.get("adapter_status") == "validated":
                raise ValueError(f"{source_id}: validated source file is missing: {raw_path}")
        reports.append(report)
    print(
        json.dumps(
            {
                "registry": str(registry_path),
                "run_id": registry.get("run_id"),
                "sources": reports,
                "valid": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_extract(args: argparse.Namespace) -> None:
    pipeline = load_json(Path(args.pipeline_config))
    source = load_json(Path(args.source_config))
    validate_pipeline_config(pipeline)
    validate_source_config(source)
    input_paths = [Path(path) for path in args.input]
    validations = []
    for path in input_paths:
        schema = parquet_schema(path)
        validations.append(
            {"path": str(path), **validate_source_fields(source, schema["fields"])}
        )
    output = Path(args.output)
    counters: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    output_hash = hashlib.sha256()
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for input_path in input_paths:
                for row_index, row in enumerate(iter_parquet_rows(input_path, args.batch_size)):
                    record = make_record(
                        row,
                        row_index,
                        source,
                        pipeline,
                        source_file=input_path.name,
                    )
                    counters["seen"] += 1
                    counters["kept"] += int(record["rule_keep"])
                    counters["dropped"] += int(not record["rule_keep"])
                    for reason in record["rule_reason_codes"]:
                        counters[f"drop:{reason}"] += 1
                    line = canonical_json(record) + "\n"
                    handle.write(line)
                    output_hash.update(line.encode("utf-8"))
                    if args.max_rows and counters["seen"] >= args.max_rows:
                        break
                if args.max_rows and counters["seen"] >= args.max_rows:
                    break
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_config": str(Path(args.pipeline_config)),
        "source_config": str(Path(args.source_config)),
        "inputs": args.input,
        "source_validations": validations,
        "output": str(output),
        "counts": dict(sorted(counters.items())),
        "output_sha256": output_hash.hexdigest(),
    }
    atomic_write_text(output.with_suffix(output.suffix + ".manifest.json"),
                      json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_teacher_requests(args: argparse.Namespace) -> None:
    output = Path(args.output)
    requests: list[dict[str, Any]] = []
    with Path(args.input).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("rule_keep") and record.get("license_status") != "rejected":
                requests.append(make_teacher_request(record, args.max_chars))
            if args.max_rows and len(requests) >= args.max_rows:
                break
    lines = "".join(canonical_json(item) + "\n" for item in requests)
    atomic_write_text(output, lines)
    report = {
        "input": args.input,
        "output": str(output),
        "requests": len(requests),
        "output_sha256": sha256_text(lines),
    }
    atomic_write_text(output.with_suffix(output.suffix + ".manifest.json"),
                      json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="print Parquet schema metadata")
    inspect_parser.add_argument("--input", action="append", required=True)
    inspect_parser.set_defaults(func=command_inspect)

    validate_parser = subparsers.add_parser(
        "validate-source", help="validate a source adapter against Parquet fields"
    )
    validate_parser.add_argument("--source-config", required=True)
    validate_parser.add_argument("--input", action="append", required=True)
    validate_parser.set_defaults(func=command_validate_source)

    registry_parser = subparsers.add_parser(
        "validate-registry", help="cross-check source registry, adapters, and raw files"
    )
    registry_parser.add_argument("--registry", required=True)
    registry_parser.add_argument("--repo-root", default=".")
    registry_parser.add_argument("--raw-root")
    registry_parser.set_defaults(func=command_validate_registry)

    extract_parser = subparsers.add_parser("extract", help="normalize and rule-filter rows")
    extract_parser.add_argument("--pipeline-config", required=True)
    extract_parser.add_argument("--source-config", required=True)
    extract_parser.add_argument("--input", action="append", required=True)
    extract_parser.add_argument("--output", required=True)
    extract_parser.add_argument("--batch-size", type=int, default=256)
    extract_parser.add_argument("--max-rows", type=int)
    extract_parser.set_defaults(func=command_extract)

    teacher_parser = subparsers.add_parser(
        "teacher-requests", help="build idempotent teacher request JSONL"
    )
    teacher_parser.add_argument("--input", required=True)
    teacher_parser.add_argument("--output", required=True)
    teacher_parser.add_argument("--max-chars", type=int, default=12_000)
    teacher_parser.add_argument("--max-rows", type=int)
    teacher_parser.set_defaults(func=command_teacher_requests)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
