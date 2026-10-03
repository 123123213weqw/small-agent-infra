#!/usr/bin/env python3
"""Deterministic, resumable rule cleaning of pinned FineWiki zh Parquet shards.

This is NOT a quality or license approval. Raw files remain untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import p0_data_pipeline as core


SOURCE_ID = "finewiki_zh"
SEED = 20260923
MIN_CHARS = 500
MAX_CHARS = 200_000
COLUMNS = ["text", "id", "wikiname", "page_id", "title", "url",
           "date_modified", "in_language", "version", "has_math"]


def clean_record(row: dict, revision: str, filename: str) -> tuple[dict | None, list[str]]:
    reasons = []
    identifier = row.get("id")
    url = row.get("url")
    if row.get("wikiname") != "zhwiki" or row.get("in_language") != "zh":
        reasons.append("wrong_wiki_or_language")
    if not isinstance(identifier, str) or not identifier.startswith("zhwiki/"):
        reasons.append("bad_id")
    if not isinstance(url, str) or not url.startswith("https://zh.wikipedia.org/wiki/"):
        reasons.append("bad_url")
    if not isinstance(row.get("title"), str) or not row["title"].strip():
        reasons.append("missing_title")
    text = core.normalize_text(row.get("text") or "")
    keep, rule_reasons = core.rule_filter(text, "general_zh", min_chars=MIN_CHARS,
                                         max_chars=MAX_CHARS)
    reasons.extend(rule_reasons)
    if reasons:
        return None, sorted(set(reasons))
    normalized_sha = core.sha256_text(text)
    locator = f"id={identifier}|url={url}"
    family = core.build_family_id(row, ["url"], SOURCE_ID, locator)
    return {
        "document_id": core.build_document_id(SOURCE_ID, revision, locator, normalized_sha),
        "source_id": SOURCE_ID,
        "source_revision": revision,
        "source_locator": locator,
        "family_id": family,
        "split": core.assign_split(family, SEED),
        "domain": "general_zh",
        "language": "zh",
        "license_status": "pending",
        "id": identifier,
        "url": url,
        "title": row["title"],
        "page_id": int(row["page_id"]) if row.get("page_id") is not None else None,
        "version": str(row["version"]) if row.get("version") is not None else None,
        "date_modified": str(row["date_modified"]) if row.get("date_modified") is not None else None,
        "has_math": bool(row.get("has_math")),
        "text": text,
        "chars": len(text),
        "normalized_sha256": normalized_sha,
        "raw_file": filename,
    }, []


def output_schema():
    import pyarrow as pa
    return pa.schema([
        ("document_id", pa.string()), ("source_id", pa.string()),
        ("source_revision", pa.string()), ("source_locator", pa.string()),
        ("family_id", pa.string()), ("split", pa.string()),
        ("domain", pa.string()), ("language", pa.string()),
        ("license_status", pa.string()), ("id", pa.string()),
        ("url", pa.string()), ("title", pa.string()),
        ("page_id", pa.int64()), ("version", pa.string()),
        ("date_modified", pa.string()), ("has_math", pa.bool_()),
        ("text", pa.string()), ("chars", pa.int64()),
        ("normalized_sha256", pa.string()), ("raw_file", pa.string()),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    import pyarrow as pa
    import pyarrow.parquet as pq

    spec = json.loads(args.spec.read_text())
    raw_manifest_path = args.root / "raw_manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text())
    if (spec["revision"] != raw_manifest["revision"] or
            raw_manifest["spec_sha256"] != core.sha256_file(args.spec) or
            raw_manifest["license_status"] != "pending" or
            len(raw_manifest["files"]) != len(spec["files"])):
        raise ValueError("raw manifest/spec mismatch")
    raw_index = {x["file"]: x for x in raw_manifest["files"]}
    output_dir = args.root / "clean_rule_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = output_schema()
    global_hashes: set[str] = set()
    aggregate = Counter()
    reports = []

    for entry in spec["files"]:
        filename = Path(entry["path"]).name
        raw = args.root / "raw" / filename
        output = output_dir / filename
        sidecar = output_dir / f"{filename}.manifest.json"
        if (filename not in raw_index or not raw.is_file() or
                raw.stat().st_size != entry["bytes"] or
                raw_index[filename]["sha256"] != entry["sha256"]):
            raise ValueError(f"raw file not verified: {filename}")
        if output.exists() or sidecar.exists():
            # The data file is renamed before its sidecar is written. If the
            # process died in that narrow window, regenerate only that shard.
            if output.exists() and not sidecar.exists():
                output.unlink()
            elif not output.exists():
                raise ValueError(f"incomplete output pair: {filename}")
        if output.exists():
            report = json.loads(sidecar.read_text())
            if report["output_sha256"] != core.sha256_file(output):
                raise ValueError(f"existing output SHA mismatch: {filename}")
            for batch in pq.ParquetFile(output).iter_batches(columns=["normalized_sha256"]):
                for digest in batch.column(0).to_pylist():
                    if digest in global_hashes:
                        raise ValueError("duplicate across completed shards")
                    global_hashes.add(digest)
            reports.append(report)
            aggregate.update(report["counts"])
            print(f"verified existing {filename}", flush=True)
            continue
        source = pq.ParquetFile(raw)
        if not set(COLUMNS).issubset(source.schema_arrow.names):
            raise ValueError(f"schema mismatch in {filename}: {source.schema_arrow.names}")
        temporary = output_dir / f".{filename}.partial"
        temporary.unlink(missing_ok=True)  # only an incomplete output from this stage
        counts = Counter()
        with pq.ParquetWriter(temporary, schema=schema, compression="zstd",
                              compression_level=3) as writer:
            for batch in source.iter_batches(batch_size=args.batch_size, columns=COLUMNS):
                accepted = []
                for row in batch.to_pylist():
                    counts["input_rows"] += 1
                    raw_length = len(row.get("text") or "")
                    input_bin = "lt500" if raw_length < 500 else "500to1999" if raw_length < 2000 else "ge2000"
                    counts[f"input_length:{input_bin}"] += 1
                    record, reasons = clean_record(row, spec["revision"], filename)
                    if record is None:
                        counts["rule_rejected"] += 1
                        for reason in reasons:
                            counts[f"reject:{reason}"] += 1
                        continue
                    digest = record["normalized_sha256"]
                    if digest in global_hashes:
                        counts["internal_exact_duplicate"] += 1
                        continue
                    global_hashes.add(digest)
                    accepted.append(record)
                    counts["kept"] += 1
                    counts["kept_chars"] += record["chars"]
                    counts["kept_has_math"] += record["has_math"]
                    kept_bin = "lt500" if record["chars"] < 500 else "500to1999" if record["chars"] < 2000 else "ge2000"
                    counts[f"kept_length:{kept_bin}"] += 1
                    counts[f"split:{record['split']}"] += 1
                    counts["markup_template_prefix"] += "{{" in record["text"][:1000]
                if accepted:
                    writer.write_table(pa.Table.from_pylist(accepted, schema=schema))
                if counts["input_rows"] % 25000 < args.batch_size:
                    print(f"{filename}: seen={counts['input_rows']} kept={counts['kept']}", flush=True)
        output_hash = core.sha256_file(temporary)
        temporary.replace(output)
        report = {
            "stage": "finewiki_zh_rule_clean_audit_only",
            "source_revision": spec["revision"], "raw_file": filename,
            "raw_sha256": entry["sha256"], "output_file": filename,
            "output_bytes": output.stat().st_size, "output_sha256": output_hash,
            "counts": dict(counts), "min_chars": MIN_CHARS, "max_chars": MAX_CHARS,
            "license_status": "pending", "training_eligible": False,
        }
        core.atomic_write_text(sidecar, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        reports.append(report)
        aggregate.update(counts)
        print(json.dumps({"file": filename, "counts": dict(counts)}, ensure_ascii=False), flush=True)

    summary = {
        "stage": "finewiki_zh_rule_clean_audit_only",
        "dataset": spec["dataset"], "revision": spec["revision"],
        "spec_sha256": core.sha256_file(args.spec),
        "cleaner_sha256": core.sha256_file(Path(__file__)),
        "core_sha256": core.sha256_file(Path(core.__file__)),
        "counts": dict(aggregate), "files": reports,
        "license_status": "pending", "training_eligible": False,
        "remaining_gates": ["quality_model_audit", "cross_source_near_dedup", "license_review", "heldout_exclusion"],
    }
    core.atomic_write_text(output_dir / "manifest.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"complete": True, "counts": dict(aggregate)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
