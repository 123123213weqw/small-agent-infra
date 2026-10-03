#!/usr/bin/env python3
"""Build deterministic, offline P0 normalized and candidate Parquet shards.

The pipeline has two explicit stages:

1. ``normalize`` adapts raw Parquet rows into the canonical document schema and
   records rule decisions without destroying rejected rows.
2. ``build`` performs a source-set exact-content deduplication, groups families,
   greedily removes near duplicates in deterministic sample-rank order, counts
   exact tokenizer tokens, and writes bounded candidate pools.
3. ``consolidate`` combines independently built candidate pools, verifies their
   provenance, repeats exact and near deduplication across every pool, enforces
   family/split consistency, and writes one unified candidate index.

Large-file dependencies (PyArrow and tokenizers) are imported only by commands;
the hashing and MinHash helpers stay testable with the standard library.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import sys
import tempfile
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
CORE_SPEC = importlib.util.spec_from_file_location(
    "p0_data_pipeline", ROOT / "scripts" / "p0_data_pipeline.py"
)
CORE = importlib.util.module_from_spec(CORE_SPEC)
assert CORE_SPEC.loader is not None
CORE_SPEC.loader.exec_module(CORE)


CANDIDATE_PIPELINE_VERSION = "p0_candidate_pipeline_v1"
TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[\u3400-\u4dbf\u4e00-\u9fff]|[^\s]"
)
MINHASH_PRIME = (1 << 61) - 1
IMPLEMENTATION_HASHES = {
    "candidate_script_sha256": CORE.sha256_file(Path(__file__).resolve()),
    "document_pipeline_sha256": CORE.sha256_file(
        (ROOT / "scripts" / "p0_data_pipeline.py").resolve()
    ),
}


def implementation_hashes() -> dict[str, str]:
    # Snapshot taken at process start so a later deployment cannot make a
    # long-running job claim that it executed different on-disk code.
    return dict(IMPLEMENTATION_HASHES)


def stable_sample_rank(seed: int, document_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{document_id}".encode("utf-8")).hexdigest()


def normalized_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def text_shingles(text: str, size: int = 5, max_shingles: int = 8192) -> set[int]:
    tokens = normalized_tokens(text)
    if not tokens:
        return set()
    if len(tokens) < size:
        raw = {"\x1f".join(tokens)}
    else:
        raw = {"\x1f".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}
    # Bound pathological documents deterministically by keeping the smallest hashes.
    hashes = (
        int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")
        % MINHASH_PRIME
        for value in raw
    )
    return set(heapq.nsmallest(max_shingles, hashes))


def minhash_coefficients(num_perm: int, seed: int) -> tuple[list[int], list[int]]:
    generator = random.Random(seed)
    a = [generator.randrange(1, MINHASH_PRIME) for _ in range(num_perm)]
    b = [generator.randrange(0, MINHASH_PRIME) for _ in range(num_perm)]
    return a, b


def minhash_signature(
    shingles: set[int], coefficients: tuple[list[int], list[int]]
) -> tuple[int, ...]:
    a_values, b_values = coefficients
    if not shingles:
        return tuple([MINHASH_PRIME] * len(a_values))
    return tuple(
        min((a * value + b) % MINHASH_PRIME for value in shingles)
        for a, b in zip(a_values, b_values)
    )


def jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


class MinHashDeduper:
    """Deterministic LSH candidate lookup followed by exact shingle Jaccard."""

    def __init__(
        self,
        *,
        seed: int,
        shingle_size: int,
        num_perm: int,
        bands: int,
        threshold: float,
    ) -> None:
        if num_perm <= 0 or bands <= 0 or num_perm % bands:
            raise ValueError("minhash_num_perm must be positive and divisible by bands")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("near_jaccard_threshold must be in [0, 1]")
        self.shingle_size = shingle_size
        self.num_perm = num_perm
        self.bands = bands
        self.rows_per_band = num_perm // bands
        self.threshold = threshold
        self.coefficients = minhash_coefficients(num_perm, seed)
        self.buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
        self.representative_shingles: dict[str, set[int]] = {}

    def _keys(self, signature: tuple[int, ...]) -> Iterator[tuple[int, tuple[int, ...]]]:
        for band in range(self.bands):
            start = band * self.rows_per_band
            yield band, signature[start : start + self.rows_per_band]

    def find_duplicate(self, document_id: str, text: str) -> str | None:
        shingles = text_shingles(text, self.shingle_size)
        signature = minhash_signature(shingles, self.coefficients)
        possible: set[str] = set()
        keys = list(self._keys(signature))
        for key in keys:
            possible.update(self.buckets.get(key, ()))
        for representative in sorted(possible):
            if jaccard(shingles, self.representative_shingles[representative]) >= self.threshold:
                return representative
        self.representative_shingles[document_id] = shingles
        for key in keys:
            self.buckets[key].append(document_id)
        return None


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(CORE.canonical_json(value).encode("utf-8")).hexdigest()


def load_registry(path: Path) -> dict[str, Any]:
    registry = CORE.load_json(path)
    if registry.get("registry_version") != "p0_source_registry_v1":
        raise ValueError("unsupported source registry")
    if not isinstance(registry.get("sources"), list) or not registry["sources"]:
        raise ValueError("source registry has no sources")
    return registry


def selected_registry_entries(
    registry: dict[str, Any], source_ids: set[str] | None
) -> list[dict[str, Any]]:
    entries = registry["sources"]
    if source_ids is None:
        return entries
    selected = [entry for entry in entries if entry["source_id"] in source_ids]
    missing = sorted(source_ids - {entry["source_id"] for entry in selected})
    if missing:
        raise ValueError(f"unknown source ids: {missing}")
    return selected


def require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit("PyArrow is required for candidate pipeline commands") from error
    return pa, pq


def normalized_arrow_schema():
    pa, _ = require_pyarrow()
    return pa.schema(
        [
            ("document_id", pa.string()),
            ("source_id", pa.string()),
            ("source_revision", pa.string()),
            ("source_locator", pa.string()),
            ("family_id", pa.string()),
            ("domain", pa.string()),
            ("language", pa.string()),
            ("text", pa.large_string()),
            ("raw_sha256", pa.string()),
            ("normalized_sha256", pa.string()),
            ("license_status", pa.string()),
            ("pipeline_version", pa.string()),
            ("split", pa.string()),
            ("rule_keep", pa.bool_()),
            ("rule_reason_codes", pa.list_(pa.string())),
            ("source_metadata_json", pa.large_string()),
        ]
    )


def file_report(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": CORE.sha256_file(path)}


class ParquetShardWriter:
    def __init__(self, directory: Path, prefix: str, rows_per_shard: int, schema) -> None:
        self.directory = directory
        self.prefix = prefix
        self.rows_per_shard = rows_per_shard
        self.schema = schema
        self.buffer: list[dict[str, Any]] = []
        self.shards: list[dict[str, Any]] = []
        self.directory.mkdir(parents=True, exist_ok=True)

    def add(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.rows_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        pa, pq = require_pyarrow()
        path = self.directory / f"{self.prefix}-{len(self.shards):05d}.parquet"
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        pq.write_table(table, path, compression="zstd", use_dictionary=True, write_statistics=True)
        report = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": CORE.sha256_file(path),
        }
        report["rows"] = len(self.buffer)
        self.shards.append(report)
        self.buffer.clear()

    def close(self) -> list[dict[str, Any]]:
        self.flush()
        return self.shards


def atomic_replace_directory(temporary: Path, destination: Path, replace: bool) -> None:
    if destination.exists():
        if not replace:
            raise FileExistsError(f"output already exists: {destination}; pass --replace")
        shutil.rmtree(destination)
    os.replace(temporary, destination)


def command_normalize(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry)
    repo_root = Path(args.repo_root).resolve()
    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()
    pipeline = CORE.load_json(Path(args.pipeline_config))
    CORE.validate_pipeline_config(pipeline)
    registry = load_registry(registry_path)
    source_ids = set(args.source_id) if args.source_id else None
    entries = selected_registry_entries(registry, source_ids)
    summaries = []
    for entry in entries:
        source_id = entry["source_id"]
        if entry.get("adapter_status") != "validated":
            if args.skip_unvalidated:
                summaries.append({"source_id": source_id, "status": "skipped_unvalidated"})
                continue
            raise ValueError(f"{source_id}: adapter_status is not validated")
        destination = output_root / "normalized" / source_id
        existing_manifest_path = destination / "manifest.json"
        if destination.exists() and not args.replace:
            if existing_manifest_path.is_file():
                existing = CORE.load_json(existing_manifest_path)
                source = CORE.load_json(repo_root / entry["source_config"])
                if (
                    existing.get("stage") == "normalize"
                    and existing.get("source_id") == source_id
                    and existing.get("pipeline_config_sha256") == canonical_hash(pipeline)
                    and existing.get("source_adapter_sha256") == canonical_hash(source)
                ):
                    summaries.append(
                        {"source_id": source_id, "status": "skipped_complete", "counts": existing.get("counts", {})}
                    )
                    continue
            raise FileExistsError(
                f"existing normalized output does not match current config: {destination}; pass --replace"
            )
        raw_path = raw_root / entry["raw_relpath"]
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        if entry.get("actual_bytes") is not None and raw_path.stat().st_size != entry["actual_bytes"]:
            raise ValueError(f"{source_id}: raw byte count mismatch")
        if entry.get("sha256") and CORE.sha256_file(raw_path) != entry["sha256"]:
            raise ValueError(f"{source_id}: raw sha256 mismatch")
        source_path = repo_root / entry["source_config"]
        source = CORE.load_json(source_path)
        CORE.validate_source_config(source)
        raw_schema = CORE.parquet_schema(raw_path)
        source_validation = CORE.validate_source_fields(source, raw_schema["fields"])

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{source_id}.tmp-", dir=destination.parent))
        writer = ParquetShardWriter(temporary, "part", args.rows_per_shard, normalized_arrow_schema())
        counters: Counter[str] = Counter()
        try:
            for row_index, row in enumerate(CORE.iter_parquet_rows(raw_path, args.batch_size)):
                counters["input_seen"] += 1
                if not CORE.row_matches_source(row, source):
                    counters["source_filtered"] += 1
                    continue
                record = CORE.make_record(
                    row,
                    row_index,
                    source,
                    pipeline,
                    source_file=raw_path.name,
                )
                counters["seen"] += 1
                counters["kept"] += int(record["rule_keep"])
                counters["dropped"] += int(not record["rule_keep"])
                for reason in record["rule_reason_codes"]:
                    counters[f"drop:{reason}"] += 1
                writer.add(record)
                if args.max_rows and counters["seen"] >= args.max_rows:
                    break
            shards = writer.close()
            manifest = {
                "candidate_pipeline_version": CANDIDATE_PIPELINE_VERSION,
                "implementation": implementation_hashes(),
                "stage": "normalize",
                "run_id": registry["run_id"],
                "source_id": source_id,
                "source_registry_sha256": canonical_hash(registry),
                "pipeline_config_sha256": canonical_hash(pipeline),
                "source_adapter_sha256": canonical_hash(source),
                "raw": file_report(raw_path),
                "source_validation": source_validation,
                "counts": dict(sorted(counters.items())),
                "shards": shards,
            }
            CORE.atomic_write_text(
                temporary / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
            )
            atomic_replace_directory(temporary, destination, args.replace)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        summaries.append(
            {"source_id": source_id, "status": "completed", "counts": dict(counters), "shards": len(shards)}
        )
    print(json.dumps({"stage": "normalize", "sources": summaries}, ensure_ascii=False, indent=2))


def normalized_files(root: Path, entries: list[dict[str, Any]]) -> list[tuple[dict[str, Any], Path]]:
    result = []
    for entry in entries:
        directory = root / "normalized" / entry["source_id"]
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = CORE.load_json(manifest_path)
        for shard in manifest["shards"]:
            result.append((entry, directory / shard["file"]))
    return result


def create_exact_index(database: Path) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE documents (
            normalized_sha256 TEXT PRIMARY KEY,
            sample_rank TEXT NOT NULL,
            document_id TEXT NOT NULL,
            family_id TEXT NOT NULL,
            domain TEXT NOT NULL,
            license_status TEXT NOT NULL,
            parquet_path TEXT NOT NULL,
            row_index INTEGER NOT NULL
        ) WITHOUT ROWID
        """
    )
    connection.execute("CREATE INDEX documents_domain_rank ON documents(domain, sample_rank)")
    connection.execute("CREATE INDEX documents_family ON documents(family_id)")
    return connection


def validate_source_prescreen(rules: Any) -> None:
    """Validate optional, source-specific metadata thresholds before indexing."""
    if not isinstance(rules, dict):
        raise ValueError("source_prescreen must be a mapping of source IDs to rules")
    for source_id, rule in rules.items():
        if not isinstance(source_id, str) or not source_id or not isinstance(rule, dict):
            raise ValueError("source_prescreen needs non-empty source IDs and rule objects")
        if not rule or set(rule) != {"min_score"}:
            raise ValueError(f"{source_id}: only min_score is supported")
        threshold = rule["min_score"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ValueError(f"{source_id}: min_score must be finite numeric")


def passes_source_prescreen(metadata_json: str | None, rule: dict[str, Any]) -> bool:
    """Fail closed on missing, malformed or non-finite source classifier scores."""
    try:
        metadata = json.loads(metadata_json or "")
        score = metadata["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return False
        return math.isfinite(score) and score >= rule["min_score"]
    except (TypeError, ValueError, KeyError):
        return False


def index_normalized_documents(
    files: list[tuple[dict[str, Any], Path]],
    database: Path,
    seed: int,
    allow_pending_license: bool,
    source_prescreen: dict[str, dict[str, Any]] | None = None,
) -> tuple[sqlite3.Connection, dict[str, int]]:
    _, pq = require_pyarrow()
    connection = create_exact_index(database)
    counters: Counter[str] = Counter()
    statement = """
        INSERT INTO documents (
            normalized_sha256, sample_rank, document_id, family_id, domain,
            license_status, parquet_path, row_index
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(normalized_sha256) DO UPDATE SET
            sample_rank=excluded.sample_rank,
            document_id=excluded.document_id,
            family_id=excluded.family_id,
            domain=excluded.domain,
            license_status=excluded.license_status,
            parquet_path=excluded.parquet_path,
            row_index=excluded.row_index
        WHERE excluded.sample_rank < documents.sample_rank
    """
    source_prescreen = source_prescreen or {}
    for entry, path in files:
        source_rule = source_prescreen.get(entry["source_id"])
        table = pq.read_table(
            path,
            columns=[
                "normalized_sha256", "document_id", "family_id", "domain",
                "license_status", "rule_keep",
                *(["source_metadata_json"] if source_rule else []),
            ],
        )
        rows = table.to_pylist()
        pending = []
        for row_index, row in enumerate(rows):
            counters["seen"] += 1
            if not row["rule_keep"]:
                counters["rejected_rule"] += 1
                continue
            license_status = row["license_status"]
            if license_status == "rejected" or (
                license_status != "approved" and not allow_pending_license
            ):
                counters["rejected_license"] += 1
                continue
            if source_rule and not passes_source_prescreen(row["source_metadata_json"], source_rule):
                counters[f"rejected_source_prescreen:{entry['source_id']}"] += 1
                continue
            rank = stable_sample_rank(seed, row["document_id"])
            pending.append(
                (
                    row["normalized_sha256"], rank, row["document_id"], row["family_id"],
                    row["domain"], license_status, str(path), row_index,
                )
            )
        connection.executemany(statement, pending)
        connection.commit()
        counters["rule_and_license_pass"] += len(pending)
    unique = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    counters["exact_unique"] = unique
    counters["exact_duplicates"] = counters["rule_and_license_pass"] - unique
    return connection, dict(sorted(counters.items()))


def fetch_records(locations: list[tuple[str, int, str]]) -> dict[str, dict[str, Any]]:
    """Fetch selected rows efficiently while keeping deterministic rank keys."""
    pa, pq = require_pyarrow()
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for path, row_index, rank in locations:
        grouped[path].append((row_index, rank))
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(grouped):
        selections = grouped[path]
        table = pq.read_table(path)
        indices = [row_index for row_index, _ in selections]
        selected = table.take(pa.array(indices, type=pa.int64())).to_pylist()
        for (_, rank), record in zip(selections, selected):
            result[rank] = record
    return result


def load_tokenizer(path: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise SystemExit("tokenizers is required for exact candidate token counts") from error
    return Tokenizer.from_file(str(path))


def candidate_arrow_schema():
    pa, _ = require_pyarrow()
    return normalized_arrow_schema().append(pa.field("token_count", pa.int64())).append(
        pa.field("sample_rank", pa.string())
    ).append(pa.field("candidate_pipeline_version", pa.string()))


def unified_index_arrow_schema():
    pa, _ = require_pyarrow()
    return pa.schema(
        [
            ("document_id", pa.string()),
            ("normalized_sha256", pa.string()),
            ("source_id", pa.string()),
            ("source_locator", pa.string()),
            ("family_id", pa.string()),
            ("domain", pa.string()),
            ("language", pa.string()),
            ("split", pa.string()),
            ("license_status", pa.string()),
            ("token_count", pa.int64()),
            ("sample_rank", pa.string()),
            ("candidate_relpath", pa.string()),
            ("row_in_shard", pa.int64()),
        ]
    )


def dedup_exclusion_arrow_schema():
    pa, _ = require_pyarrow()
    return pa.schema(
        [
            ("document_id", pa.string()),
            ("normalized_sha256", pa.string()),
            ("source_id", pa.string()),
            ("domain", pa.string()),
            ("sample_rank", pa.string()),
            ("reason", pa.string()),
            ("duplicate_of_document_id", pa.string()),
            ("duplicate_of_source_id", pa.string()),
            ("duplicate_of_domain", pa.string()),
        ]
    )


def candidate_manifest_compatibility(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Return shared provenance or raise before independently built pools mix."""
    if not manifests:
        raise ValueError("at least one candidate manifest is required")
    expected_stage = "build_candidates"
    for manifest in manifests:
        if manifest.get("stage") != expected_stage:
            raise ValueError(f"input stage must be {expected_stage!r}")
    # A registry may be extended between partial builds (for example when a
    # replacement code source is added).  That is safe because every input
    # manifest and row retains its own source identity.  The run, pipeline and
    # tokenizer must remain identical; all registry hashes are preserved.
    shared_paths = ("run_id", "pipeline_config_sha256")
    shared: dict[str, Any] = {}
    for key in shared_paths:
        values = {manifest.get(key) for manifest in manifests}
        if len(values) != 1 or None in values:
            raise ValueError(f"candidate manifests disagree on {key}: {sorted(map(str, values))}")
        shared[key] = next(iter(values))
    registry_hashes = {manifest.get("source_registry_sha256") for manifest in manifests}
    if None in registry_hashes:
        raise ValueError("candidate manifest is missing source_registry_sha256")
    shared["source_registry_sha256_values"] = sorted(registry_hashes)
    tokenizer_hashes = {
        manifest.get("tokenizer", {}).get("tokenizer_json_sha256") for manifest in manifests
    }
    if len(tokenizer_hashes) != 1 or None in tokenizer_hashes:
        raise ValueError("candidate manifests use different tokenizers")
    shared["tokenizer"] = dict(manifests[0]["tokenizer"])
    pending_modes = {
        manifest.get("license_gate", {}).get("allow_pending_for_smoke") for manifest in manifests
    }
    if len(pending_modes) != 1:
        raise ValueError("candidate manifests use different license-gate modes")
    shared["allow_pending_for_smoke"] = next(iter(pending_modes))
    return shared


def record_order_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        record["sample_rank"],
        record["document_id"],
        record["source_id"],
        record["domain"],
    )


def exact_deduplicate_records(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the globally lowest deterministic rank for identical normalized text."""
    retained: dict[str, dict[str, Any]] = {}
    removed: list[dict[str, Any]] = []
    for record in sorted(records, key=record_order_key):
        digest = record["normalized_sha256"]
        previous = retained.get(digest)
        if previous is None:
            retained[digest] = record
            continue
        removed.append({"record": record, "representative": previous, "reason": "exact_duplicate"})
    return sorted(retained.values(), key=record_order_key), removed


def validate_document_and_family_consistency(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    document_fingerprints: dict[str, tuple[str, str, str]] = {}
    family_splits: dict[str, set[str]] = defaultdict(set)
    family_documents: Counter[str] = Counter()
    for record in records:
        fingerprint = (
            record["normalized_sha256"],
            record["family_id"],
            record["split"],
        )
        previous = document_fingerprints.setdefault(record["document_id"], fingerprint)
        if previous != fingerprint:
            raise ValueError(f"conflicting rows for document_id {record['document_id']}")
        family_splits[record["family_id"]].add(record["split"])
        family_documents[record["family_id"]] += 1
    conflicting = sorted(family for family, splits in family_splits.items() if len(splits) != 1)
    if conflicting:
        preview = ", ".join(conflicting[:5])
        raise ValueError(f"family_id appears in multiple splits ({len(conflicting)}): {preview}")
    return {
        "unique_documents": len(document_fingerprints),
        "unique_families": len(family_splits),
        "largest_family_documents": max(family_documents.values(), default=0),
        "split_conflicts": 0,
        "document_id_conflicts": 0,
    }


def select_domain_candidates(
    connection: sqlite3.Connection,
    domain: str,
    token_target: int,
    tokenizer,
    dedup_config: dict[str, Any],
    seed: int,
    writer: ParquetShardWriter,
) -> dict[str, Any]:
    deduper = MinHashDeduper(
        seed=seed,
        shingle_size=dedup_config["shingle_size"],
        num_perm=dedup_config["minhash_num_perm"],
        bands=dedup_config["minhash_bands"],
        threshold=dedup_config["near_jaccard_threshold"],
    )
    batch_rows = dedup_config["selection_batch_rows"]
    cursor = connection.execute(
        """
        SELECT sample_rank, parquet_path, row_index
        FROM documents WHERE domain=? ORDER BY sample_rank
        """,
        (domain,),
    )
    counters: Counter[str] = Counter()
    license_statuses: Counter[str] = Counter()
    near_examples: list[dict[str, str]] = []
    while counters["selected_tokens"] < token_target:
        batch = cursor.fetchmany(batch_rows)
        if not batch:
            break
        records = fetch_records([(path, row_index, rank) for rank, path, row_index in batch])
        for rank, _, _ in batch:
            record = records[rank]
            counters["examined"] += 1
            duplicate_of = deduper.find_duplicate(record["document_id"], record["text"])
            if duplicate_of is not None:
                counters["near_duplicates"] += 1
                if len(near_examples) < 20:
                    near_examples.append(
                        {"document_id": record["document_id"], "duplicate_of": duplicate_of}
                    )
                continue
            token_count = len(tokenizer.encode(record["text"], add_special_tokens=False).ids) + 1
            candidate = dict(record)
            candidate.update(
                {
                    "token_count": token_count,
                    "sample_rank": rank,
                    "candidate_pipeline_version": CANDIDATE_PIPELINE_VERSION,
                }
            )
            writer.add(candidate)
            license_statuses[record["license_status"]] += 1
            counters["selected_documents"] += 1
            counters["selected_tokens"] += token_count
            if counters["selected_tokens"] >= token_target:
                break
    return {
        "domain": domain,
        "target_tokens": token_target,
        "counts": dict(sorted(counters.items())),
        "license_statuses": dict(sorted(license_statuses.items())),
        "near_duplicate_examples": near_examples,
        "target_met": counters["selected_tokens"] >= token_target,
    }


def command_build(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.candidate_dir_name):
        raise ValueError("candidate-dir-name must be a single safe directory name")
    registry = load_registry(Path(args.registry))
    pipeline = CORE.load_json(Path(args.pipeline_config))
    CORE.validate_pipeline_config(pipeline)
    source_prescreen = pipeline.get("source_prescreen", {})
    validate_source_prescreen(source_prescreen)
    data_root = Path(args.data_root).resolve()
    entries = selected_registry_entries(registry, set(args.source_id) if args.source_id else None)
    unknown_rules = sorted(set(source_prescreen) - {entry["source_id"] for entry in registry["sources"]})
    if unknown_rules:
        raise ValueError(f"source_prescreen has unknown source IDs: {unknown_rules}")
    files = normalized_files(data_root, entries)
    if not files:
        raise ValueError("no normalized Parquet shards found")
    tokenizer_path = Path(args.tokenizer_json).resolve()
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer_hash = CORE.sha256_file(tokenizer_path)

    work = data_root / "work"
    connection, index_counts = index_normalized_documents(
        files,
        work / f"{args.candidate_dir_name}_exact_index.sqlite",
        pipeline["pipeline_seed"],
        args.allow_pending_license,
        source_prescreen,
    )
    family_stats = connection.execute(
        "SELECT COUNT(DISTINCT family_id), MAX(n) FROM "
        "(SELECT family_id, COUNT(*) AS n FROM documents GROUP BY family_id)"
    ).fetchone()

    destination = data_root / args.candidate_dir_name
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.candidate_dir_name}.tmp-", dir=data_root))
    domain_reports = []
    all_shards = []
    selected_license_statuses: Counter[str] = Counter()
    try:
        targets = pipeline["candidate_token_targets"]
        available_domains = {
            row[0] for row in connection.execute("SELECT DISTINCT domain FROM documents")
        }
        for domain, target in targets.items():
            if args.domain and domain not in args.domain:
                continue
            if domain not in available_domains:
                domain_reports.append(
                    {"domain": domain, "target_tokens": target, "target_met": False, "status": "no_input"}
                )
                continue
            writer = ParquetShardWriter(
                temporary / domain,
                "part",
                args.rows_per_shard,
                candidate_arrow_schema(),
            )
            report = select_domain_candidates(
                connection,
                domain,
                target,
                tokenizer,
                pipeline["dedup"],
                pipeline["pipeline_seed"],
                writer,
            )
            shards = writer.close()
            selected_license_statuses.update(report.get("license_statuses", {}))
            report["shards"] = shards
            all_shards.extend({**shard, "domain": domain} for shard in shards)
            domain_reports.append(report)
        strict_targets_met = all(
            report.get("target_met", False)
            for report in domain_reports
            if report.get("status") != "no_input" or not args.allow_partial
        )
        if not args.allow_partial and any(not report.get("target_met", False) for report in domain_reports):
            missing = [report["domain"] for report in domain_reports if not report.get("target_met", False)]
            raise ValueError(f"candidate token targets not met: {missing}")
        manifest = {
            "candidate_pipeline_version": CANDIDATE_PIPELINE_VERSION,
            "implementation": implementation_hashes(),
            "stage": "build_candidates",
            "run_id": registry["run_id"],
            "source_registry_sha256": canonical_hash(registry),
            "pipeline_config_sha256": canonical_hash(pipeline),
            "tokenizer": {
                **pipeline["tokenizer"],
                "tokenizer_json": str(tokenizer_path),
                "tokenizer_json_sha256": tokenizer_hash,
            },
            "license_gate": {
                "allow_pending_for_smoke": args.allow_pending_license,
                "statuses": dict(sorted(selected_license_statuses.items())),
                "training_eligible": bool(selected_license_statuses)
                and set(selected_license_statuses) <= {"approved"},
            },
            "index_counts": index_counts,
            "family_stats": {
                "unique_families": family_stats[0] or 0,
                "largest_family_documents": family_stats[1] or 0,
            },
            "domains": domain_reports,
            "targets_met": strict_targets_met,
            "shards": all_shards,
        }
        CORE.atomic_write_text(
            temporary / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        atomic_replace_directory(temporary, destination, args.replace)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        connection.close()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_candidate_pool(candidate_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and checksum one immutable candidate pool declared by its manifest."""
    _, pq = require_pyarrow()
    manifest_path = candidate_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = CORE.load_json(manifest_path)
    records: list[dict[str, Any]] = []
    for shard in manifest.get("shards", []):
        path = candidate_dir / shard["domain"] / shard["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != shard["bytes"]:
            raise ValueError(f"candidate shard byte count mismatch: {path}")
        if CORE.sha256_file(path) != shard["sha256"]:
            raise ValueError(f"candidate shard sha256 mismatch: {path}")
        rows = pq.read_table(path).to_pylist()
        if len(rows) != shard["rows"]:
            raise ValueError(f"candidate shard row count mismatch: {path}")
        records.extend(rows)
    return manifest, records


def exclusion_row(item: dict[str, Any]) -> dict[str, Any]:
    record = item["record"]
    representative = item["representative"]
    return {
        "document_id": record["document_id"],
        "normalized_sha256": record["normalized_sha256"],
        "source_id": record["source_id"],
        "domain": record["domain"],
        "sample_rank": record["sample_rank"],
        "reason": item["reason"],
        "duplicate_of_document_id": representative["document_id"],
        "duplicate_of_source_id": representative["source_id"],
        "duplicate_of_domain": representative["domain"],
    }


def command_consolidate(args: argparse.Namespace) -> None:
    """Merge partial pools and enforce global cross-source invariants."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.output_dir_name):
        raise ValueError("output-dir-name must be a single safe directory name")
    data_root = Path(args.data_root).resolve()
    pipeline = CORE.load_json(Path(args.pipeline_config))
    CORE.validate_pipeline_config(pipeline)
    input_dirs = []
    for value in args.candidate_dir:
        path = Path(value)
        input_dirs.append((path if path.is_absolute() else data_root / path).resolve())
    if len(set(input_dirs)) != len(input_dirs):
        raise ValueError("candidate-dir arguments must be unique")

    manifests: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for candidate_dir in input_dirs:
        manifest, pool_records = load_candidate_pool(candidate_dir)
        manifests.append(manifest)
        records.extend(pool_records)
        inputs.append(
            {
                "directory": str(candidate_dir),
                "manifest_sha256": CORE.sha256_file(candidate_dir / "manifest.json"),
                "documents": len(pool_records),
                "tokens": sum(row["token_count"] for row in pool_records),
            }
        )
    shared = candidate_manifest_compatibility(manifests)
    if shared["pipeline_config_sha256"] != canonical_hash(pipeline):
        raise ValueError("pipeline config does not match candidate manifests")
    consistency_before = validate_document_and_family_consistency(records)

    exact_unique, exact_removed = exact_deduplicate_records(records)
    deduper = MinHashDeduper(
        seed=pipeline["pipeline_seed"],
        shingle_size=pipeline["dedup"]["shingle_size"],
        num_perm=pipeline["dedup"]["minhash_num_perm"],
        bands=pipeline["dedup"]["minhash_bands"],
        threshold=pipeline["dedup"]["near_jaccard_threshold"],
    )
    representatives: dict[str, dict[str, Any]] = {}
    retained: list[dict[str, Any]] = []
    near_removed: list[dict[str, Any]] = []
    for record in exact_unique:
        duplicate_of = deduper.find_duplicate(record["document_id"], record["text"])
        if duplicate_of is None:
            retained.append(record)
            representatives[record["document_id"]] = record
        else:
            near_removed.append(
                {
                    "record": record,
                    "representative": representatives[duplicate_of],
                    "reason": "near_duplicate",
                }
            )
    consistency_after = validate_document_and_family_consistency(retained)

    destination = data_root / args.output_dir_name
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_dir_name}.tmp-", dir=data_root))
    try:
        by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in retained:
            by_domain[record["domain"]].append(record)
        output_shards: list[dict[str, Any]] = []
        index_rows: list[dict[str, Any]] = []
        domain_reports: list[dict[str, Any]] = []
        input_by_domain: Counter[str] = Counter()
        input_tokens_by_domain: Counter[str] = Counter()
        for record in records:
            input_by_domain[record["domain"]] += 1
            input_tokens_by_domain[record["domain"]] += record["token_count"]
        removed_by_domain: Counter[str] = Counter()
        removed_tokens_by_domain: Counter[str] = Counter()
        for item in exact_removed + near_removed:
            removed_by_domain[item["record"]["domain"]] += 1
            removed_tokens_by_domain[item["record"]["domain"]] += item["record"]["token_count"]

        for domain in sorted(by_domain):
            domain_records = sorted(by_domain[domain], key=record_order_key)
            writer = ParquetShardWriter(
                temporary / domain,
                "part",
                args.rows_per_shard,
                candidate_arrow_schema(),
            )
            for position, record in enumerate(domain_records):
                writer.add(record)
                shard_number = position // args.rows_per_shard
                index_rows.append(
                    {
                        "document_id": record["document_id"],
                        "normalized_sha256": record["normalized_sha256"],
                        "source_id": record["source_id"],
                        "source_locator": record["source_locator"],
                        "family_id": record["family_id"],
                        "domain": domain,
                        "language": record["language"],
                        "split": record["split"],
                        "license_status": record["license_status"],
                        "token_count": record["token_count"],
                        "sample_rank": record["sample_rank"],
                        "candidate_relpath": f"{domain}/part-{shard_number:05d}.parquet",
                        "row_in_shard": position % args.rows_per_shard,
                    }
                )
            shards = writer.close()
            output_shards.extend({**shard, "domain": domain} for shard in shards)
            kept_tokens = sum(row["token_count"] for row in domain_records)
            domain_reports.append(
                {
                    "domain": domain,
                    "input_documents": input_by_domain[domain],
                    "input_tokens": input_tokens_by_domain[domain],
                    "removed_documents": removed_by_domain[domain],
                    "removed_tokens": removed_tokens_by_domain[domain],
                    "retained_documents": len(domain_records),
                    "retained_tokens": kept_tokens,
                }
            )

        pa, pq = require_pyarrow()
        index_path = temporary / "unified_index.parquet"
        pq.write_table(
            pa.Table.from_pylist(index_rows, schema=unified_index_arrow_schema()),
            index_path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        exclusion_rows = [exclusion_row(item) for item in exact_removed + near_removed]
        exclusions_path = temporary / "dedup_exclusions.parquet"
        pq.write_table(
            pa.Table.from_pylist(exclusion_rows, schema=dedup_exclusion_arrow_schema()),
            exclusions_path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        cross_source_exact = sum(
            item["record"]["source_id"] != item["representative"]["source_id"]
            for item in exact_removed
        )
        cross_source_near = sum(
            item["record"]["source_id"] != item["representative"]["source_id"]
            for item in near_removed
        )
        cross_domain_removed = sum(
            item["record"]["domain"] != item["representative"]["domain"]
            for item in exact_removed + near_removed
        )
        manifest = {
            "candidate_pipeline_version": CANDIDATE_PIPELINE_VERSION,
            "implementation": implementation_hashes(),
            "stage": "consolidate_candidates",
            "run_id": shared["run_id"],
            "source_registry_sha256_values": shared["source_registry_sha256_values"],
            "pipeline_config_sha256": shared["pipeline_config_sha256"],
            "tokenizer": shared["tokenizer"],
            "license_gate": {
                "allow_pending_for_smoke": shared["allow_pending_for_smoke"],
                "training_eligible": bool(retained)
                and all(record["license_status"] == "approved" for record in retained),
            },
            "inputs": inputs,
            "counts": {
                "input_documents": len(records),
                "input_tokens": sum(record["token_count"] for record in records),
                "exact_unique_documents": len(exact_unique),
                "exact_duplicates_removed": len(exact_removed),
                "near_duplicates_removed": len(near_removed),
                "cross_source_exact_duplicates": cross_source_exact,
                "cross_source_near_duplicates": cross_source_near,
                "cross_domain_duplicates_removed": cross_domain_removed,
                "retained_documents": len(retained),
                "retained_tokens": sum(record["token_count"] for record in retained),
            },
            "family_consistency_before_dedup": consistency_before,
            "family_consistency_after_dedup": consistency_after,
            "domains": domain_reports,
            "unified_index": {
                "file": index_path.name,
                "bytes": index_path.stat().st_size,
                "sha256": CORE.sha256_file(index_path),
                "rows": len(index_rows),
            },
            "dedup_exclusions": {
                "file": exclusions_path.name,
                "bytes": exclusions_path.stat().st_size,
                "sha256": CORE.sha256_file(exclusions_path),
                "rows": len(exclusion_rows),
            },
            "shards": output_shards,
        }
        CORE.atomic_write_text(
            temporary / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        atomic_replace_directory(temporary, destination, args.replace)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser("normalize", help="write canonical normalized Parquet shards")
    normalize.add_argument("--registry", required=True)
    normalize.add_argument("--pipeline-config", required=True)
    normalize.add_argument("--repo-root", default=".")
    normalize.add_argument("--raw-root", required=True)
    normalize.add_argument("--output-root", required=True)
    normalize.add_argument("--source-id", action="append")
    normalize.add_argument("--rows-per-shard", type=int, default=10_000)
    normalize.add_argument("--batch-size", type=int, default=512)
    normalize.add_argument("--max-rows", type=int)
    normalize.add_argument("--skip-unvalidated", action="store_true")
    normalize.add_argument("--replace", action="store_true")
    normalize.set_defaults(func=command_normalize)

    build = subparsers.add_parser("build", help="deduplicate and select token-bounded candidates")
    build.add_argument("--registry", required=True)
    build.add_argument("--pipeline-config", required=True)
    build.add_argument("--data-root", required=True)
    build.add_argument("--tokenizer-json", required=True)
    build.add_argument("--source-id", action="append")
    build.add_argument("--domain", action="append")
    build.add_argument("--rows-per-shard", type=int, default=1_000)
    build.add_argument("--candidate-dir-name", default="candidates")
    build.add_argument("--allow-pending-license", action="store_true")
    build.add_argument("--allow-partial", action="store_true")
    build.add_argument("--replace", action="store_true")
    build.set_defaults(func=command_build)

    consolidate = subparsers.add_parser(
        "consolidate", help="merge partial candidate pools and deduplicate globally"
    )
    consolidate.add_argument("--pipeline-config", required=True)
    consolidate.add_argument("--data-root", required=True)
    consolidate.add_argument("--candidate-dir", action="append", required=True)
    consolidate.add_argument("--output-dir-name", default="candidates_consolidated")
    consolidate.add_argument("--rows-per-shard", type=int, default=1_000)
    consolidate.add_argument("--replace", action="store_true")
    consolidate.set_defaults(func=command_consolidate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
