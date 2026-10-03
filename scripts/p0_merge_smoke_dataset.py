#!/usr/bin/env python3
"""Merge frozen real and synthetic records into the immutable P0 1M smoke set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p0_candidate_pipeline_for_merge", ROOT / "scripts/p0_candidate_pipeline.py")
CANDIDATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CANDIDATE)
VERSION = "p0_merge_smoke_dataset_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_frozen(directory: Path, expected_stage: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != expected_stage:
        raise ValueError(f"unexpected stage in {manifest_path}: {manifest.get('stage')}")
    rows = []
    for shard in manifest["shards"]:
        path = directory / shard["domain"] / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"input shard hash mismatch: {path}")
        shard_rows = pq.read_table(path).to_pylist()
        if len(shard_rows) != shard["rows"]:
            raise ValueError(f"input shard row mismatch: {path}")
        rows.extend(shard_rows)
    return manifest, rows


def canonical_record(row: dict[str, Any]) -> dict[str, Any]:
    teacher_fields = {
        key: row.get(key)
        for key in (
            "teacher_route", "teacher_status", "teacher_decision", "teacher_confidence",
            "teacher_quality", "teacher_completeness", "teacher_educational_value",
            "teacher_format_integrity", "teacher_reason_codes", "teacher_evidence",
            "teacher_record_sha256",
        )
        if key in row
    }
    source_metadata = row.get("source_metadata_json") or "{}"
    # Assert metadata is valid JSON before preserving its canonical value.
    source_metadata = canonical_json(json.loads(source_metadata))
    return {
        "document_id": row["document_id"],
        "source_id": row["source_id"],
        "source_revision": row["source_revision"],
        "source_locator": row["source_locator"],
        "family_id": row["family_id"],
        "domain": row["domain"],
        "language": row["language"],
        "text": row["text"],
        "normalized_sha256": row["normalized_sha256"],
        "license_status": row["license_status"],
        "split": row["split"],
        "token_count": row["token_count"],
        "sample_rank": row["sample_rank"],
        "source_metadata_json": source_metadata,
        "teacher_metadata_json": canonical_json(teacher_fields),
    }


def merge(real_dir: Path, synthetic_dir: Path, output: Path, seed: int) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    real_manifest, real_rows = load_frozen(real_dir, "accepted_real")
    synthetic_manifest, synthetic_rows = load_frozen(synthetic_dir, "accepted_synthetic")
    tagged = [("real", canonical_record(row)) for row in real_rows]
    tagged.extend(("synthetic", canonical_record(row)) for row in synthetic_rows)
    ids = [row["document_id"] for _, row in tagged]
    hashes = [row["normalized_sha256"] for _, row in tagged]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate document_id across accepted pools")
    if len(set(hashes)) != len(hashes):
        raise ValueError("exact content duplicate across accepted pools")

    deduper = CANDIDATE.MinHashDeduper(seed=seed, shingle_size=5, num_perm=32, bands=8, threshold=0.82)
    near_duplicates = []
    for origin, row in sorted(tagged, key=lambda item: (item[1]["sample_rank"], item[1]["document_id"])):
        duplicate_of = deduper.find_duplicate(row["document_id"], row["text"])
        if duplicate_of:
            near_duplicates.append({"document_id": row["document_id"], "duplicate_of": duplicate_of, "origin": origin})
    if near_duplicates:
        raise ValueError(f"near duplicates remain in accepted pools: {near_duplicates[:3]}")

    family_splits: dict[str, set[str]] = defaultdict(set)
    for _, row in tagged:
        family_splits[row["family_id"]].add(row["split"])
    conflicts = {family: sorted(splits) for family, splits in family_splits.items() if len(splits) > 1}
    if conflicts:
        raise ValueError(f"family split conflicts: {list(conflicts.items())[:3]}")

    output.mkdir(parents=True, exist_ok=True)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, row in tagged:
        by_domain[row["domain"]].append(row)
    shards = []
    for domain, rows in sorted(by_domain.items()):
        rows.sort(key=lambda row: (row["sample_rank"], row["document_id"]))
        directory = output / domain
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "part-00000.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        shards.append(
            {
                "domain": domain, "file": path.name, "rows": len(rows),
                "tokens": sum(row["token_count"] for row in rows),
                "bytes": path.stat().st_size, "sha256": sha256_file(path),
            }
        )
    licenses = Counter(row["license_status"] for _, row in tagged)
    manifest = {
        "version": VERSION,
        "stage": "p0_1m_smoke_frozen",
        "inputs": {
            "accepted_real": {"directory": str(real_dir), "manifest_sha256": sha256_file(real_dir / "manifest.json")},
            "accepted_synthetic": {"directory": str(synthetic_dir), "manifest_sha256": sha256_file(synthetic_dir / "manifest.json")},
        },
        "tokenizer": real_manifest["tokenizer"],
        "counts": {
            "documents": len(tagged),
            "tokens": sum(row["token_count"] for _, row in tagged),
            "real_documents": len(real_rows),
            "real_tokens": sum(row["token_count"] for row in real_rows),
            "synthetic_documents": len(synthetic_rows),
            "synthetic_tokens": sum(row["token_count"] for row in synthetic_rows),
            "exact_duplicates": 0,
            "near_duplicates": 0,
            "families": len(family_splits),
            "family_split_conflicts": 0,
        },
        "splits": dict(sorted(Counter(row["split"] for _, row in tagged).items())),
        "license_gate": {
            "statuses": dict(sorted(licenses.items())),
            "training_eligible": bool(licenses) and set(licenses) <= {"approved", "generated"},
        },
        "shards": shards,
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-real", type=Path, required=True)
    parser.add_argument("--accepted-synthetic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    merge(args.accepted_real.resolve(), args.accepted_synthetic.resolve(), args.output.resolve(), args.seed)


if __name__ == "__main__":
    main()
