#!/usr/bin/env python3
"""Freeze teacher-approved P0 candidates into immutable Parquet shards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any


VERSION = "p0_freeze_accepted_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_candidates(pool: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    manifest_path = pool / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "consolidate_candidates":
        raise ValueError("candidate pool is not consolidated")
    records: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        path = pool / shard["domain"] / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"candidate shard hash mismatch: {path}")
        rows = pq.read_table(path).to_pylist()
        if len(rows) != shard["rows"]:
            raise ValueError(f"candidate shard row mismatch: {path}")
        records.extend(rows)
    return manifest, records


def freeze(
    pool: Path,
    adjudicated_path: Path,
    prefilter_path: Path,
    output: Path,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    source_manifest, candidates = load_candidates(pool)
    candidate_by_id = {row["document_id"]: row for row in candidates}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("duplicate document_id in candidate pool")

    adjudicated_rows = load_jsonl(adjudicated_path)
    adjudicated = {row["document_id"]: row for row in adjudicated_rows}
    if len(adjudicated) != len(adjudicated_rows):
        raise ValueError("duplicate document_id in adjudication")
    prefilter_rows = load_jsonl(prefilter_path)
    prefilter = {row["document_id"]: row for row in prefilter_rows}
    if len(prefilter) != len(prefilter_rows):
        raise ValueError("duplicate document_id in prefilter exclusions")
    if set(adjudicated) & set(prefilter):
        raise ValueError("document appears in both adjudication and prefilter exclusions")
    decided = set(adjudicated) | set(prefilter)
    if decided != set(candidate_by_id):
        missing = sorted(set(candidate_by_id) - decided)
        unknown = sorted(decided - set(candidate_by_id))
        raise ValueError(f"decision partition mismatch: missing={missing[:3]} unknown={unknown[:3]}")

    accepted_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    for document_id, candidate in candidate_by_id.items():
        if document_id in prefilter:
            decision = prefilter[document_id]
            rejected.append(
                {
                    "document_id": document_id,
                    "source_id": candidate["source_id"],
                    "source_locator": candidate["source_locator"],
                    "domain": candidate["domain"],
                    "token_count": candidate["token_count"],
                    "rejection_stage": "cheap_prefilter",
                    "teacher_status": None,
                    "reason_codes": decision["reason_codes"],
                    "decision_record_sha256": sha256_text(canonical_json(decision)),
                }
            )
            continue
        decision = adjudicated[document_id]
        if decision["route"] == "keep":
            final_domain = decision["recommended_bucket"]
            if final_domain == "drop":
                raise ValueError(f"kept document recommends drop: {document_id}")
            enriched = {
                **candidate,
                "original_domain": candidate["domain"],
                "domain": final_domain,
                "teacher_route": decision["route"],
                "teacher_status": decision["teacher_status"],
                "teacher_decision": decision["decision"],
                "teacher_confidence": decision["confidence"],
                "teacher_quality": decision["quality"],
                "teacher_completeness": decision["completeness"],
                "teacher_educational_value": decision["educational_value"],
                "teacher_format_integrity": decision["format_integrity"],
                "teacher_reason_codes": decision["reason_codes"],
                "teacher_evidence": decision["evidence"],
                "teacher_record_sha256": sha256_text(canonical_json(decision)),
            }
            accepted_by_domain[final_domain].append(enriched)
        else:
            rejected.append(
                {
                    "document_id": document_id,
                    "source_id": candidate["source_id"],
                    "source_locator": candidate["source_locator"],
                    "domain": candidate["domain"],
                    "token_count": candidate["token_count"],
                    "rejection_stage": "teacher_gate",
                    "teacher_status": decision["teacher_status"],
                    "reason_codes": decision["reason_codes"],
                    "decision_record_sha256": sha256_text(canonical_json(decision)),
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    shards = []
    accepted_total = 0
    accepted_tokens = 0
    for domain, rows in sorted(accepted_by_domain.items()):
        rows.sort(key=lambda row: row["document_id"])
        domain_dir = output / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        shard_path = domain_dir / "part-00000.parquet"
        pq.write_table(pa.Table.from_pylist(rows), shard_path, compression="zstd")
        tokens = sum(row["token_count"] for row in rows)
        accepted_total += len(rows)
        accepted_tokens += tokens
        shards.append(
            {
                "domain": domain,
                "file": shard_path.name,
                "rows": len(rows),
                "tokens": tokens,
                "bytes": shard_path.stat().st_size,
                "sha256": sha256_file(shard_path),
            }
        )
    rejected.sort(key=lambda row: row["document_id"])
    rejected_path = output / "rejected_audit.parquet"
    pq.write_table(pa.Table.from_pylist(rejected), rejected_path, compression="zstd")

    licenses = Counter(row["license_status"] for rows in accepted_by_domain.values() for row in rows)
    manifest = {
        "freeze_version": VERSION,
        "stage": "accepted_real",
        "source_pool": str(pool.resolve()),
        "inputs": {
            "candidate_manifest": {
                "file": str((pool / "manifest.json").resolve()),
                "sha256": sha256_file(pool / "manifest.json"),
            },
            "adjudicated": {
                "file": str(adjudicated_path.resolve()),
                "sha256": sha256_file(adjudicated_path),
            },
            "prefilter_exclusions": {
                "file": str(prefilter_path.resolve()),
                "sha256": sha256_file(prefilter_path),
            },
        },
        "tokenizer": source_manifest["tokenizer"],
        "counts": {
            "candidate_documents": len(candidates),
            "candidate_tokens": sum(row["token_count"] for row in candidates),
            "accepted_documents": accepted_total,
            "accepted_tokens": accepted_tokens,
            "rejected_documents": len(rejected),
            "rejected_tokens": sum(row["token_count"] for row in rejected),
        },
        "license_gate": {
            "statuses": dict(sorted(licenses.items())),
            "training_eligible": bool(licenses) and set(licenses) <= {"approved"},
        },
        "shards": shards,
        "rejected_audit": {
            "file": rejected_path.name,
            "rows": len(rejected),
            "bytes": rejected_path.stat().st_size,
            "sha256": sha256_file(rejected_path),
            "stages": dict(sorted(Counter(row["rejection_stage"] for row in rejected).items())),
        },
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path, required=True)
    parser.add_argument("--prefilter-exclusions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.pool.resolve(), args.adjudicated.resolve(), args.prefilter_exclusions.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
