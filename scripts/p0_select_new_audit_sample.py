#!/usr/bin/env python3
"""Select a deterministic blind sample of previously unjudged P0 candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


DOMAINS = ("general_zh", "general_en", "math_en", "code_python", "code_shell", "code_other")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--old-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per-domain", type=int, default=30)
    parser.add_argument("--min-chars", type=int, default=400)
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--seed", default="p0-500m-new-audit-180-v1")
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists() or args.per_domain < 1:
        raise ValueError("output exists or invalid sample size")

    import pyarrow.parquet as pq

    old_manifest = json.loads((args.old_pool / "manifest.json").read_text(encoding="utf-8"))
    old_index = args.old_pool / old_manifest["unified_index"]["file"]
    if sha256_file(old_index) != old_manifest["unified_index"]["sha256"]:
        raise ValueError("old pool index checksum mismatch")
    old_ids = set(pq.read_table(old_index, columns=["document_id"])["document_id"].to_pylist())

    manifest_path = args.pool / "manifest.json"
    pool = json.loads(manifest_path.read_text(encoding="utf-8"))
    if pool.get("stage") != "consolidate_candidates":
        raise ValueError("pool must be a consolidated candidate corpus")
    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    eligible_counts = defaultdict(int)
    for shard in pool["shards"]:
        path = args.pool / shard["domain"] / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"shard checksum mismatch: {path}")
        for row in pq.read_table(path, columns=[
            "document_id", "domain", "source_id", "source_locator", "language", "text", "token_count"
        ]).to_pylist():
            if row["document_id"] in old_ids:
                continue
            if not args.min_chars <= len(row["text"]) <= args.max_chars:
                continue
            eligible_counts[row["domain"]] += 1
            rank = hashlib.sha256(f"{args.seed}\0{row['document_id']}".encode()).hexdigest()
            groups[row["domain"]].append((rank, row))

    selected = []
    for domain in DOMAINS:
        ranked = sorted(groups[domain], key=lambda item: item[0])
        if len(ranked) < args.per_domain:
            raise ValueError(f"{domain}: only {len(ranked)} eligible rows")
        for _, row in ranked[:args.per_domain]:
            envelope = {
                "document_id": row["document_id"],
                "source_id": row["source_id"],
                "source_locator": row["source_locator"],
                "current_bucket": row["domain"],
                "language": row["language"],
                "token_count": row["token_count"],
                "document_text": row["text"],
            }
            selected.append({
                "document_id": row["document_id"],
                "current_bucket": domain,
                "text_chars": len(row["text"]),
                "review_payload": json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected)
    args.output.write_text(content, encoding="utf-8")
    report = {
        "stage": "blind_new_candidate_calibration_sample_not_training_approved",
        "source_pool_manifest_sha256": sha256_file(manifest_path),
        "old_pool_index_sha256": sha256_file(old_index),
        "seed": args.seed,
        "per_domain": args.per_domain,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "eligible_counts": dict(sorted(eligible_counts.items())),
        "selected_documents": len(selected),
        "input_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
    args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
