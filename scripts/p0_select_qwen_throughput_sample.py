#!/usr/bin/env python3
"""Freeze a disjoint, real-document sample for local Qwen throughput trials."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--count", type=int, default=720)
    ap.add_argument("--min-ref-tokens", type=int, default=100)
    ap.add_argument("--max-ref-tokens", type=int, default=2500)
    ap.add_argument("--seed", default="p0-qwen-l40-throughput-v1")
    args = ap.parse_args()
    if args.output.exists() or args.manifest.exists() or args.count <= 0:
        raise ValueError("output exists or invalid count")

    import pyarrow.parquet as pq

    index_path = args.pool / "unified_index.parquet"
    rows = pq.read_table(index_path, columns=[
        "document_id", "split", "token_count", "candidate_relpath", "row_in_shard"
    ]).to_pylist()
    eligible = [row for row in rows if row["split"] == "train"
                and args.min_ref_tokens <= row["token_count"] <= args.max_ref_tokens]
    eligible.sort(key=lambda row: hashlib.sha256(
        f"{args.seed}\0{row['document_id']}".encode()).digest())
    selected = eligible[:args.count]
    if len(selected) < args.count:
        raise ValueError(f"only {len(selected)} eligible rows")
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_shard[row["candidate_relpath"]].append(row)

    output_by_id = {}
    for relpath, wanted in by_shard.items():
        table = pq.read_table(args.pool / relpath, columns=[
            "document_id", "source_id", "source_locator", "domain", "language", "text", "token_count"
        ])
        for index_row in wanted:
            row = table.slice(index_row["row_in_shard"], 1).to_pylist()[0]
            if row["document_id"] != index_row["document_id"]:
                raise ValueError(f"index/shard ID mismatch: {relpath}")
            envelope = {
                "document_id": row["document_id"],
                "source_id": row["source_id"],
                "source_locator": row["source_locator"],
                "current_bucket": row["domain"],
                "language": row["language"],
                "token_count": row["token_count"],
                "document_text": row["text"],
            }
            output_by_id[row["document_id"]] = {
                "document_id": row["document_id"],
                "current_bucket": row["domain"],
                "text_chars": len(row["text"]),
                "token_count": row["token_count"],
                "review_payload": json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(output_by_id[row["document_id"]], ensure_ascii=False, sort_keys=True) + "\n"
                      for row in selected)
    args.output.write_text(content, encoding="utf-8")
    report = {
        "stage": "throughput_probe_not_training_approved",
        "seed": args.seed,
        "pool_manifest_sha256": sha256_file(args.pool / "manifest.json"),
        "index_sha256": sha256_file(index_path),
        "input_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "eligible_documents": len(eligible),
        "selected_documents": len(selected),
        "min_ref_tokens": args.min_ref_tokens,
        "max_ref_tokens": args.max_ref_tokens,
        "by_domain": dict(sorted(Counter(output_by_id[r["document_id"]]["current_bucket"] for r in selected).items())),
    }
    args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
