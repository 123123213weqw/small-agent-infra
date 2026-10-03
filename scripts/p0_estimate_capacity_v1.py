#!/usr/bin/env python3
"""Estimate own-tokenizer capacity of normalized sources from spaced shard samples.

This is an order-of-magnitude planning audit, not an exact training-token count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--normalized-root", type=Path, required=True)
    p.add_argument("--tokenizer-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shards-per-source", type=int, default=10)
    p.add_argument("--rows-per-shard", type=int, default=30)
    args = p.parse_args()
    if args.output.exists() or args.shards_per_source < 1 or args.rows_per_shard < 1:
        raise ValueError("output exists or sample sizes are invalid")

    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    results = []
    for source_dir in sorted(args.normalized_root.iterdir()):
        manifest_path = source_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards = sorted(source_dir.glob("part-*.parquet"))
        if not shards:
            raise ValueError(f"missing shards for {source_dir.name}")
        indices = sorted({min(len(shards) - 1, int((i + 0.5) * len(shards) / min(len(shards), args.shards_per_source)))
                          for i in range(min(len(shards), args.shards_per_source))})
        sampled_tokens = []
        sampled_chars = []
        for index in indices:
            shard = shards[index]
            table = pq.read_table(shard, columns=["document_id", "text", "rule_keep"])
            rows = [row for row in table.to_pylist() if row["rule_keep"] and row["text"]]
            rng = random.Random(f"p0-500m-capacity-v1:{source_dir.name}:{shard.name}")
            selected = rng.sample(rows, min(len(rows), args.rows_per_shard))
            for row in selected:
                sampled_tokens.append(len(tokenizer.encode(row["text"], add_special_tokens=False).ids))
                sampled_chars.append(len(row["text"]))
        if not sampled_tokens:
            raise ValueError(f"no kept rows sampled for {source_dir.name}")
        kept = manifest["counts"]["kept"]
        average = statistics.mean(sampled_tokens)
        results.append({
            "source": source_dir.name,
            "normalized_kept_documents": kept,
            "sampled_documents": len(sampled_tokens),
            "mean_own_tokens": round(average, 2),
            "median_own_tokens": statistics.median(sampled_tokens),
            "estimated_own_tokens_before_quality_and_dedup": round(kept * average),
            "mean_chars": round(statistics.mean(sampled_chars), 2),
            "sampled_shards": [shards[i].name for i in indices],
        })
    report = {
        "stage": "capacity_estimate_sample_not_accepted_training_tokens",
        "method": "spaced_shards_seeded_rows",
        "shards_per_source": args.shards_per_source,
        "rows_per_shard": args.rows_per_shard,
        "total_estimated_own_tokens_before_quality_and_dedup": sum(
            row["estimated_own_tokens_before_quality_and_dedup"] for row in results
        ),
        "sources": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
