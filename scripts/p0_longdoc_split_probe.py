#!/usr/bin/env python3
"""Offline structural split pilot for Wave1 overlong records; never approves data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics


def split_estimates(text: str, reference_tokens: int, limit: int) -> list[int] | None:
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) < 2 or not text:
        return None
    # Only a structural estimate. No tokenizer or downstream quality judgment.
    weights = [max(1, round(reference_tokens * len(p) / len(text))) for p in paragraphs]
    if max(weights) > limit:
        return None
    chunks: list[int] = []
    current = 0
    for weight in weights:
        if current and current + weight > limit:
            chunks.append(current)
            current = 0
        current += weight
    if current:
        chunks.append(current)
    return chunks if len(chunks) >= 2 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--overlong", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-domain", type=int, default=50)
    parser.add_argument("--chunk-limit", type=int, default=2500)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import pyarrow.parquet as pq

    by_domain: dict[str, list[dict]] = defaultdict(list)
    for line in args.overlong.open():
        if line.strip():
            record = json.loads(line)
            by_domain[record["current_bucket"]].append(record)
    selected: dict[str, dict[int, dict]] = defaultdict(dict)
    for domain, rows in by_domain.items():
        rows.sort(key=lambda r: hashlib.sha256(r["document_id"].encode()).digest())
        for row in rows[:args.per_domain]:
            selected[row["candidate_relpath"]][row["row_in_shard"]] = row

    stats: dict[str, Counter] = defaultdict(Counter)
    chunk_lengths: dict[str, list[int]] = defaultdict(list)
    examples: dict[str, list[dict]] = defaultdict(list)
    for relpath, targets in selected.items():
        parquet = pq.ParquetFile(args.pool / relpath)
        offset = 0
        found: set[int] = set()
        for batch in parquet.iter_batches(batch_size=256, columns=["document_id", "text", "token_count"]):
            for j, row in enumerate(batch.to_pylist()):
                ix = offset + j
                if ix not in targets:
                    continue
                target = targets[ix]
                if row["document_id"] != target["document_id"]:
                    raise ValueError(f"shard index mismatch: {relpath}:{ix}")
                found.add(ix)
                domain = target["current_bucket"]
                tokens = row["token_count"]
                chunks = split_estimates(row["text"], tokens, args.chunk_limit)
                stats[domain]["sample_documents"] += 1
                stats[domain]["sample_reference_tokens"] += tokens
                if chunks is not None:
                    stats[domain]["structurally_splittable_documents"] += 1
                    stats[domain]["structurally_splittable_reference_tokens"] += tokens
                    stats[domain]["estimated_chunks"] += len(chunks)
                    chunk_lengths[domain].extend(chunks)
                if len(examples[domain]) < 6:
                    examples[domain].append({"document_id": row["document_id"],
                                             "reference_tokens": tokens,
                                             "estimated_chunks": chunks[:12] if chunks else None})
            offset += batch.num_rows
        if found != set(targets):
            raise ValueError(f"missing shard rows: {relpath}: {set(targets) - found}")
    report = {
        "stage": "wave1_overlong_structural_probe_not_quality_approval",
        "method": "hash sample per bucket; split on blank lines; estimate token lengths from character share; no content judgment",
        "overlong_population": {key: len(rows) for key, rows in by_domain.items()},
        "chunk_limit_estimated_reference_tokens": args.chunk_limit,
        "by_domain": {domain: {**dict(count),
                               "median_estimated_chunk_tokens": statistics.median(chunk_lengths[domain])
                               if chunk_lengths[domain] else None,
                               "examples": examples[domain]}
                      for domain, count in sorted(stats.items())},
        "caveat": "Splittable is not usable: cuts may lose context; token lengths are approximated; Qwen/DeepSeek not rerun.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"overlong_population": report["overlong_population"],
                      "by_domain": {d: {k: v for k, v in x.items() if k != "examples"}
                                    for d, x in report["by_domain"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
