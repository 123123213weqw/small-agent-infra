#!/usr/bin/env python3
"""Estimate own-tokenizer raw capacity from spaced row groups; no quality claim."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    plan_path = args.root / "download_plan_v1.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    results = []
    by_source = defaultdict(lambda: {"files": 0, "rows": 0, "bytes": 0, "sampled_rows": 0,
                                     "sampled_own_tokens": 0, "estimated_own_tokens": 0})
    for item in plan["items"]:
        path = args.root / "raw" / item["source_id"] / Path(item["path"]).name
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"size mismatch: {path}")
        pf = pq.ParquetFile(path)
        groups = pf.num_row_groups
        group_ids = sorted({min(groups - 1, int((i + 0.5) * groups / 12)) for i in range(12)})
        field = "code" if item["source_id"] == "github_code_clean_more" else "text"
        sample_tokens = 0
        sample_rows = 0
        source_seed = int(hashlib.sha256(item["path"].encode()).hexdigest(), 16)
        rng = random.Random(source_seed)
        for group_id in group_ids:
            table = pf.read_row_group(group_id, columns=[field])
            positions = rng.sample(range(table.num_rows), min(25, table.num_rows))
            texts = [table[field][i].as_py() or "" for i in positions]
            sample_rows += len(texts)
            sample_tokens += sum(len(encoded.ids) for encoded in tokenizer.encode_batch(texts, add_special_tokens=False))
        estimate = round(pf.metadata.num_rows * sample_tokens / sample_rows)
        report = {"source_id": item["source_id"], "file": path.name, "rows": pf.metadata.num_rows,
                  "bytes": path.stat().st_size, "sampled_rows": sample_rows,
                  "sampled_own_tokens": sample_tokens, "estimated_own_tokens": estimate}
        results.append(report)
        target = by_source[item["source_id"]]
        target["files"] += 1
        target["rows"] += report["rows"]
        target["bytes"] += report["bytes"]
        target["sampled_rows"] += sample_rows
        target["sampled_own_tokens"] += sample_tokens
        target["estimated_own_tokens"] += estimate
    report = {"stage": "raw_spaced_rowgroup_token_estimate_not_quality_or_dedup_approved",
              "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
              "sampling": "12 spaced row groups per file, 25 seeded rows per group",
              "note": "Code estimate includes every language/license. Source subsets overlap previously downloaded data.",
              "by_source": dict(by_source), "files": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": report["stage"], "by_source": report["by_source"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
