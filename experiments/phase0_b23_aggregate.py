#!/usr/bin/env python3
"""Aggregate compact B2.3 stress shards."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_dir.glob("shard_*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not rows:
        raise RuntimeError("no B2.3 shard rows found")
    expected = 3 * 4 * 4 * 4 * 4
    if len(rows) != expected:
        raise RuntimeError(f"incomplete B2.3 sweep: {len(rows)}/{expected} rows")

    by_seed = defaultdict(list)
    for row in rows:
        by_seed[int(row["model_seed"])].append(row)
    summary = {"rows": len(rows), "expected_rows": expected, "seeds": {}}
    for seed, items in sorted(by_seed.items()):
        feasible = [item for item in items if item["feasible_capacity"]]
        worst = min(feasible, key=lambda item: (item["success"], item["required_recall"]))
        summary["seeds"][str(seed)] = {
            "mean_feasible_success": sum(item["success"] for item in feasible) / len(feasible),
            "mean_feasible_required_recall": sum(item["required_recall"] for item in feasible) / len(feasible),
            "worst_feasible_condition": worst,
        }
    (args.input_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["model_seed", "length", "capacity", "hops", "hard_edges", "feasible_capacity", "success", "required_recall", "decision_accuracy", "mean_regret", "episodes_with_required_eviction", "median_first_required_eviction", "seconds"]
    with (args.input_dir / "all_conditions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
