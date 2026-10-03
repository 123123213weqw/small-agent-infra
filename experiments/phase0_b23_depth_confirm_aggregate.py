#!/usr/bin/env python3
"""Aggregate the corrected B2.3-C depth-only confirmation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def mean(items, key):
    return sum(float(item[key]) for item in items) / len(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_dir.glob("seed*_depth.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if len(rows) != 30:
        raise RuntimeError(f"incomplete corrected depth scan: {len(rows)}/30")
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["length"]), int(row["hops"]))].append(row)
    summary = {"rows": len(rows), "conditions": {}, "pass": {}}
    for (length, hops), items in sorted(groups.items()):
        summary["conditions"][f"length{length}/hops{hops}"] = {
            "seed_success": {str(item["model_seed"]): item["success"] for item in items},
            "mean_success": mean(items, "success"),
            "mean_required_recall": mean(items, "required_recall"),
            "mean_critical_error_rate": mean(items, "critical_error_rate"),
            "oracle_success": mean(items, "oracle_success"),
        }
    two = [row for row in rows if int(row["hops"]) == 2]
    four = [row for row in rows if int(row["hops"]) == 4]
    eight = [row for row in rows if int(row["hops"]) == 8]
    summary["pass"] = {
        "two_hop_control_all_one": min(float(row["success"]) for row in two) == 1.0,
        "four_hop_mean_at_least_0.90": mean(four, "success") >= 0.90,
        "eight_hop_mean_at_least_0.80": mean(eight, "success") >= 0.80,
        "oracle_all_one": min(float(row["oracle_success"]) for row in rows) == 1.0,
    }
    (args.input_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
