#!/usr/bin/env python3
"""Aggregate B2.3-D hard relation distractor results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def mean(items, key):
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_dir.glob("seed*_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if len(rows) != 36:
        raise RuntimeError(f"incomplete hard-relation sweep: {len(rows)}/36")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["variant"], int(row["decoy_count"]))].append(row)
    metrics = (
        "success",
        "required_recall",
        "critical_error_rate",
        "mean_retained_decoys",
        "required_decoy_auc",
        "mean_safe_margin",
        "positive_safe_margin_rate",
        "oracle_success",
        "fifo_success",
    )
    summary = {"rows": len(rows), "conditions": {}, "pass": {}}
    for (variant, count), items in sorted(groups.items()):
        summary["conditions"][f"{variant}/K{count}"] = {
            "seed_success": {str(item["model_seed"]): item["success"] for item in items},
            **{metric: mean(items, metric) for metric in metrics},
        }
    baseline = [row for row in rows if int(row["decoy_count"]) == 0]
    d1_64 = [row for row in rows if row["variant"] == "disconnected" and int(row["decoy_count"]) == 64]
    d2_64 = [row for row in rows if row["variant"] == "spoke" and int(row["decoy_count"]) == 64]
    summary["pass"] = {
        "K0_all_seeds_one": min(float(row["success"]) for row in baseline) == 1.0,
        "oracle_all_one": min(float(row["oracle_success"]) for row in rows) == 1.0,
        "D1_K64_each_seed_at_least_0.95": min(float(row["success"]) for row in d1_64) >= 0.95,
        "D2_K64_each_seed_at_least_0.90": min(float(row["success"]) for row in d2_64) >= 0.90,
        "K64_required_recall_at_least_0.98": min(float(row["required_recall"]) for row in [*d1_64, *d2_64]) >= 0.98,
    }
    (args.input_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
