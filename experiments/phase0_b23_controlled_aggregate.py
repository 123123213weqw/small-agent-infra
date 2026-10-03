#!/usr/bin/env python3
"""Aggregate the paired B2.3 controlled boundary check."""

from __future__ import annotations

import argparse
import csv
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
    for path in sorted(args.input_dir.glob("seed*_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    expected = 3 * 2 * 7 * 3
    if len(rows) != expected:
        raise RuntimeError(f"incomplete controlled sweep: {len(rows)}/{expected}")

    summary = {"rows": len(rows), "expected_rows": expected, "by_variant_seed": {}, "by_variant_length": {}}
    groups = defaultdict(list)
    for row in rows:
        groups[(row["variant"], int(row["model_seed"]))].append(row)
    for (variant, seed), items in sorted(groups.items()):
        summary["by_variant_seed"][f"{variant}/seed{seed}"] = {
            "mean_success": mean(items, "success"),
            "mean_required_recall": mean(items, "required_recall"),
            "mean_critical_error_rate": mean(items, "critical_error_rate"),
            "oracle_success": mean(items, "oracle_success"),
            "fifo_success": mean(items, "fifo_success"),
        }
    length_groups = defaultdict(list)
    for row in rows:
        length_groups[(row["variant"], int(row["length"]))].append(row)
    for (variant, length), items in sorted(length_groups.items()):
        summary["by_variant_length"][f"{variant}/length{length}"] = {
            "mean_success": mean(items, "success"),
            "mean_required_recall": mean(items, "required_recall"),
            "mean_critical_error_rate": mean(items, "critical_error_rate"),
        }
    (args.input_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = list(rows[0])
    with (args.input_dir / "all_conditions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
