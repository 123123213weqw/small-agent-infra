#!/usr/bin/env python3
"""Aggregate B2.3-B/C axis scans."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def average(items, key):
    return sum(float(item[key]) for item in items) / len(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_dir.glob("seed*_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    expected = 3 * (len((128, 256, 512)) * len((4, 8, 12, 16, 24, 32)) + len((128, 256)) * len((2, 3, 4, 6, 8)))
    if len(rows) != expected:
        raise RuntimeError(f"incomplete axis scans: {len(rows)}/{expected}")

    metrics = ("success", "required_recall", "critical_error_rate", "oracle_success", "fifo_success")
    summary = {"rows": len(rows), "expected_rows": expected, "capacity": {}, "depth": {}, "pass": {}}
    capacity_groups = defaultdict(list)
    depth_groups = defaultdict(list)
    for row in rows:
        if row["axis"] == "capacity":
            capacity_groups[(int(row["length"]), int(row["capacity"]))].append(row)
        else:
            depth_groups[(int(row["length"]), int(row["hops"]))].append(row)
    for key, items in sorted(capacity_groups.items()):
        summary["capacity"][f"length{key[0]}/S{key[1]}"] = {metric: average(items, metric) for metric in metrics}
    for key, items in sorted(depth_groups.items()):
        summary["depth"][f"length{key[0]}/hops{key[1]}"] = {metric: average(items, metric) for metric in metrics}

    target_capacity = [row for row in rows if row["axis"] == "capacity" and int(row["capacity"]) == 32]
    depth_four = [row for row in rows if row["axis"] == "depth" and int(row["hops"]) == 4]
    depth_eight = [row for row in rows if row["axis"] == "depth" and int(row["hops"]) == 8]
    summary["pass"] = {
        "capacity_S32_all_seeds_at_least_0.95": min(float(row["success"]) for row in target_capacity) >= 0.95,
        "depth_4hop_mean_at_least_0.90": average(depth_four, "success") >= 0.90,
        "depth_8hop_mean_at_least_0.80": average(depth_eight, "success") >= 0.80,
        "oracle_all_one": min(float(row["oracle_success"]) for row in rows) == 1.0,
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
