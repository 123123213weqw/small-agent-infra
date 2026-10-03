#!/usr/bin/env python3
"""Aggregate the six B2.4 model comparisons."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def mean(values):
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_dir.glob("*/hard_relations.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if len(rows) != 72:
        raise RuntimeError(f"incomplete B2.4 evaluation: {len(rows)}/72")
    groups = defaultdict(list)
    for row in rows:
        groups[row["name"].rsplit("_seed", 1)[0]].append(row)
    summary = {"rows": len(rows), "models": {}, "winner": None}
    ranking = []
    for model_name, items in sorted(groups.items()):
        k0 = [item for item in items if int(item["decoy_count"]) == 0]
        k16 = [item for item in items if int(item["decoy_count"]) == 16]
        k64 = [item for item in items if int(item["decoy_count"]) == 64]
        result = {
            "parameters": sorted(set(int(item["parameters"]) for item in items)),
            "K0_success": mean([float(item["success"]) for item in k0]),
            "K16_success": mean([float(item["success"]) for item in k16]),
            "K64_success": mean([float(item["success"]) for item in k64]),
            "K64_required_recall": mean([float(item["required_recall"]) for item in k64]),
            "K64_auc": mean([float(item["required_decoy_auc"]) for item in k64]),
        }
        summary["models"][model_name] = result
        ranking.append((result["K64_success"], result["K64_required_recall"], -result["parameters"][0], model_name))
    summary["winner"] = max(ranking)[-1]
    (args.input_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
