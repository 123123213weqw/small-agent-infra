#!/usr/bin/env python3
"""Write a frozen mixture proportional to accepted train tokens per domain."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.source / "manifest.json").read_text())
    if manifest.get("stage") != "accepted_real" or not manifest["license_gate"]["training_eligible"]:
        raise ValueError("source is not training-approved real data")
    # Use the frozen per-record index rather than all-split domain totals.
    totals = {}
    index = args.source / "combined_index.jsonl"
    if not index.exists():
        index = args.source / "accepted_index.jsonl"
    with index.open() as f:
        for line in f:
            row = json.loads(line)
            if row["split"] == "train":
                totals[row["domain"]] = totals.get(row["domain"], 0) + row["own_text_tokens"]
    denominator = sum(totals.values())
    if denominator <= 0 or args.output.exists():
        raise ValueError("empty train split or mixture output exists")
    args.output.write_text(json.dumps({domain: n / denominator for domain, n in sorted(totals.items())},
                                      indent=2, ensure_ascii=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
