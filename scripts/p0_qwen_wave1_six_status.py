#!/usr/bin/env python3
"""Read-only progress summary for six-card, length-banded Wave1 rating."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.data_dir / "input_bands6_v1" / "manifest.json").read_text())
    bands = []
    first = last = None
    for band in manifest["bands"]:
        summary = {"name": band["name"], "expected": band["documents"],
                   "expected_prompt_tokens": band["prompt_tokens"], "attempted": 0,
                   "ok": 0, "error_attempts": 0, "prompt_tokens": 0, "completion_tokens": 0}
        for index in range(len(band["shards"])):
            path = args.data_dir / "results6" / band["name"] / f"shard-{index}.results.jsonl"
            attempted: set[str] = set()
            good: set[str] = set()
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    result = json.loads(line)
                    doc_id = json.loads(result["source_id"])[0]
                    attempted.add(doc_id)
                    stamp = result.get("generated_at")
                    if stamp:
                        first = min(first, stamp) if first else stamp
                        last = max(last, stamp) if last else stamp
                    if result.get("status") == "ok" and doc_id not in good:
                        good.add(doc_id)
                        summary["prompt_tokens"] += result.get("usage", {}).get("prompt_tokens", 0)
                        summary["completion_tokens"] += result.get("usage", {}).get("completion_tokens", 0)
                    elif result.get("status") != "ok":
                        summary["error_attempts"] += 1
            summary["attempted"] += len(attempted)
            summary["ok"] += len(good)
        summary["ok_pct"] = round(100 * summary["ok"] / max(1, summary["expected"]), 2)
        summary["prompt_work_pct"] = round(100 * summary["prompt_tokens"] / max(1, summary["expected_prompt_tokens"]), 2)
        bands.append(summary)
    total = {key: sum(b[key] for b in bands) for key in
             ("expected", "expected_prompt_tokens", "attempted", "ok", "error_attempts", "prompt_tokens", "completion_tokens")}
    total["ok_pct"] = round(100 * total["ok"] / max(1, total["expected"]), 2)
    total["prompt_work_pct"] = round(100 * total["prompt_tokens"] / max(1, total["expected_prompt_tokens"]), 2)
    source_manifest = json.loads((args.data_dir / "input_shards" / "manifest.json").read_text())
    total["overlong_quarantined"] = source_manifest["overlong_documents"]
    total["finished"] = (args.data_dir / "FINISHED6.json").exists()
    if first and last:
        elapsed = max(1, (datetime.fromisoformat(last) - datetime.fromisoformat(first)).total_seconds())
        rate = total["prompt_tokens"] / elapsed
        total["observed_wall_s"] = round(elapsed)
        total["mean_prompt_tokens_per_s_so_far"] = round(rate, 1)
        total["rough_eta_hours_from_prompt_rate"] = round(max(0, total["expected_prompt_tokens"]-total["prompt_tokens"]) / max(1, rate) / 3600, 1)
    print(json.dumps({"total": total, "bands": bands}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
