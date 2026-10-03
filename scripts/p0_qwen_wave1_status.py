#!/usr/bin/env python3
"""Summarize four-card Wave1 rating progress without reading source documents."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.data_dir / "input_shards" / "manifest.json").read_text())
    rows = []
    for index, shard in enumerate(manifest["shards"]):
        path = args.data_dir / "results" / f"shard-{index}.results.jsonl"
        good = set()
        attempted = set()
        errors = prompt = completion = 0
        first = last = None
        if path.exists():
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
                        prompt += result.get("usage", {}).get("prompt_tokens", 0)
                        completion += result.get("usage", {}).get("completion_tokens", 0)
                    elif result.get("status") != "ok":
                        errors += 1
        rows.append({"shard": index, "expected": shard["documents"], "attempted": len(attempted),
                     "ok": len(good), "error_attempts": errors, "prompt_tokens": prompt,
                     "completion_tokens": completion, "first": first, "last": last})
    total = {key: sum(row[key] for row in rows) for key in
             ("expected", "attempted", "ok", "error_attempts", "prompt_tokens", "completion_tokens")}
    total["ok_pct"] = round(100 * total["ok"] / max(1, total["expected"]), 2)
    total_prompt_est = sum(shard["estimated_tokens_with_output_reservation"] for shard in manifest["shards"]) - 384 * total["expected"]
    total["estimated_total_prompt_tokens"] = total_prompt_est
    total["estimated_prompt_work_pct"] = round(100 * total["prompt_tokens"] / max(1, total_prompt_est), 2)
    total["overlong_quarantined"] = manifest["overlong_documents"]
    total["finished_marker"] = (args.data_dir / "FINISHED.json").exists()
    first = min((row["first"] for row in rows if row["first"]), default=None)
    last = max((row["last"] for row in rows if row["last"]), default=None)
    if first and last:
        elapsed = max(1, (datetime.fromisoformat(last) - datetime.fromisoformat(first)).total_seconds())
        total["observed_wall_s"] = round(elapsed)
        total["mean_ok_docs_per_s_so_far"] = round(total["ok"] / elapsed, 3)
        rate = total["prompt_tokens"] / elapsed
        total["mean_prompt_tokens_per_s_so_far"] = round(rate, 1)
        total["rough_eta_hours_from_prompt_rate"] = round(max(0, total_prompt_est-total["prompt_tokens"]) / max(1, rate) / 3600, 1)
    print(json.dumps({"total": total, "shards": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
