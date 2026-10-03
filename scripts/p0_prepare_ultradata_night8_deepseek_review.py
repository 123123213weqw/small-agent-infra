#!/usr/bin/env python3
"""Blind review sample from Qwen-kept night8 records: 40 random + 10 flagged/source."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random


SOURCES = ("ultradata_code_l2_py", "ultradata_math_l2_preview",
           "ultrafineweb_en", "ultrafineweb_zh")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--night-root", type=Path, required=True)
    ap.add_argument("--postfilter-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError(args.output)
    summary = json.loads((args.postfilter_dir / "summary.json").read_text())
    if summary["qwen_keep"] != 80409:
        raise ValueError("postfilter summary changed")
    pools = defaultdict(list)
    for row in read_jsonl(args.postfilter_dir / "postfilter_index.jsonl"):
        if row["postfilter_status"] == "exact_duplicate":
            continue
        pools[row["source_id"]].append(row)
    rng = random.Random(args.seed)
    selections = []
    for source in SOURCES:
        pool = sorted(pools[source], key=lambda r: r["document_id"])
        random_rows = rng.sample(pool, 40)
        random_ids = {row["document_id"] for row in random_rows}
        flagged_pool = [row for row in pool if row["quality_flags"] and row["document_id"] not in random_ids]
        if len(flagged_pool) < 10:
            raise ValueError(f"too few flagged records for {source}")
        flagged_rows = rng.sample(flagged_pool, 10)
        selections.extend((source + "_random_keep", row) for row in random_rows)
        selections.extend((source + "_targeted_flag", row) for row in flagged_rows)
    selected = {row["document_id"]: (stratum, row) for stratum, row in selections}
    if len(selected) != 200:
        raise ValueError("review sample identity collision")
    blind_rows = {}
    for band in ("heavy", "medium", "light"):
        for shard in range(8):
            for row in read_jsonl(args.night_root / "inputs" / band / f"shard-{shard}.input.jsonl"):
                doc_id = row["document_id"]
                if doc_id in selected:
                    blind_rows[doc_id] = {key: row[key] for key in (
                        "document_id", "source_id", "current_bucket", "raw_row",
                        "source_path", "upstream_score", "text_chars", "prompt_tokens_est", "review_payload")}
    if set(blind_rows) != set(selected):
        raise ValueError("selected audit records not found in input shards")
    order = sorted(selected)
    rng.shuffle(order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for doc_id in order:
            handle.write(json.dumps(blind_rows[doc_id], ensure_ascii=False, sort_keys=True) + "\n")
    strata = Counter(stratum for stratum, _ in selections)
    manifest = {"stage": "blind_qwen_keep_review_not_training_approval",
                "seed": args.seed, "input_sha256": sha256(args.output),
                "postfilter_index_sha256": sha256(args.postfilter_dir / "postfilter_index.jsonl"),
                "selection": {"random_keep_per_source": 40, "targeted_flag_per_source": 10},
                "strata": dict(sorted(strata.items())),
                "items": [{"document_id": doc_id, "source_id": selected[doc_id][1]["source_id"],
                           "selection_stratum": selected[doc_id][0],
                           "postfilter_status": selected[doc_id][1]["postfilter_status"],
                           "quality_flags": selected[doc_id][1]["quality_flags"],
                           "qwen_quality": selected[doc_id][1]["qwen_quality"],
                           "qwen_confidence": selected[doc_id][1]["qwen_confidence"]}
                          for doc_id in order]}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "selected": len(order),
                      "strata": dict(sorted(strata.items())), "sha256": manifest["input_sha256"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
