#!/usr/bin/env python3
"""Blind, stratified DeepSeek review sample from completed Qwen source probe."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


SOURCES = (
    "ultradata_code_l2_py", "ultradata_math_l2_preview",
    "ultrafineweb_en", "ultrafineweb_zh",
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def suspicious_code(row: dict) -> bool:
    payload = json.loads(row["review_payload"])
    text = payload["document_text"]
    locator = str(payload.get("source_locator", "")).lower()
    flat = len(text) >= 500 and text.count("\n") <= 2
    path_risk = any(part in locator for part in (
        "/vendor/", "/dist/", "/build/", "/generated/", "/archive/",
        ".min.js", "site-packages", "/node_modules/",
    ))
    return flat or path_risk


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260925)
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise ValueError("refusing to overwrite review sample")
    if not (args.root / "FINISHED6.json").exists():
        raise ValueError("source probe has not finished")
    inputs = {}
    labels = {}
    for band in ("heavy", "medium", "light"):
        for shard in range(6):
            for row in read_jsonl(args.root / "inputs" / band / f"shard-{shard}.input.jsonl"):
                doc_id = row["document_id"]
                if doc_id in inputs:
                    raise ValueError("duplicate input document ID")
                inputs[doc_id] = row
            for result in read_jsonl(args.root / "results" / band / f"shard-{shard}.results.jsonl"):
                if result.get("status") != "ok":
                    continue
                doc_id, = json.loads(result["source_id"])
                if doc_id in labels:
                    raise ValueError("duplicate successful result")
                labels[doc_id] = result["annotation"]

    pools = {(source, decision): [] for source in SOURCES for decision in ("keep", "drop")}
    for doc_id, label in labels.items():
        row = inputs[doc_id]
        pools[(row["source_id"], label["decision"])].append(row)
    rng = random.Random(args.seed)
    selected = []
    strata = Counter()
    for source in SOURCES:
        keeps = sorted(pools[(source, "keep")], key=lambda r: r["document_id"])
        drops = sorted(pools[(source, "drop")], key=lambda r: r["document_id"])
        if source == "ultradata_code_l2_py":
            suspicious = [r for r in keeps if suspicious_code(r)]
            if len(suspicious) < 20:
                raise ValueError(f"only {len(suspicious)} suspicious Qwen-kept code items")
            targeted = rng.sample(suspicious, 20)
            targeted_ids = {r["document_id"] for r in targeted}
            random_keeps = rng.sample([r for r in keeps if r["document_id"] not in targeted_ids], 20)
            groups = [("code_keep_suspicious", targeted),
                      ("code_keep_random", random_keeps),
                      ("code_drop_random", rng.sample(drops, 20))]
        else:
            groups = [(source + "_keep_random", rng.sample(keeps, 20)),
                      (source + "_drop_random", rng.sample(drops, 20))]
        for stratum, rows in groups:
            for row in rows:
                selected.append((stratum, row))
                strata[stratum] += 1
    rng.shuffle(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for stratum, row in selected:
            # Qwen answer and selection stratum stay out of the teacher input.
            blind = {key: row[key] for key in (
                "document_id", "source_id", "current_bucket", "raw_row",
                "source_path", "upstream_score", "text_chars", "prompt_tokens_est", "review_payload",
            )}
            handle.write(json.dumps(blind, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "stage": "blind_deepseek_independent_review_not_training_approved",
        "seed": args.seed,
        "source_probe_manifest_sha256": hashlib.sha256((args.root / "inputs" / "manifest.json").read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "items": [{"document_id": row["document_id"], "selection_stratum": stratum,
                   "qwen_decision": labels[row["document_id"]]["decision"],
                   "qwen_annotation": labels[row["document_id"]]}
                  for stratum, row in selected],
        "strata": dict(strata),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"documents": len(selected), "strata": dict(strata),
                      "input_sha256": manifest["input_sha256"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
