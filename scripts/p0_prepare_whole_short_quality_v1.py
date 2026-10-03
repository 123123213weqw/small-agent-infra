#!/usr/bin/env python3
"""Prepare whole-document, non-truncated Qwen quality audit for <6000-char samples."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--sample-manifest", type=Path, required=True)
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-output-tokens", type=int, default=1024)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    m = json.loads(a.sample_manifest.read_text())
    if m["sample_sha256"] != sha(a.sample):
        raise ValueError("sample hash mismatch")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    prompt = a.prompt.read_text(encoding="utf-8").strip()
    rows = []
    counts = Counter()
    for doc in map(json.loads, a.sample.open(encoding="utf-8")):
        if doc["chars"] >= 6000:
            continue
        payload = json.dumps({
            "document_id": doc["document_id"],
            "current_bucket": "general_zh", "document_text": doc["text"],
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        enc = tokenizer.apply_chat_template([
            {"role": "system", "content": prompt},
            {"role": "user", "content": payload},
        ], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        n = len(enc if isinstance(enc, list) else enc["input_ids"])
        if n + a.max_output_tokens > a.max_model_len:
            raise ValueError(f"whole document does not fit: {doc['document_id']} {n}")
        rows.append({
            "document_id": doc["document_id"], "source": doc["source"],
            "stratum": doc["stratum"], "chars": doc["chars"],
            "population_in_stratum": doc["population_in_stratum"],
            "sample_size_in_stratum": doc["sample_size_in_stratum"],
            "prompt_tokens_est": n, "review_payload": payload,
        })
        counts[doc["source"]] += 1
    if counts != {"finewiki_zh": 152, "wave1_general_zh": 150}:
        raise ValueError(f"sample counts changed: {counts}")
    a.output.mkdir(parents=True)
    inp = a.output / "whole_short.input.jsonl"
    with inp.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    report = {
        "stage": "finewiki_short_whole_document_quality_input_audit_only",
        "sample_manifest_sha256": sha(a.sample_manifest),
        "sample_sha256": sha(a.sample), "prompt_sha256": sha(a.prompt),
        "input_sha256": sha(inp), "model": str(a.model),
        "max_model_len": a.max_model_len, "max_output_tokens": a.max_output_tokens,
        "counts": dict(counts), "max_prompt_tokens": max(r["prompt_tokens_est"] for r in rows),
        "training_eligible": False,
        "note": "Full documents only; >=6000-character candidates are excluded from initial training-ready source.",
    }
    (a.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
