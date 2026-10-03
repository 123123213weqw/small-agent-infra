#!/usr/bin/env python3
"""Split a frozen FineWiki audit sample by exact Qwen prompt context fit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prompt_token_count(encoded) -> int:
    # transformers BatchEncoding is Mapping-like but not necessarily a dict;
    # len(encoded) would count keys (typically 2), not token IDs.
    ids = encoded if isinstance(encoded, list) else encoded["input_ids"]
    return len(ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--sample-manifest", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-output-tokens", type=int, default=384)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    sample_manifest = json.loads(args.sample_manifest.read_text())
    if (sample_manifest["stage"] != "finewiki_zh_stratified_quality_probe_audit_only" or
            sample_manifest["sample_sha256"] != sha256_file(args.sample)):
        raise ValueError("sample manifest mismatch")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    eligible, overlong = [], []
    counts = Counter()
    with args.sample.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            payload = row["review_payload"]
            encoded = tokenizer.apply_chat_template([
                {"role": "system", "content": prompt},
                {"role": "user", "content": payload},
            ], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            prompt_tokens = prompt_token_count(encoded)
            row["prompt_tokens_est"] = prompt_tokens
            counts[f"sampled:{row['stratum']}"] += 1
            if prompt_tokens + args.max_output_tokens > args.max_model_len:
                overlong.append(row)
                counts[f"overlong:{row['stratum']}"] += 1
            else:
                eligible.append(row)
                counts[f"eligible:{row['stratum']}"] += 1
    if sum(v for k, v in counts.items() if k.startswith("sampled:")) != sample_manifest["sampled"]:
        raise ValueError("sample count mismatch")
    args.output_dir.mkdir(parents=True)
    for name, rows in (("scoreable.input.jsonl", eligible), ("overlong.jsonl", overlong)):
        with (args.output_dir / name).open("x", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "stage": "finewiki_zh_qwen_context_split_audit_only",
        "sample_manifest_sha256": sha256_file(args.sample_manifest),
        "prompt_sha256": sha256_file(args.prompt),
        "model": str(args.model),
        "max_model_len": args.max_model_len,
        "max_output_tokens": args.max_output_tokens,
        "counts": dict(counts),
        "scoreable_sha256": sha256_file(args.output_dir / "scoreable.input.jsonl"),
        "overlong_sha256": sha256_file(args.output_dir / "overlong.jsonl"),
        "training_eligible": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
