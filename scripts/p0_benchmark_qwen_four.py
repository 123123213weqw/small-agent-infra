#!/usr/bin/env python3
"""Benchmark four optimized L40 Qwen replicas on 720 real candidate documents."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen


GPUS = (0, 2, 3, 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--annotate-script", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--schema", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"output already exists: {args.output_dir}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    original = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({r["document_id"] for r in original}) != len(original):
        raise ValueError("duplicate input document ID")
    eligible = []
    for row in original:
        tokens = tokenizer.apply_chat_template([
            {"role": "system", "content": prompt},
            {"role": "user", "content": row["review_payload"]},
        ], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        row["prompt_tokens_est"] = len(tokens["input_ids"])
        if row["prompt_tokens_est"] + 384 <= 4096:
            eligible.append(row)
    shards: list[list[dict]] = [[] for _ in GPUS]
    costs = [0] * len(GPUS)
    for row in sorted(eligible, key=lambda r: (-r["prompt_tokens_est"], r["document_id"])):
        index = min(range(len(GPUS)), key=lambda i: (costs[i], len(shards[i]), i))
        shards[index].append(row)
        costs[index] += row["prompt_tokens_est"] + 384
    args.output_dir.mkdir(parents=True)
    shard_meta = []
    for index, (gpu, shard) in enumerate(zip(GPUS, shards)):
        path = args.output_dir / f"shard-{index}.input.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in shard), encoding="utf-8")
        shard_meta.append({"gpu": gpu, "documents": len(shard), "prompt_tokens": sum(x["prompt_tokens_est"] for x in shard),
                           "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"source_input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "source_documents": len(original), "eligible_documents": len(eligible), "excluded_overlong": len(original)-len(eligible),
                "gpu_indices": GPUS, "shards": shard_meta,
                "stage": "four_gpu_throughput_probe_not_training_approved"}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for attempt in range(60):
        healthy = True
        for gpu in GPUS:
            try:
                with urlopen(f"http://127.0.0.1:1800{gpu}/health", timeout=5) as response:
                    healthy &= response.status == 200
            except Exception:
                healthy = False
        if healthy:
            break
        time.sleep(10)
    else:
        raise RuntimeError("four replicas not healthy within 10 minutes")

    processes = []
    env = {**os.environ, "QWEN_LOCAL_API_KEY": "EMPTY"}
    started = time.monotonic()
    for index, gpu in enumerate(GPUS):
        log = (args.output_dir / f"shard-{index}.log").open("w", encoding="utf-8")
        cmd = [str(args.python), str(args.annotate_script),
               "--input", str(args.output_dir / f"shard-{index}.input.jsonl"),
               "--output", str(args.output_dir / f"shard-{index}.results.jsonl"),
               "--prompt-file", str(args.prompt), "--schema-file", str(args.schema),
               "--text-field", "review_payload", "--id-fields", "document_id",
               "--carry-fields", "current_bucket,token_count,prompt_tokens_est",
               "--api-base", f"http://127.0.0.1:1800{gpu}/v1", "--api-key-env", "QWEN_LOCAL_API_KEY",
               "--model", "qwen3.8-27b-fp8", "--thinking", "omit", "--temperature", "0",
               "--chat-template-kwargs", '{"enable_thinking":false}', "--response-format", "json_schema",
               "--require-document-id-match", "--workers", "32", "--max-tokens", "384",
               "--timeout", "180", "--max-retries", "0"]
        processes.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    exit_codes = []
    for process, log in processes:
        exit_codes.append(process.wait())
        log.close()
    elapsed = time.monotonic() - started
    results = [json.loads(line) for index in range(len(GPUS)) for line in
               (args.output_dir / f"shard-{index}.results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = [row for row in results if row["status"] == "ok"]
    report = {"requested": len(eligible), "returned": len(results), "valid": len(valid),
              "errors": dict(sorted(Counter(row.get("error", "unknown") for row in results if row["status"] != "ok").items())),
              "exit_codes": exit_codes, "wall_s": round(elapsed, 3),
              "valid_docs_per_s": round(len(valid) / elapsed, 3),
              "prompt_tokens_per_s": round(sum(row.get("usage", {}).get("prompt_tokens", 0) for row in valid) / elapsed, 1),
              "completion_tokens_per_s": round(sum(row.get("usage", {}).get("completion_tokens", 0) for row in valid) / elapsed, 1)}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
