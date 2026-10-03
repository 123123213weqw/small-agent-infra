#!/usr/bin/env python3
"""Benchmark valid-document throughput of eight local vLLM quality scorers.

Each configuration sees a disjoint but domain/length-balanced sample, so
prefix-cache hits from an earlier configuration cannot inflate later scores.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


CONFIGS = (("json_object", 1), ("json_object", 2), ("json_object", 4),
           ("json_object", 8), ("json_schema", 4))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def choose_groups(input_path: Path, prompt_path: Path, model_path: Path,
                  group_size: int, max_model_len: int, max_output_tokens: int) -> tuple[list[list[dict]], dict]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    eligible = []
    for row in rows:
        encoded = tokenizer.apply_chat_template([
            {"role": "system", "content": prompt},
            {"role": "user", "content": row["review_payload"]},
        ], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        row["prompt_tokens_est"] = len(encoded["input_ids"])
        if row["prompt_tokens_est"] + max_output_tokens <= max_model_len:
            eligible.append(row)
    count = group_size * len(CONFIGS)
    if len(eligible) < count:
        raise ValueError(f"need {count} eligible documents, have {len(eligible)}")
    selected = sorted(eligible, key=lambda row: hashlib.sha256(
        ("p0-qwen-bench-disjoint-v1:" + row["document_id"]).encode()).digest())[:count]
    selected.sort(key=lambda row: (row["current_bucket"], row["prompt_tokens_est"]))
    groups = [selected[i::len(CONFIGS)] for i in range(len(CONFIGS))]
    if any(len(group) != group_size for group in groups):
        raise AssertionError("group size mismatch")
    manifest = {
        "input_sha256": sha256_file(input_path),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "sample_documents": len(rows),
        "eligible_documents": len(eligible),
        "selected_documents": len(selected),
        "max_model_len": max_model_len,
        "max_output_tokens": max_output_tokens,
        "groups": [{"config": f"{mode}-w{workers}", "documents": len(group),
                    "prompt_tokens": sum(row["prompt_tokens_est"] for row in group),
                    "domains": dict(sorted(Counter(row["current_bucket"] for row in group).items()))}
                   for (mode, workers), group in zip(CONFIGS, groups)],
    }
    return groups, manifest


def write_shards(rows: list[dict], folder: Path) -> list[Path]:
    shards: list[list[dict]] = [[] for _ in range(8)]
    costs = [0] * 8
    for row in sorted(rows, key=lambda item: item["prompt_tokens_est"], reverse=True):
        target = min(range(8), key=lambda i: (costs[i], len(shards[i]), i))
        shards[target].append(row)
        costs[target] += row["prompt_tokens_est"]
    folder.mkdir(parents=True, exist_ok=False)
    paths = []
    for i, shard in enumerate(shards):
        path = folder / f"shard-{i}.input.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in shard), encoding="utf-8")
        paths.append(path)
    return paths


def run_config(*, rows: list[dict], mode: str, workers: int, folder: Path,
               annotate_script: Path, python: Path, prompt: Path, schema: Path,
               max_output_tokens: int) -> dict:
    paths = write_shards(rows, folder)
    env = {**os.environ, "QWEN_LOCAL_API_KEY": "EMPTY"}
    processes = []
    started = time.monotonic()
    for i, path in enumerate(paths):
        output = folder / f"shard-{i}.results.jsonl"
        command = [str(python), str(annotate_script),
                   "--input", str(path), "--output", str(output),
                   "--prompt-file", str(prompt), "--schema-file", str(schema),
                   "--text-field", "review_payload", "--id-fields", "document_id",
                   "--carry-fields", "current_bucket,text_chars,token_count,prompt_tokens_est",
                   "--api-base", f"http://127.0.0.1:1800{i}/v1",
                   "--api-key-env", "QWEN_LOCAL_API_KEY", "--model", "qwen3.8-27b-fp8",
                   "--thinking", "omit", "--temperature", "0", "--workers", str(workers),
                   "--max-tokens", str(max_output_tokens), "--timeout", "120",
                   "--max-retries", "0", "--response-format", mode,
                   "--chat-template-kwargs", '{"enable_thinking":false}',
                   "--require-document-id-match"]
        log = (folder / f"shard-{i}.log").open("w", encoding="utf-8")
        processes.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    codes = []
    for proc, log in processes:
        codes.append(proc.wait())
        log.close()
    elapsed = time.monotonic() - started
    results = []
    for i in range(8):
        path = folder / f"shard-{i}.results.jsonl"
        if path.exists():
            results.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    ok = [row for row in results if row["status"] == "ok"]
    report = {
        "mode": mode, "workers_per_endpoint": workers,
        "requested": len(rows), "returned": len(results), "valid": len(ok),
        "errors": dict(sorted(Counter(row.get("error", "unknown") for row in results if row["status"] != "ok").items())),
        "process_exit_codes": codes,
        "wall_s": round(elapsed, 3),
        "valid_docs_per_s": round(len(ok) / elapsed, 3),
        "valid_prompt_tokens_per_s": round(sum(row.get("usage", {}).get("prompt_tokens", 0) for row in ok) / elapsed, 1),
        "valid_completion_tokens_per_s": round(sum(row.get("usage", {}).get("completion_tokens", 0) for row in ok) / elapsed, 1),
        "mean_completion_tokens": round(sum(row.get("usage", {}).get("completion_tokens", 0) for row in ok) / max(1, len(ok)), 1),
    }
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--annotate-script", type=Path, required=True)
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--group-size", type=int, default=96)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-output-tokens", type=int, default=384)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"output already exists: {args.output_dir}")
    groups, manifest = choose_groups(args.input, args.prompt, args.model_path,
                                     args.group_size, args.max_model_len, args.max_output_tokens)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reports = []
    for (mode, workers), group in zip(CONFIGS, groups):
        print(f"START {mode} workers={workers} docs={len(group)}", flush=True)
        report = run_config(rows=group, mode=mode, workers=workers,
                            folder=args.output_dir / f"{mode}-w{workers}",
                            annotate_script=args.annotate_script, python=args.python,
                            prompt=args.prompt, schema=args.schema,
                            max_output_tokens=args.max_output_tokens)
        reports.append(report)
        (args.output_dir / "reports.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
