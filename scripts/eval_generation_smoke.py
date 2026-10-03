#!/usr/bin/env python3
"""Reproducible greedy generation checks for every full training checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
from safetensors.torch import load_model
from tokenizers import Tokenizer

from small_agent.evaluation.generation_smoke import (
    compare_cache_tokens,
    generation_metrics,
    summarize_step,
)
from small_agent.models import build_model, load_model_spec


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp = Path(handle.name)
    os.replace(temp, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(value)
        temp = Path(handle.name)
    os.replace(temp, path)


def validate_config(config: dict[str, Any]) -> tuple[list[dict[str, str]], int]:
    if config.get("version") != "generation_smoke_v1":
        raise ValueError("unsupported prompt config version")
    decoding = config.get("decoding", {})
    if decoding.get("strategy") != "greedy":
        raise ValueError("smoke test requires deterministic greedy decoding")
    max_new_tokens = int(decoding.get("max_new_tokens", 0))
    if not 1 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be in [1, 512]")
    prompts = config.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts must be a nonempty list")
    ids: set[str] = set()
    for prompt in prompts:
        if not isinstance(prompt, dict) or any(
            not isinstance(prompt.get(field), str) or not prompt[field]
            for field in ("id", "domain", "text")
        ):
            raise ValueError("each prompt needs nonempty id, domain and text")
        if prompt["id"] in ids:
            raise ValueError(f"duplicate prompt id: {prompt['id']}")
        ids.add(prompt["id"])
    return prompts, max_new_tokens


def selected_checkpoints(run: Path, requested: list[int] | None) -> list[tuple[int, Path]]:
    found: dict[int, Path] = {}
    for path in sorted((run / "checkpoints").glob("step_*")):
        if path.is_dir() and (path / "manifest.json").is_file():
            step = int(path.name.removeprefix("step_"))
            found[step] = path
    if not found:
        raise FileNotFoundError("no full checkpoint manifests found")
    steps = sorted(set(requested)) if requested else sorted(found)
    missing = [step for step in steps if step not in found]
    if missing:
        raise FileNotFoundError(f"missing checkpoint steps: {missing}")
    return [(step, found[step]) for step in steps]


def verify_checkpoint(path: Path, step: int, run_manifest: dict[str, Any]) -> tuple[str, str]:
    checkpoint_manifest_path = path / "manifest.json"
    checkpoint_manifest = load_json(checkpoint_manifest_path)
    if checkpoint_manifest.get("version") not in {
        "small_agent_full_checkpoint_v1",
        "small_agent_distributed_full_checkpoint_v1",
    }:
        raise ValueError(f"unsupported checkpoint: {path}")
    if checkpoint_manifest.get("step") != step:
        raise ValueError(f"checkpoint step mismatch: {path}")
    compatibility = checkpoint_manifest.get("compatibility", {})
    for key, run_key in (
        ("run_config_sha256", "run_config"),
        ("model_spec_sha256", "model_spec"),
        ("data_manifest_sha256", "data_manifest"),
    ):
        if compatibility.get(key) != run_manifest[run_key]["sha256"]:
            raise ValueError(f"checkpoint/run {key} mismatch: {path}")
    model_path = path / "model.safetensors"
    artifact = checkpoint_manifest["artifacts"][model_path.name]
    if model_path.stat().st_size != artifact["bytes"]:
        raise ValueError(f"checkpoint model size mismatch: {model_path}")
    actual_hash = sha256_file(model_path)
    if actual_hash != artifact["sha256"]:
        raise ValueError(f"checkpoint model SHA-256 mismatch: {model_path}")
    return sha256_file(checkpoint_manifest_path), actual_hash


def generate_one(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    eos_id: int,
    pad_id: int,
    use_cache: bool,
    device: torch.device,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not prompt_ids:
        raise ValueError("prompt encoded to zero tokens")
    inputs = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(inputs)
    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
    token_ids = output[0, len(prompt_ids) :].tolist()
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return {
        "prompt_token_ids": prompt_ids,
        "token_ids": token_ids,
        "text": text,
        "metrics": generation_metrics(token_ids, text, eos_id),
    }


def render_markdown(results: list[dict[str, Any]], config: dict[str, Any]) -> str:
    lines = [
        "# 固定提示词生成验收",
        "",
        "贪心解码；这只是生成/缓存回归检查，不等同于能力评测。",
        f"提示词版本：`{config['version']}`；每条最多 {config['decoding']['max_new_tokens']} 个新 token。",
        "",
    ]
    for result in results:
        lines += [f"## Step {result['step']}", ""]
        for sample in result["samples"]:
            cache = sample["cache_comparison"]
            lines += [
                f"### {sample['id']} ({sample['domain']})",
                f"- 输入：{json.dumps(sample['prompt'], ensure_ascii=False)}",
                f"- cache 开：{json.dumps(sample['cached']['text'], ensure_ascii=False)}",
                f"- cache 关：{json.dumps(sample['uncached']['text'], ensure_ascii=False)}",
                f"- cache 完全一致：{cache['exact_match']}；首次分歧 token：{cache['first_divergence_index']}",
                f"- 重复 4-gram 率（开/关）：{sample['cached']['metrics']['repeated_4gram_rate']:.3f} / {sample['uncached']['metrics']['repeated_4gram_rate']:.3f}",
                "",
            ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/eval/generation_smoke_v1.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--step", type=int, action="append", help="default: every full checkpoint")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve() if args.output else run / "eval" / "generation_smoke_v1"
    config_path = args.config.resolve()
    config = load_json(config_path)
    prompts, max_new_tokens = validate_config(config)
    run_manifest_path = run / "run_manifest.json"
    run_manifest = load_json(run_manifest_path)
    spec_path = Path(run_manifest["model_spec"]["path"])
    data_manifest_path = Path(run_manifest["data_manifest"]["path"])
    if sha256_file(spec_path) != run_manifest["model_spec"]["sha256"]:
        raise ValueError("model spec changed since run")
    if sha256_file(data_manifest_path) != run_manifest["data_manifest"]["sha256"]:
        raise ValueError("data manifest changed since run")
    data_manifest = load_json(data_manifest_path)
    tokenizer_path = args.tokenizer_dir.resolve() / "tokenizer.json"
    tokenizer_manifest = load_json(args.tokenizer_dir.resolve() / "manifest.json")
    tokenizer_hash = sha256_file(tokenizer_path)
    if tokenizer_hash != data_manifest["tokenizer_sha256"]:
        raise ValueError("tokenizer differs from training data")
    if tokenizer_hash != tokenizer_manifest["artifacts"]["tokenizer.json"]["sha256"]:
        raise ValueError("tokenizer frozen manifest mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = int(data_manifest["eos_token_id"])
    pad_id = int(tokenizer_manifest["special_token_ids"]["<pad>"])
    if tokenizer.token_to_id("<eos>") != eos_id or tokenizer.token_to_id("<pad>") != pad_id:
        raise ValueError("special token IDs differ")
    if torch.cuda.is_available() is False and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    checkpoint_list = selected_checkpoints(run, args.step)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".generation_smoke.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"generation smoke is already running: {output}") from error
        results = []
        for step, checkpoint in checkpoint_list:
            checkpoint_manifest_hash, model_hash = verify_checkpoint(checkpoint, step, run_manifest)
            identity = {
                "run_manifest_sha256": sha256_file(run_manifest_path),
                "checkpoint_manifest_sha256": checkpoint_manifest_hash,
                "checkpoint_model_sha256": model_hash,
                "data_manifest_sha256": run_manifest["data_manifest"]["sha256"],
                "tokenizer_sha256": tokenizer_hash,
                "prompt_config_sha256": sha256_file(config_path),
            }
            result_path = output / f"step_{step:08d}.json"
            if result_path.exists():
                existing = load_json(result_path)
                if existing.get("identity") != identity or existing.get("step") != step:
                    raise ValueError(f"existing result has different inputs: {result_path}")
                results.append(existing)
                print(json.dumps({"step": step, "status": "verified_existing"}), flush=True)
                continue
            model = build_model(load_model_spec(spec_path))
            missing, unexpected = load_model(model, str(checkpoint / "model.safetensors"), strict=True)
            if missing or unexpected:
                raise ValueError(f"model state mismatch: missing={missing}, unexpected={unexpected}")
            model = model.to(device, dtype=torch.bfloat16).eval()
            samples = []
            for prompt in prompts:
                cached = generate_one(
                    model, tokenizer, prompt["text"], max_new_tokens, eos_id, pad_id, True, device
                )
                uncached = generate_one(
                    model, tokenizer, prompt["text"], max_new_tokens, eos_id, pad_id, False, device
                )
                samples.append({
                    "id": prompt["id"], "domain": prompt["domain"], "prompt": prompt["text"],
                    "cached": cached, "uncached": uncached,
                    "cache_comparison": compare_cache_tokens(
                        cached["token_ids"], uncached["token_ids"]
                    ),
                })
            result = {
                "version": "generation_smoke_result_v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_manifest["run_id"],
                "step": step,
                "identity": identity,
                "decoding": config["decoding"],
                "samples": samples,
                "summary": summarize_step(samples),
            }
            atomic_json(result_path, result)
            results.append(result)
            print(json.dumps({"step": step, "status": "complete", "summary": result["summary"]}), flush=True)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        summary = {
            "version": "generation_smoke_summary_v1",
            "run_id": run_manifest["run_id"],
            "steps": [result["step"] for result in results],
            "step_summaries": {str(result["step"]): result["summary"] for result in results},
            "note": "Deterministic generation regression only; not a general ability score.",
        }
        atomic_json(output / "summary.json", summary)
        atomic_text(output / "report.zh.md", render_markdown(results, config))
        print(json.dumps({"status": "complete", "output": str(output), "steps": summary["steps"]}), flush=True)


if __name__ == "__main__":
    main()
