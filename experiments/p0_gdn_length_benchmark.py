#!/usr/bin/env python3
"""Measure 8-layer GDN/GQA forward+backward at fixed lengths on one L40."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import Qwen3NextForCausalLM

from p0_gdn_hybrid_smoke import FUSED_NORM_ACTIVE, model_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    stream = np.load(args.data / manifest["splits"]["train"]["file"], mmap_mode="r", allow_pickle=False)
    if args.length + 1 > len(stream):
        raise ValueError("stream shorter than requested length")
    torch.manual_seed(20260924)
    config = model_config(manifest["vocab_size"])
    config.max_position_embeddings = max(8192, args.length + 1)
    model = Qwen3NextForCausalLM(config).cuda().train()
    ids = torch.from_numpy(np.asarray(stream[:args.length + 1], dtype=np.int64)).cuda()[None, :]
    fast = bool(importlib.util.find_spec("fla")) and bool(importlib.util.find_spec("causal_conv1d"))
    if not fast:
        raise RuntimeError("fast GDN dependencies not active")
    times = []
    losses = []
    for iteration in range(args.warmup + args.repeat):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at iteration {iteration}")
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if iteration >= args.warmup:
            times.append(elapsed)
            losses.append(float(loss))
        print(json.dumps({"iteration": iteration, "warmup": iteration < args.warmup,
                          "seconds": round(elapsed, 4), "loss": float(loss),
                          "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
                          "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 4)}), flush=True)
    report = {
        "stage": "gdn_gqa_length_benchmark", "length": args.length,
        "batch_size": 1, "warmup": args.warmup, "repeat": args.repeat,
        "fast_gdn_kernels": fast, "fused_norm_active": FUSED_NORM_ACTIVE,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": times, "mean_seconds": sum(times) / len(times),
        "tokens_per_second": args.length / (sum(times) / len(times)),
        "losses": losses,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
