#!/usr/bin/env python3
"""Engineering-only optimizer/checkpoint smoke for frozen Base-1B-v1."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import Qwen3NextConfig, Qwen3NextForCausalLM


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--steps", type=int, required=True, help="final global step")
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-last", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    args = p.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if distributed:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_hash = sha256(args.config)
    data_hash = sha256(args.data / "manifest.json")
    cfg = Qwen3NextConfig(**json.loads(args.config.read_text()))
    if cfg.num_hidden_layers != 32 or cfg.vocab_size != 32768:
        raise ValueError("not frozen Base-1B-v1")
    if args.context > cfg.max_position_embeddings:
        raise ValueError("context exceeds configured position range")
    if args.accumulation_steps < 1:
        raise ValueError("accumulation steps must be positive")
    manifest = json.loads((args.data / "manifest.json").read_text())
    train_info = manifest["splits"]["train"]
    train_path = args.data / train_info["file"]
    if sha256(train_path) != train_info["sha256"]:
        raise ValueError("training token stream checksum mismatch")
    stream = np.load(train_path, mmap_mode="r", allow_pickle=False)
    if len(stream) <= args.context:
        raise ValueError("token stream shorter than context")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sample_rng = torch.Generator(device="cpu").manual_seed(args.seed + 1000 + rank)
    model = Qwen3NextForCausalLM(cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    if params != 1_005_213_696:
        raise AssertionError(f"parameter count changed: {params}")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    if distributed:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    core = model.module if distributed else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, fused=True)
    checkpoint = args.run_dir / "checkpoint.pt"
    step = 0
    if args.resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["config_sha256"] != config_hash or state["data_sha256"] != data_hash:
            raise ValueError("configuration or data changed across resume")
        if state["world_size"] != world_size or state["context"] != args.context:
            raise ValueError("distributed layout or context changed across resume")
        if state.get("accumulation_steps", 1) != args.accumulation_steps:
            raise ValueError("accumulation changed across resume")
        if state["gradient_checkpointing"] != args.gradient_checkpointing:
            raise ValueError("gradient checkpointing setting changed across resume")
        core.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        states = state["rng_states"][rank]
        sample_rng.set_state(states["sample"])
        torch.set_rng_state(states["torch"])
        torch.cuda.set_rng_state(states["cuda"], device=device)
        step = state["step"]
        del state
    elif checkpoint.exists():
        raise FileExistsError(f"existing checkpoint, use --resume: {checkpoint}")
    if distributed:
        dist.barrier()
    if rank == 0:
        print(json.dumps({"event": "start", "parameters": params, "world_size": world_size,
                          "context": args.context, "batch_size_per_gpu": args.batch_size,
                          "accumulation_steps": args.accumulation_steps,
                          "gradient_checkpointing": args.gradient_checkpointing,
                          "start_step": step, "target_step": args.steps,
                          "config_sha256": config_hash, "data_sha256": data_hash}), flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        local_loss = torch.zeros((), device=device)
        for microstep in range(args.accumulation_steps):
            indices = torch.randint(0, len(stream) - args.context + 1,
                                    (args.batch_size,), generator=sample_rng).tolist()
            x = np.stack([stream[i:i + args.context] for i in indices]).astype(np.int64)
            tokens = torch.from_numpy(x).to(device)
            sync = not distributed or microstep == args.accumulation_steps - 1
            with (nullcontext() if sync else model.no_sync()):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # HF causal LM shifts labels internally; passing pre-shifted
                    # labels trains a two-token-ahead prediction task.
                    loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}, microstep {microstep}")
                local_loss += loss.detach() / args.accumulation_steps
                (loss / args.accumulation_steps).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        max_elapsed = torch.tensor(elapsed, device=device)
        mean_loss = local_loss
        max_allocated = torch.tensor(torch.cuda.max_memory_allocated(device), device=device,
                                     dtype=torch.float64)
        max_reserved = torch.tensor(torch.cuda.max_memory_reserved(device), device=device,
                                    dtype=torch.float64)
        if distributed:
            dist.all_reduce(max_elapsed, op=dist.ReduceOp.MAX)
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss /= world_size
            dist.all_reduce(max_allocated, op=dist.ReduceOp.MAX)
            dist.all_reduce(max_reserved, op=dist.ReduceOp.MAX)
        row = {"event": "step", "step": step, "loss": float(mean_loss),
               "grad_norm": float(grad_norm), "seconds": float(max_elapsed),
               "global_tokens": (args.context - 1) * args.batch_size * args.accumulation_steps * world_size,
               "tokens_per_second": ((args.context - 1) * args.batch_size * args.accumulation_steps
                                     * world_size / float(max_elapsed)),
               "peak_allocated_gib": float(max_allocated) / 2**30,
               "peak_reserved_gib": float(max_reserved) / 2**30}
        if rank == 0:
            print(json.dumps(row), flush=True)
            with (args.run_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")

    if args.save_last:
        local_state = {"sample": sample_rng.get_state(), "torch": torch.get_rng_state(),
                       "cuda": torch.cuda.get_rng_state(device=device)}
        rng_states = [None] * world_size
        if distributed:
            dist.all_gather_object(rng_states, local_state)
        else:
            rng_states[0] = local_state
        if rank == 0:
            state = {"step": step, "config_sha256": config_hash, "data_sha256": data_hash,
                     "world_size": world_size, "context": args.context,
                     "accumulation_steps": args.accumulation_steps,
                     "gradient_checkpointing": args.gradient_checkpointing,
                     "model": core.state_dict(), "optimizer": optimizer.state_dict(),
                     "rng_states": rng_states}
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save(state, temporary)
            os.replace(temporary, checkpoint)
            print(json.dumps({"event": "checkpoint", "step": step,
                              "bytes": checkpoint.stat().st_size,
                              "sha256": sha256(checkpoint)}), flush=True)
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
