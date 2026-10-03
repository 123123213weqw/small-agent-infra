#!/usr/bin/env python3
"""Small ordinary causal LM for P0 pipeline/resume validation, not GDN evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Block(nn.Module):
    def __init__(self, width: int, heads: int, intermediate: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.norm2 = nn.LayerNorm(width)
        self.gate = nn.Linear(width, intermediate, bias=False)
        self.up = nn.Linear(width, intermediate, bias=False)
        self.down = nn.Linear(intermediate, width, bias=False)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, d // self.heads).transpose(1, 2)
        k = k.view(b, t, self.heads, d // self.heads).transpose(1, 2)
        v = v.view(b, t, self.heads, d // self.heads).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(attn.transpose(1, 2).reshape(b, t, d))
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class TinyLM(nn.Module):
    def __init__(self, vocab: int, context: int, width: int = 512, layers: int = 6,
                 heads: int = 8, intermediate: int = 1536):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.position = nn.Embedding(context, width)
        self.blocks = nn.ModuleList([Block(width, heads, intermediate) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.embedding(ids) + self.position(positions)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


def load_streams(data: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("stage") != "internal_smoke_token_stream":
        raise ValueError("unexpected token stream stage")
    streams = {}
    for split, info in manifest["splits"].items():
        path = data / info["file"]
        if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
            raise ValueError(f"stream hash mismatch: {path}")
        stream = np.load(path, mmap_mode="r", allow_pickle=False)
        if len(stream) != info["tokens_including_eos"]:
            raise ValueError(f"token count mismatch: {split}")
        streams[split] = stream
    return manifest, streams


def get_batch(stream: np.ndarray, batch_size: int, context: int,
              generator: torch.Generator, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(stream) - context - 1
    if max_start < 0:
        raise ValueError("stream too short for context")
    starts = torch.randint(max_start + 1, (batch_size,), generator=generator).tolist()
    chunk = np.stack([stream[start:start + context + 1] for start in starts]).astype(np.int64)
    x = torch.from_numpy(chunk[:, :-1]).to(device, non_blocking=True)
    y = torch.from_numpy(chunk[:, 1:]).to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def validate(model: TinyLM, stream: np.ndarray, context: int, device: str) -> float:
    model.eval()
    losses = []
    for start in range(0, len(stream) - context, context):
        chunk = torch.from_numpy(np.asarray(stream[start:start + context + 1], dtype=np.int64)).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(chunk[:-1][None, :])
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), chunk[1:].reshape(-1))
        losses.append(float(loss))
    model.train()
    if not losses:
        raise ValueError("validation split has no complete context block")
    return sum(losses) / len(losses)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True, help="final global step")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this L40 engineering smoke")
    data_manifest, streams = load_streams(args.data)
    context = data_manifest["sequence_length"]
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "checkpoint.pt"
    if checkpoint.exists() and not args.resume:
        raise FileExistsError("existing checkpoint; pass --resume or use a fresh output directory")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    model = TinyLM(data_manifest["vocab_size"], context).to(args.device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, fused=True)
    step = 0
    if args.resume:
        # CPU RNG states must stay ByteTensor on CPU; optimizer.load_state_dict
        # moves its parameter states to the parameters' device as needed.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["data_manifest_sha256"] != sha256(args.data / "manifest.json"):
            raise ValueError("resume data manifest mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["batch_generator"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        step = state["step"]
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(args.output / "tensorboard"), purge_step=step if args.resume else None)
    print(json.dumps({"event": "start", "step": step, "target_step": args.steps,
                      "parameter_count": parameter_count, "context": context,
                      "batch_size": args.batch_size, "data": str(args.data)}, ensure_ascii=False), flush=True)
    start_time = time.time()
    last_train_loss = math.nan
    last_val_loss = math.nan
    for step in range(step + 1, args.steps + 1):
        x, y = get_batch(streams["train"], args.batch_size, context, generator, args.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        last_train_loss = float(loss)
        writer.add_scalar("train/loss", last_train_loss, step)
        writer.add_scalar("train/grad_norm", grad_norm, step)
        if step % args.save_every == 0 or step == args.steps:
            last_val_loss = validate(model, streams["validation"], context, args.device)
            writer.add_scalar("validation/loss", last_val_loss, step)
            writer.flush()
            state = {
                "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "batch_generator": generator.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "data_manifest_sha256": sha256(args.data / "manifest.json"),
                "parameter_count": parameter_count,
            }
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save(state, temporary)
            temporary.replace(checkpoint)
            print(json.dumps({"event": "checkpoint", "step": step, "train_loss": last_train_loss,
                              "validation_loss": last_val_loss, "grad_norm": grad_norm,
                              "elapsed_seconds": round(time.time() - start_time, 2)}, ensure_ascii=False), flush=True)
    writer.close()
    summary = {"stage": "engineering_smoke", "architecture": "ordinary_6_layer_transformer_not_gdn",
               "step": step, "parameter_count": parameter_count,
               "train_loss": last_train_loss, "validation_loss": last_val_loss,
               "context": context, "batch_size": args.batch_size,
               "tokens_processed": step * args.batch_size * context,
               "unique_train_tokens": data_manifest["splits"]["train"]["tokens_including_eos"],
               "data_manifest_sha256": sha256(args.data / "manifest.json")}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
