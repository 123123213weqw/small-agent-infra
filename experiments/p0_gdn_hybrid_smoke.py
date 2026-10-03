#!/usr/bin/env python3
"""From-scratch 8-layer dense GDN/GQA correctness and short training smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
from transformers.models.qwen3_next import modeling_qwen3_next as qwen3_next_modeling


FUSED_NORM_ACTIVE = qwen3_next_modeling.FusedRMSNormGated is not None


LAYERS = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_config(vocab_size: int) -> Qwen3NextConfig:
    return Qwen3NextConfig(
        vocab_size=vocab_size, hidden_size=512, intermediate_size=1536,
        num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=2,
        head_dim=64, linear_num_key_heads=8, linear_num_value_heads=8,
        linear_key_head_dim=64, linear_value_head_dim=64,
        linear_conv_kernel_dim=4, layer_types=LAYERS,
        num_experts=0, decoder_sparse_step=1,
        max_position_embeddings=8192, tie_word_embeddings=True,
        use_cache=True, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        # Avoid the Transformers 4.57.1 torch.get_current_dtype() fallback:
        # torch 2.5.1 has no such API. The model weights still initialize fp32;
        # this specifies the optional fused GDN norm's compute dtype.
        dtype=torch.bfloat16,
    )


def load_data(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["stage"] != "internal_smoke_token_stream":
        raise ValueError("unexpected data stage")
    streams = {}
    for split, info in manifest["splits"].items():
        path = directory / info["file"]
        if sha256(path) != info["sha256"]:
            raise ValueError(f"stream hash mismatch: {split}")
        streams[split] = np.load(path, mmap_mode="r", allow_pickle=False)
    return manifest, streams


def batch(stream: np.ndarray, length: int, size: int, generator: torch.Generator,
          device: str) -> torch.Tensor:
    starts = torch.randint(len(stream) - length, (size,), generator=generator).tolist()
    chunks = np.stack([stream[s:s + length + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(chunks).to(device)


def run_checks(model: Qwen3NextForCausalLM, stream: np.ndarray, device: str, fast_kernels: bool) -> dict:
    model.eval()
    ids = torch.from_numpy(np.asarray(stream[:12], dtype=np.int64)).to(device)[None, :]
    # FLA's chunk GDN kernel is BF16/FP16-only; the reference PyTorch path
    # supports FP32. Compare prefill and recurrent decode at the same dtype.
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=fast_kernels):
        whole = model(input_ids=ids, use_cache=False).logits
        cache = None
        pieces = []
        for i in range(ids.shape[1]):
            out = model(input_ids=ids[:, i:i + 1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            pieces.append(out.logits)
        incremental = torch.cat(pieces, dim=1)
        max_abs = float((whole - incremental).abs().max())
        disagreement = whole.argmax(-1) != incremental.argmax(-1)
        greedy_mismatch_count = int(disagreement.sum())
        greedy_match = greedy_mismatch_count == 0
        margins = whole.float().topk(2, dim=-1).values
        max_mismatch_margin = (
            float((margins[..., 0] - margins[..., 1])[disagreement].max())
            if greedy_mismatch_count else 0.0
        )
    tolerance = 0.1 if fast_kernels else 1e-3
    if max_abs > tolerance or max_mismatch_margin > 2 * max_abs:
        raise AssertionError(
            f"prefill/decode disagreement: max_abs={max_abs}, max_mismatch_margin={max_mismatch_margin}"
        )
    model.train()
    x = torch.from_numpy(np.asarray(stream[20:37], dtype=np.int64)).to(device)[None, :]
    # HF causal LM shifts labels internally. Passing already-shifted labels
    # would accidentally train a two-token-ahead prediction task.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=fast_kernels):
        loss = model(input_ids=x, labels=x, use_cache=False).loss
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite initial loss")
    loss.backward()
    grad = model.model.embed_tokens.weight.grad
    if grad is None or not torch.isfinite(grad).all() or float(grad.norm()) == 0:
        raise AssertionError("missing/non-finite embedding gradient")
    gradient_norm = float(grad.norm())
    model.zero_grad(set_to_none=True)
    return {"max_prefill_decode_abs_diff": max_abs, "dtype": "bf16" if fast_kernels else "fp32",
            "greedy_match": greedy_match, "greedy_mismatch_count": greedy_mismatch_count,
            "max_mismatch_margin": max_mismatch_margin, "initial_loss": float(loss),
            "embedding_gradient_norm": gradient_norm}


@torch.no_grad()
def val_loss(model: Qwen3NextForCausalLM, stream: np.ndarray, length: int,
             device: str) -> float:
    model.eval()
    values = []
    for start in range(0, min(len(stream) - length, length * 4), length):
        x = torch.from_numpy(np.asarray(stream[start:start + length + 1], dtype=np.int64)).to(device)[None, :]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            values.append(float(model(input_ids=x, labels=x, use_cache=False).loss))
    model.train()
    return sum(values) / len(values)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True, help="final global step")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.context > 2048 or args.context < 16:
        p.error("context must be in [16, 2048]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    manifest, streams = load_data(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "checkpoint.pt"
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(checkpoint)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    model = Qwen3NextForCausalLM(model_config(manifest["vocab_size"])).to(args.device)
    params = sum(x.numel() for x in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, fused=True)
    fast_kernels = bool(importlib.util.find_spec("fla")) and bool(importlib.util.find_spec("causal_conv1d"))
    step = 0
    if args.resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["data_manifest_sha256"] != sha256(args.data / "manifest.json"):
            raise ValueError("data changed between runs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["batch_generator"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        step = state["step"]
    else:
        checks = run_checks(model, streams["train"], args.device, fast_kernels)
        (args.output / "correctness.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(json.dumps({"event": "correctness", **checks}), flush=True)
    print(json.dumps({"event": "start", "step": step, "target_step": args.steps,
                      "parameters": params, "fast_gdn_kernels": fast_kernels,
                      "fused_norm_active": FUSED_NORM_ACTIVE,
                      "layer_types": LAYERS, "context": args.context}), flush=True)
    started = time.time()
    for step in range(step + 1, args.steps + 1):
        chunk = batch(streams["train"], args.context, args.batch_size, generator, args.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=chunk, labels=chunk, use_cache=False).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        print(json.dumps({"event": "step", "step": step, "train_loss": float(loss),
                          "grad_norm": grad_norm}), flush=True)
    validation_loss = val_loss(model, streams["validation"], args.context, args.device)
    if not np.isfinite(validation_loss):
        raise FloatingPointError("non-finite validation loss")
    state = {
        "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "batch_generator": generator.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "data_manifest_sha256": sha256(args.data / "manifest.json"),
        "parameters": params, "context": args.context, "batch_size": args.batch_size,
    }
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(checkpoint)
    summary = {"stage": "gdn_gqa_correctness_smoke", "step": step,
               "parameters": params, "fast_gdn_kernels": fast_kernels,
               "fused_norm_active": FUSED_NORM_ACTIVE,
               "context": args.context, "batch_size": args.batch_size,
               "tokens_processed": step * args.context * args.batch_size,
               "last_train_loss": float(loss), "validation_loss": validation_loss,
               "elapsed_seconds_this_invocation": round(time.time() - started, 3)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "checkpoint", **summary}), flush=True)


if __name__ == "__main__":
    main()
