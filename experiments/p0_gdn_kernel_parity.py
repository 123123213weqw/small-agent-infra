#!/usr/bin/env python3
"""Compare accelerated and PyTorch-reference GDN kernels on identical weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen3NextForCausalLM
from transformers.models.qwen3_next import modeling_qwen3_next as qm

from p0_gdn_hybrid_smoke import model_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--length", type=int, required=True)
    args = p.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    stream = np.load(args.data / manifest["splits"]["train"]["file"], mmap_mode="r", allow_pickle=False)
    torch.manual_seed(20260924)
    config = model_config(manifest["vocab_size"])
    config.max_position_embeddings = max(8192, args.length + 1)
    model = Qwen3NextForCausalLM(config).cuda().eval()
    ids = torch.from_numpy(np.asarray(stream[:args.length + 1], dtype=np.int64)).cuda()[None, :]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        fast = model(input_ids=ids, labels=ids, use_cache=False)
    fast_gradient = None
    if args.length <= 128:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fast_train_loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        fast_train_loss.backward()
        fast_gradient = model.model.embed_tokens.weight.grad.detach().float().clone()
        model.zero_grad(set_to_none=True)
    for layer in model.model.layers:
        if layer.layer_type == "linear_attention":
            gdn = layer.linear_attn
            gdn.chunk_gated_delta_rule = qm.torch_chunk_gated_delta_rule
            gdn.recurrent_gated_delta_rule = qm.torch_recurrent_gated_delta_rule
            gdn.causal_conv1d_fn = None
            gdn.causal_conv1d_update = qm.torch_causal_conv1d_update
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = model(input_ids=ids, labels=ids, use_cache=False)
    reference_gradient = None
    if fast_gradient is not None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference_train_loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        reference_train_loss.backward()
        reference_gradient = model.model.embed_tokens.weight.grad.detach().float()
    difference = (fast.logits.float() - reference.logits.float()).abs()
    report = {
        "stage": "gdn_kernel_parity", "length": args.length, "dtype": "bf16",
        "max_logit_abs_diff": float(difference.max()),
        "mean_logit_abs_diff": float(difference.mean()),
        "fast_loss": float(fast.loss), "reference_loss": float(reference.loss),
        "loss_abs_diff": abs(float(fast.loss) - float(reference.loss)),
        "greedy_mismatches": int((fast.logits.argmax(-1) != reference.logits.argmax(-1)).sum()),
    }
    if fast_gradient is not None:
        report["embedding_gradient_cosine"] = float(torch.nn.functional.cosine_similarity(
            fast_gradient.flatten(), reference_gradient.flatten(), dim=0))
        report["embedding_gradient_relative_l2"] = float(
            (fast_gradient - reference_gradient).norm() / reference_gradient.norm())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not np.isfinite(report["fast_loss"]) or not np.isfinite(report["reference_loss"]):
        raise FloatingPointError("non-finite loss")
    if report["loss_abs_diff"] > 0.02 or report["mean_logit_abs_diff"] > 0.02:
        raise AssertionError("fast/reference GDN parity exceeded tolerance")
    if fast_gradient is not None and (
        report["embedding_gradient_cosine"] < 0.99
        or report["embedding_gradient_relative_l2"] > 0.1
    ):
        raise AssertionError("fast/reference gradient parity exceeded tolerance")


if __name__ == "__main__":
    main()
