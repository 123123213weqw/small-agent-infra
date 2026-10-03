"""Paired held-out format checks for the trained contextual-memory Reader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase1_reader_v2 import (
    ContextualMemoryReader,
    evaluate,
    evaluate_full_context,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--eval-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-record-tokens", type=int, default=48)
    p.add_argument("--read-scale", type=float, default=0.1)
    p.add_argument("--value-source", choices=("contextual", "token_embedding"), default="contextual")
    p.add_argument("--seed", type=int, default=260923)
    p.add_argument("--variants", default="standard,heldout_ids,paraphrase,delay,unseen_natural,delay64,id_extreme,record_paraphrase_only,query_paraphrase_only")
    args = p.parse_args()
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    reader = ContextualMemoryReader(
        base, read_scale=args.read_scale, value_source=args.value_source,
    ).to(device)
    saved = torch.load(args.adapter, map_location="cpu", weights_only=True)
    if saved["manifest"].get("value_source", "contextual") != args.value_source:
        raise ValueError("Adapter value_source does not match evaluation mode")
    reader.memory_attention.load_state_dict(saved["adapter"])
    reader.eval()
    results = []
    for variant in args.variants.split(","):
        spec = SimpleNamespace(
            slots=args.slots, eval_samples=args.eval_samples,
            batch_size=args.batch_size, max_record_tokens=args.max_record_tokens,
            seed=args.seed, eval_variant=variant,
        )
        full = evaluate_full_context(base, tokenizer, spec, device)
        memory = evaluate(reader, tokenizer, spec, device)
        row = {"variant": variant, "full_context": full, "memory_reader": memory}
        print(json.dumps(row), flush=True)
        results.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "adapter": str(args.adapter), "model": "Qwen/Qwen3-0.6B-Base",
        "eval_samples_per_variant": args.eval_samples, "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
