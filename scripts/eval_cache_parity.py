#!/usr/bin/env python3
"""Check cached decoding against full forward logits on frozen validation tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_agent.data import ShardedTokenDataset  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_stats(cache: object) -> dict[str, object]:
    stats: dict[str, object] = {"class": type(cache).__name__}
    if hasattr(cache, "get_seq_length"):
        stats["sequence_length"] = int(cache.get_seq_length())
    arrays: dict[str, object] = {}
    total_bytes = 0
    for name in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
        items = getattr(cache, name, None)
        if items is None:
            continue
        shapes = []
        for item in items:
            if isinstance(item, torch.Tensor):
                shapes.append(list(item.shape))
                total_bytes += item.numel() * item.element_size()
            else:
                shapes.append(None)
        arrays[name] = {"layers": len(items), "shapes": shapes}
    stats["arrays"] = arrays
    stats["tensor_bytes"] = total_bytes
    return stats


def compare_logits(reference: torch.Tensor, cached: torch.Tensor) -> dict[str, float | bool | int]:
    a, b = reference.float(), cached.float()
    delta = (a - b).abs()
    probs_a = torch.softmax(a, dim=-1)
    probs_b = torch.softmax(b, dim=-1)
    ref_top = torch.topk(a, 2)
    cache_top = torch.topk(b, 2)
    reference_argmax = int(a.argmax())
    cached_argmax = int(b.argmax())
    reference_tie = bool(ref_top.values[0] == ref_top.values[1])
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "top1_equal": reference_argmax == cached_argmax,
        "tie_aware_top1_equal": reference_argmax == cached_argmax
        or (reference_tie and bool(a[cached_argmax] == ref_top.values[0])),
        "reference_top1_tied": reference_tie,
        "reference_top1_id": reference_argmax,
        "cached_top1_id": cached_argmax,
        "reference_top1_margin": float(ref_top.values[0] - ref_top.values[1]),
        "cached_top1_margin": float(cache_top.values[0] - cache_top.values[1]),
        "total_variation": float((probs_a - probs_b).abs().sum() / 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-ref", type=int, default=0)
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--continuation-tokens", type=int, default=8)
    parser.add_argument("--max-abs-tolerance", type=float, default=0.1)
    parser.add_argument("--max-tv-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    export = json.loads((args.model / "export_manifest.json").read_text(encoding="utf-8"))
    if export["stage"] != "internal_non_distributable_hf_export":
        raise ValueError("not a diagnostic HF export")
    manifest_path = args.data / "manifest.json"
    if sha256_file(manifest_path) != export["data_manifest_sha256"]:
        raise ValueError("evaluation data differs from training manifest")
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = ShardedTokenDataset(args.data, "validation", int(dataset_manifest["sequence_length"]))
    row = dataset[args.sample_ref]
    valid = int(row["valid_tokens"])
    needed = max(args.prefix_lengths) + args.continuation_tokens
    if valid < needed:
        raise ValueError(f"validation sample too short: valid={valid}, needed={needed}")
    if any(length < 1 for length in args.prefix_lengths) or args.continuation_tokens < 1:
        raise ValueError("all lengths must be positive")

    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False
    ).to(device).eval()
    ids = row["input_ids"][:needed].to(device)
    results = []
    with torch.inference_mode():
        for prefix_length in args.prefix_lengths:
            count = args.continuation_tokens
            tokens = ids[: prefix_length + count].unsqueeze(0)
            full = model(input_ids=tokens, use_cache=False).logits[0]
            prefill = model(input_ids=tokens[:, :prefix_length], use_cache=True)
            cache = prefill.past_key_values
            initial_stats = cache_stats(cache)
            current = prefill.logits[0, -1]
            comparisons = []
            for k in range(count):
                comparisons.append(compare_logits(full[prefix_length - 1 + k], current))
                next_token = tokens[:, prefix_length + k : prefix_length + k + 1]
                next_result = model(input_ids=next_token, past_key_values=cache, use_cache=True)
                cache = next_result.past_key_values
                current = next_result.logits[0, -1]
            final_stats = cache_stats(cache)
            result = {
                "prefix_length": prefix_length,
                "continuation_tokens": count,
                "max_abs": max(x["max_abs"] for x in comparisons),
                "mean_abs": sum(x["mean_abs"] for x in comparisons) / count,
                "max_total_variation": max(x["total_variation"] for x in comparisons),
                "top1_agreement": sum(x["top1_equal"] for x in comparisons) / count,
                "tie_aware_top1_agreement": sum(x["tie_aware_top1_equal"] for x in comparisons) / count,
                "per_position": comparisons,
                "cache_prefill": initial_stats,
                "cache_final": final_stats,
            }
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in {"per_position", "cache_prefill", "cache_final"}}), flush=True)

    passed = all(
        item["max_abs"] <= args.max_abs_tolerance
        and item["max_total_variation"] <= args.max_tv_tolerance
        and item["tie_aware_top1_agreement"] == 1.0
        for item in results
    )
    report = {
        "stage": "internal_diagnostic_cache_parity",
        "checkpoint_step": export["checkpoint_step"],
        "export_manifest_sha256": sha256_file(args.model / "export_manifest.json"),
        "data_manifest_sha256": sha256_file(manifest_path),
        "validation_sample": dataset.reference(args.sample_ref),
        "valid_tokens": valid,
        "dtype": "bfloat16",
        "tolerances": {"max_abs": args.max_abs_tolerance, "max_total_variation": args.max_tv_tolerance},
        "passed": passed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"passed": passed, "output": str(args.output)}), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
