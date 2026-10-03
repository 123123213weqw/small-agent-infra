#!/usr/bin/env python3
"""Token-weighted validation loss by domain for a frozen sharded checkpoint."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset
from safetensors.torch import load_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_agent.data import ShardedTokenDataset  # noqa: E402
from small_agent.evaluation import evaluate_causal_lm  # noqa: E402
from small_agent.models import build_model, load_model_spec  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.run_config.read_text(encoding="utf-8"))
    data_dir = Path(config["data_dir"])
    spec_path = ROOT / config["model_spec"]
    model_path = args.checkpoint / "model.safetensors"
    checkpoint_manifest = json.loads((args.checkpoint / "manifest.json").read_text(encoding="utf-8"))
    artifact = checkpoint_manifest["artifacts"]["model.safetensors"]
    if model_path.stat().st_size != artifact["bytes"] or sha256_file(model_path) != artifact["sha256"]:
        raise ValueError("checkpoint model integrity failure")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dataset = ShardedTokenDataset(data_dir, "validation", int(config["context_length"]))
    groups: dict[str, list[int]] = defaultdict(list)
    for sample_ref in range(len(dataset)):
        groups[str(dataset.reference(sample_ref)["domain"])].append(sample_ref)

    model = build_model(load_model_spec(spec_path))
    missing, unexpected = load_model(model, str(model_path), strict=True)
    if missing or unexpected:
        raise ValueError(f"model state mismatch: missing={missing}, unexpected={unexpected}")
    model = model.to(device)
    model.config.use_cache = False
    model.eval()

    results = {}
    for domain, sample_refs in sorted(groups.items()):
        loader = DataLoader(
            Subset(dataset, sample_refs),
            batch_size=int(config["validation"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        result = evaluate_causal_lm(model, loader, device)
        results[domain] = {
            "loss": result.loss,
            "perplexity": result.perplexity,
            "predicted_tokens": result.predicted_tokens,
            "samples": len(sample_refs),
            "batches": result.batches,
        }
        print(json.dumps({"domain": domain, **results[domain]}), flush=True)

    total_tokens = sum(item["predicted_tokens"] for item in results.values())
    weighted_loss = sum(item["loss"] * item["predicted_tokens"] for item in results.values()) / total_tokens
    report = {
        "stage": "unreviewed_candidate_diagnostic_domain_validation",
        "checkpoint_step": checkpoint_manifest["step"],
        "checkpoint_manifest_sha256": sha256_file(args.checkpoint / "manifest.json"),
        "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "predicted_tokens": total_tokens,
        "weighted_loss": weighted_loss,
        "domains": results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "predicted_tokens": total_tokens, "weighted_loss": weighted_loss}), flush=True)


if __name__ == "__main__":
    main()
