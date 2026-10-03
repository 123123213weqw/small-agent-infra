#!/usr/bin/env python3
"""Evaluate one B2.4 checkpoint on the frozen B2.3-D suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.phase0_b21_model import UtilitySetModel, parameter_count
from experiments.phase0_b21_train import load_config, resolve_seed_config
from experiments.phase0_b23_hard_relations import DECOY_COUNTS, hard_episode, rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = resolve_seed_config(load_config(root / args.config))
    device = torch.device("cuda")
    model = UtilitySetModel(config).to(device)
    payload = torch.load(root / args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"])
    model.eval()
    rows = []
    for variant in ("disconnected", "spoke"):
        for decoy_count in DECOY_COUNTS:
            prepared = [hard_episode(index, decoy_count, variant) for index in range(args.episodes)]
            row = {
                "name": config["name"],
                "parameters": parameter_count(model),
                "variant": variant,
                "decoy_count": decoy_count,
                **rollout(model, config, prepared, device),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
