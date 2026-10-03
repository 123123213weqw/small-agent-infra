#!/usr/bin/env python3
"""B2.3-B/C single-axis capacity and relation-depth extrapolation."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.phase0_ab import Episode, Record, _distractor
from experiments.phase0_b23_controlled import load_model, paired_episode, rollout


CAPACITY_LENGTHS = (128, 256, 512)
CAPACITIES = (4, 8, 12, 16, 24, 32)
DEPTH_LENGTHS = (128, 256)
HOPS = (2, 3, 4, 6, 8)
DEPTH_CAPACITY = 12


def relation_episode(index: int, length: int, hops: int) -> Episode:
    """Generate a multi-hop episode with the original distractor distribution."""
    rng = random.Random(23_400_000 + length * 100_000 + hops * 10_000 + index)
    # Keep the identifier/namespace shape identical to training (e<digits>).
    # The earlier exploratory generator used a "depth_..." namespace, which
    # changed every rendered entity prefix and invalidated its two-hop control.
    eid = f"e{length * 1_000_000 + hops * 10_000 + index}"
    slots: list[Record | None] = [None] * length
    # Match the original task: required records never occupy the endpoints.
    nodes: list[str] = []
    while len(nodes) < hops + 1:
        token = f"node_{rng.randrange(1_000_000)}"
        if token not in nodes:
            nodes.append(token)
    positions = sorted(rng.sample(range(1, length - 1), hops))
    required: list[str] = []
    for hop, position in enumerate(positions):
        a, b = nodes[hop], nodes[hop + 1]
        record = Record(
            uid=f"{eid}:r{position}",
            position=position,
            kind="edge",
            key=a,
            value=b,
            refs=(a, b),
        )
        slots[position] = record
        required.append(record.uid)
    for position in range(length):
        if slots[position] is None:
            slots[position] = _distractor(eid, position, rng)
    return Episode(
        eid=eid,
        task="relation_chain",
        records=tuple(slots),  # type: ignore[arg-type]
        clauses=(tuple(required),),
        goal_tokens=frozenset((nodes[0], nodes[-1])),
        target_key=nodes[0],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--axis", choices=("capacity", "depth"), required=True)
    parser.add_argument("--model-seed", type=int, choices=(3, 4, 5), required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model, config = load_model(root, args.model_seed, device)
    rows = []

    if args.axis == "capacity":
        conditions = [
            (length, capacity, 2)
            for length in CAPACITY_LENGTHS
            for capacity in CAPACITIES
        ]
    else:
        conditions = [
            (length, DEPTH_CAPACITY, hops)
            for length in DEPTH_LENGTHS
            for hops in HOPS
        ]

    for length, capacity, hops in conditions:
        if args.axis == "capacity":
            episodes = [paired_episode(index, length, "matched") for index in range(args.episodes)]
        else:
            # Reuse the already validated B2.3-A episodes as the exact two-hop
            # control.  Only unseen depths use the generalized generator.
            episodes = (
                [paired_episode(index, length, "matched") for index in range(args.episodes)]
                if hops == 2
                else [relation_episode(index, length, hops) for index in range(args.episodes)]
            )
        started = time.perf_counter()
        metrics = rollout(model, config, episodes, capacity, device)
        row = {
            "axis": args.axis,
            "model_seed": args.model_seed,
            "length": length,
            "capacity": capacity,
            "hops": hops,
            "episodes": args.episodes,
            "seconds": time.perf_counter() - started,
            **metrics,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
