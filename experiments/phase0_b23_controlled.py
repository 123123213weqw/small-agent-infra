#!/usr/bin/env python3
"""Controlled B2.3 boundary check with paired evaluation episodes.

This experiment changes one variable at a time: sequence length and memory
capacity.  All model seeds see byte-identical episodes.  The relation task is
fixed to two hops and no extra hard-edge sweep is mixed into the result.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.phase0_ab import Episode, Record, make_episode, run_episode
from experiments.phase0_b21_data import B21Renderer, make_decision_group
from experiments.phase0_b21_model import UtilitySetModel
from experiments.phase0_b21_train import load_config, predict_groups, resolve_seed_config


CHECKPOINTS = {
    3: ("configs/b2_2_robust/C_robust_confirm_seed3.json", "checkpoints/b2_2_robust/C_robust_confirm_seed3/best.pt"),
    4: ("configs/b2_2_robust/C_robust_confirm_seed4.json", "checkpoints/b2_2_robust/C_robust_confirm_seed4/best.pt"),
    5: ("configs/b2_2_robust/C_robust_confirm_seed5.json", "checkpoints/b2_2_robust/C_robust_confirm_seed5/best.pt"),
}
LENGTHS = (64, 96, 128, 192, 256, 384, 512)
CAPACITIES = (4, 8, 12)


def paired_episode(index: int, length: int, variant: str) -> Episode:
    # Seed deliberately excludes model_seed and variant.  Every model/variant
    # receives the same chain endpoints and required-record positions.
    rng = random.Random(23_300_000 + length * 10_000 + index)
    episode = make_episode(index=length * 1_000_000 + index, length=length, rng=rng, task="relation_chain")
    if variant == "matched":
        return episode
    required = episode.required_ids
    records: list[Record] = []
    irrelevant_index = 0
    for record in episode.records:
        if record.uid in required:
            records.append(record)
            continue
        if variant == "clean":
            records.append(
                replace(
                    record,
                    kind="log",
                    key=f"noise_{episode.eid}_{record.position}",
                    value=f"irrelevant_{record.position}",
                    refs=(),
                    consumed=False,
                    unresolved=False,
                )
            )
        elif variant == "conflict":
            # Reproduce the suspected confound in a paired way: exactly one in
            # five irrelevant records is worded as an unfinished intermediate
            # even though the symbolic target does not require it.
            make_open = irrelevant_index % 5 == 0
            records.append(
                replace(
                    record,
                    kind="intermediate" if make_open else "log",
                    key=f"noise_{episode.eid}_{record.position}",
                    value="partial_result" if make_open else f"irrelevant_{record.position}",
                    refs=(),
                    consumed=not make_open,
                    unresolved=make_open,
                )
            )
            irrelevant_index += 1
        else:
            raise ValueError(variant)
    return replace(episode, records=tuple(records))


def load_model(root: Path, seed: int, device: torch.device):
    config_path, checkpoint_path = CHECKPOINTS[seed]
    config = resolve_seed_config(load_config(root / config_path))
    model = UtilitySetModel(config).to(device)
    payload = torch.load(root / checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, config


@torch.no_grad()
def rollout(model, config, episodes: list[Episode], capacity: int, device: torch.device):
    renderers = [B21Renderer("D", f"entity_{episode.eid}") for episode in episodes]
    states = [
        {
            "memory": [],
            "evictions": 0,
            "correct": 0,
            "critical_errors": 0,
            "critical_risk": 0,
            "first": None,
        }
        for _ in episodes
    ]
    for position in range(len(episodes[0].records)):
        pending = []
        for index, (episode, renderer, state) in enumerate(zip(episodes, renderers, states)):
            candidate = episode.records[position]
            competing = [*state["memory"], candidate]
            if len(competing) <= capacity:
                state["memory"] = competing
                continue
            group = make_decision_group(
                episode,
                competing,
                candidate,
                split="b23_controlled",
                capacity=capacity,
                hard_negative_count=0,
                template_family="D",
                behavior_policy="model",
                decision_id=state["evictions"] + 1,
                renderer=renderer,
            )
            pending.append((index, group, competing))
        predictions = predict_groups(model, [item[1] for item in pending], config, device) if pending else []
        for (index, group, competing), scores in zip(pending, predictions):
            state = states[index]
            selected = min(range(len(competing)), key=lambda j: (scores[j], group["records"][j]["event_index"]))
            minimum = min(group["utilities"])
            at_risk = any(value > minimum for value in group["utilities"])
            critical_error = group["utilities"][selected] > minimum
            state["critical_risk"] += int(at_risk)
            state["critical_errors"] += int(critical_error)
            state["correct"] += int(selected in group["oracle_eviction_indices"])
            state["evictions"] += 1
            if critical_error and state["first"] is None:
                state["first"] = position
            state["memory"] = [record for j, record in enumerate(competing) if j != selected]

    successes, recalls, fifo, oracle = [], [], [], []
    first_positions = []
    for index, (episode, state) in enumerate(zip(episodes, states)):
        retained = {record.uid for record in state["memory"]}
        successes.append(float(episode.required_ids.issubset(retained)))
        recalls.append(len(episode.required_ids & retained) / len(episode.required_ids))
        fifo.append(run_episode(episode, capacity, "fifo", seed=index).success)
        oracle.append(run_episode(episode, capacity, "oracle", seed=index).success)
        if state["first"] is not None:
            first_positions.append(state["first"])
    total_evictions = sum(state["evictions"] for state in states)
    total_risk = sum(state["critical_risk"] for state in states)
    total_errors = sum(state["critical_errors"] for state in states)
    return {
        "success": sum(successes) / len(successes),
        "required_recall": sum(recalls) / len(recalls),
        "decision_accuracy": sum(state["correct"] for state in states) / max(1, total_evictions),
        "critical_error_rate": total_errors / max(1, total_risk),
        "critical_errors": total_errors,
        "critical_risk_decisions": total_risk,
        "episodes_with_critical_error": sum(state["critical_errors"] > 0 for state in states) / len(states),
        "median_first_critical_error": statistics.median(first_positions) if first_positions else None,
        "fifo_success": sum(fifo) / len(fifo),
        "oracle_success": sum(oracle) / len(oracle),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--model-seed", type=int, choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--variant", choices=("matched", "clean", "conflict"), required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model, config = load_model(root, args.model_seed, device)
    rows = []
    for length in LENGTHS:
        episodes = [paired_episode(index, length, args.variant) for index in range(args.episodes)]
        for capacity in CAPACITIES:
            started = time.perf_counter()
            metrics = rollout(model, config, episodes, capacity, device)
            row = {
                "model_seed": args.model_seed,
                "variant": args.variant,
                "length": length,
                "capacity": capacity,
                "episodes": args.episodes,
                "seconds": time.perf_counter() - started,
                **metrics,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
