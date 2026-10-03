#!/usr/bin/env python3
"""B2.3 long-horizon/capacity/relation stress sweep.

The sweep is procedural: it does not create a second dataset tree.  Each shard
writes one compact JSONL file and may be resumed safely after interruption.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.phase0_ab import Episode, Record
from experiments.phase0_b21_data import B21Renderer, make_decision_group
from experiments.phase0_b21_model import UtilitySetModel
from experiments.phase0_b21_train import load_config, predict_groups, resolve_seed_config


CHECKPOINTS = {
    3: ("configs/b2_2_robust/C_robust_confirm_seed3.json", "checkpoints/b2_2_robust/C_robust_confirm_seed3/best.pt"),
    4: ("configs/b2_2_robust/C_robust_confirm_seed4.json", "checkpoints/b2_2_robust/C_robust_confirm_seed4/best.pt"),
    5: ("configs/b2_2_robust/C_robust_confirm_seed5.json", "checkpoints/b2_2_robust/C_robust_confirm_seed5/best.pt"),
}


def make_relation_episode(index: int, length: int, hops: int, hard_edges: int, seed: int) -> Episode:
    if hops + hard_edges > length - 2:
        raise ValueError("episode is too short for requested required/hard edges")
    rng = random.Random(seed)
    eid = f"b23_{index}"
    slots: list[Record | None] = [None] * length
    nodes = [f"node_{index}_{j}" for j in range(hops + 1)]
    available = list(range(1, length - 1))
    chosen = rng.sample(available, hops + hard_edges)
    required_positions = sorted(chosen[:hops])
    hard_positions = chosen[hops:]
    required: list[str] = []

    def put(pos: int, kind: str, key: str, value: str, refs: tuple[str, ...] = ()) -> Record:
        record = Record(
            uid=f"{eid}:r{pos}",
            position=pos,
            kind=kind,
            key=key,
            value=value,
            refs=refs,
        )
        slots[pos] = record
        return record

    for hop, pos in enumerate(required_positions):
        record = put(pos, "edge", nodes[hop], nodes[hop + 1], (nodes[hop], nodes[hop + 1]))
        required.append(record.uid)

    # Spokes touch real chain nodes but cannot form an alternate endpoint path.
    for j, pos in enumerate(hard_positions):
        anchor = nodes[rng.randrange(len(nodes))]
        leaf = f"decoy_{index}_{j}"
        if j % 2:
            put(pos, "edge", leaf, anchor, (leaf, anchor))
        else:
            put(pos, "edge", anchor, leaf, (anchor, leaf))

    filler_kinds = ("fact", "state", "log", "intermediate", "subgoal")
    for pos in range(length):
        if slots[pos] is not None:
            continue
        kind = filler_kinds[rng.randrange(len(filler_kinds))]
        put(pos, kind, f"noise_{index}_{pos}", f"value_{rng.randrange(1_000_000)}")

    return Episode(
        eid=eid,
        task="relation_chain",
        records=tuple(slots),  # type: ignore[arg-type]
        clauses=(tuple(required),),
        goal_tokens=frozenset((nodes[0], nodes[-1])),
        target_key=nodes[0],
    )


def load_model(root: Path, seed: int, device: torch.device):
    config_path, checkpoint_path = CHECKPOINTS[seed]
    config = resolve_seed_config(load_config(root / config_path))
    model = UtilitySetModel(config).to(device)
    payload = torch.load(root / checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, config


@torch.no_grad()
def rollout(model, config, episodes: list[Episode], capacity: int, hard_edges: int, device: torch.device):
    renderers = [B21Renderer("D", f"entity_{episode.eid}") for episode in episodes]
    states = [
        {"memory": [], "evictions": 0, "correct": 0, "regret": 0.0, "required_evictions": 0, "first": None}
        for _ in episodes
    ]
    for position in range(max(len(episode.records) for episode in episodes)):
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
                split="b23_stress",
                capacity=capacity,
                hard_negative_count=hard_edges,
                template_family="D",
                behavior_policy="model",
                decision_id=state["evictions"] + 1,
                renderer=renderer,
            )
            pending.append((index, group, competing))
        predictions = predict_groups(model, [item[1] for item in pending], config, device) if pending else []
        for (index, group, competing), scores in zip(pending, predictions):
            state = states[index]
            chosen = min(range(len(competing)), key=lambda j: (scores[j], group["records"][j]["event_index"]))
            utility = group["utilities"][chosen]
            minimum = min(group["utilities"])
            state["correct"] += int(chosen in group["oracle_eviction_indices"])
            state["evictions"] += 1
            state["regret"] += utility - minimum
            if competing[chosen].uid in episodes[index].required_ids:
                state["required_evictions"] += 1
                if state["first"] is None:
                    state["first"] = position
            state["memory"] = [record for j, record in enumerate(competing) if j != chosen]

    success, recall = [], []
    first_positions = []
    for episode, state in zip(episodes, states):
        retained = {record.uid for record in state["memory"]}
        success.append(float(all(set(clause).issubset(retained) for clause in episode.clauses)))
        recall.append(len(episode.required_ids & retained) / len(episode.required_ids))
        if state["first"] is not None:
            first_positions.append(state["first"])
    total_evictions = sum(state["evictions"] for state in states)
    return {
        "success": sum(success) / len(success),
        "required_recall": sum(recall) / len(recall),
        "decision_accuracy": sum(state["correct"] for state in states) / max(1, total_evictions),
        "mean_regret": sum(state["regret"] for state in states) / max(1, total_evictions),
        "episodes_with_required_eviction": sum(state["required_evictions"] > 0 for state in states) / len(states),
        "median_first_required_eviction": statistics.median(first_positions) if first_positions else None,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--episodes-per-cell", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard id")
    root = args.root.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add(tuple(row["key"]))

    grid = [
        (length, capacity, hops, hard_edges)
        for length in (128, 256, 512, 1024)
        for capacity in (4, 8, 16, 32)
        for hops in (2, 4, 8, 16)
        for hard_edges in (0, 16, 32, 64)
    ]
    assigned = [(index, cell) for index, cell in enumerate(grid) if index % args.num_shards == args.shard_id]
    device = torch.device("cuda")
    step = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for model_seed in (3, 4, 5):
            model, config = load_model(root, model_seed, device)
            for condition_index, (length, capacity, hops, hard_edges) in assigned:
                key = (model_seed, length, capacity, hops, hard_edges)
                if key in completed:
                    continue
                started = time.perf_counter()
                episodes = [
                    make_relation_episode(
                        condition_index * 1000 + episode_index,
                        length,
                        hops,
                        hard_edges,
                        seed=23_000_000 + model_seed * 100_000 + condition_index * 100 + episode_index,
                    )
                    for episode_index in range(args.episodes_per_cell)
                ]
                metrics = rollout(model, config, episodes, capacity, hard_edges, device)
                row = {
                    "key": list(key),
                    "model_seed": model_seed,
                    "condition_index": condition_index,
                    "length": length,
                    "capacity": capacity,
                    "hops": hops,
                    "hard_edges": hard_edges,
                    "feasible_capacity": hops <= capacity,
                    "episodes": args.episodes_per_cell,
                    "seconds": time.perf_counter() - started,
                    **metrics,
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                step += 1
                print(f"shard={args.shard_id} step={step} key={key} success={row['success']:.4f} recall={row['required_recall']:.4f}", flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
