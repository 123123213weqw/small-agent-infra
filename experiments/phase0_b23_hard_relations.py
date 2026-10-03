#!/usr/bin/env python3
"""B2.3-D hard relation distractor evaluation.

Required and distractor records use the same edge template.  The two variants
separate trivial goal-entity filtering from dead-end topology discrimination.
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

from experiments.phase0_ab import Episode, Record, run_episode
from experiments.phase0_b21_data import B21Renderer, make_decision_group
from experiments.phase0_b21_train import predict_groups
from experiments.phase0_b23_axes import relation_episode
from experiments.phase0_b23_controlled import load_model


LENGTH = 256
CAPACITY = 12
HOPS = 4
DECOY_COUNTS = (0, 4, 8, 16, 32, 64)


def ordered_required_edges(episode: Episode) -> list[Record]:
    return sorted(
        (record for record in episode.records if record.uid in episode.required_ids),
        key=lambda record: record.position,
    )


def has_directed_path(edges: list[tuple[str, str]], start: str, end: str) -> bool:
    graph: dict[str, list[str]] = {}
    for a, b in edges:
        graph.setdefault(a, []).append(b)
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        if node == end:
            return True
        for nxt in graph.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def hard_episode(index: int, decoy_count: int, variant: str) -> tuple[Episode, frozenset[str]]:
    base = relation_episode(index, LENGTH, HOPS)
    required_edges = ordered_required_edges(base)
    chain_nodes = [required_edges[0].refs[0], *[edge.refs[1] for edge in required_edges]]
    start, end = chain_nodes[0], chain_nodes[-1]
    available = [record.position for record in base.records if record.uid not in base.required_ids]
    rng = random.Random(23_500_000 + index)
    rng.shuffle(available)
    positions = available[:decoy_count]
    used_nodes = set(chain_nodes)
    decoy_edges: list[tuple[str, str]] = []
    records = list(base.records)
    decoy_ids: set[str] = set()

    def new_node() -> str:
        while True:
            token = f"node_{rng.randrange(1_000_000)}"
            if token not in used_nodes:
                used_nodes.add(token)
                return token

    for decoy_index, position in enumerate(positions):
        if variant == "disconnected":
            a, b = new_node(), new_node()
        elif variant == "spoke":
            anchor = chain_nodes[decoy_index % len(chain_nodes)]
            leaf = new_node()
            # Every decoy touches exactly one true-chain node. Alternating the
            # direction prevents a direction-only shortcut.
            a, b = (anchor, leaf) if decoy_index % 2 == 0 else (leaf, anchor)
        else:
            raise ValueError(variant)
        old = records[position]
        record = replace(
            old,
            kind="edge",
            key=a,
            value=b,
            refs=(a, b),
            consumed=False,
            unresolved=False,
        )
        records[position] = record
        decoy_ids.add(record.uid)
        decoy_edges.append((a, b))

    if has_directed_path(decoy_edges, start, end):
        raise AssertionError("decoy-only graph accidentally connects target endpoints")
    if any(edge in {(item.refs[0], item.refs[1]) for item in required_edges} for edge in decoy_edges):
        raise AssertionError("decoy duplicates a required edge")
    return replace(base, records=tuple(records)), frozenset(decoy_ids)


@torch.no_grad()
def rollout(model, config, prepared: list[tuple[Episode, frozenset[str]]], device: torch.device):
    episodes = [item[0] for item in prepared]
    decoy_sets = [item[1] for item in prepared]
    renderers = [B21Renderer("D", f"entity_{episode.eid}") for episode in episodes]
    states = [
        {
            "memory": [],
            "evictions": 0,
            "correct": 0,
            "critical_errors": 0,
            "critical_risk": 0,
            "first": None,
            "auc_wins": 0.0,
            "auc_pairs": 0,
            "safe_margins": [],
        }
        for _ in episodes
    ]
    for position in range(LENGTH):
        pending = []
        for index, (episode, renderer, state) in enumerate(zip(episodes, renderers, states)):
            candidate = episode.records[position]
            competing = [*state["memory"], candidate]
            if len(competing) <= CAPACITY:
                state["memory"] = competing
                continue
            group = make_decision_group(
                episode,
                competing,
                candidate,
                split="b23_hard_relations",
                capacity=CAPACITY,
                hard_negative_count=len(decoy_sets[index]),
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

            required_scores = [score for score, record in zip(scores, competing) if record.uid in episodes[index].required_ids]
            decoy_scores = [score for score, record in zip(scores, competing) if record.uid in decoy_sets[index]]
            if required_scores and decoy_scores:
                state["safe_margins"].append(min(required_scores) - min(decoy_scores))
                for required_score in required_scores:
                    for decoy_score in decoy_scores:
                        state["auc_wins"] += float(required_score > decoy_score) + 0.5 * float(required_score == decoy_score)
                        state["auc_pairs"] += 1
            state["memory"] = [record for j, record in enumerate(competing) if j != selected]

    successes, recalls, fifo, oracle, retained_decoys = [], [], [], [], []
    first_positions = []
    all_margins = []
    for index, (episode, decoys, state) in enumerate(zip(episodes, decoy_sets, states)):
        retained = {record.uid for record in state["memory"]}
        successes.append(float(episode.required_ids.issubset(retained)))
        recalls.append(len(episode.required_ids & retained) / len(episode.required_ids))
        retained_decoys.append(len(decoys & retained))
        fifo.append(run_episode(episode, CAPACITY, "fifo", seed=index).success)
        oracle.append(run_episode(episode, CAPACITY, "oracle", seed=index).success)
        all_margins.extend(state["safe_margins"])
        if state["first"] is not None:
            first_positions.append(state["first"])
    total_evictions = sum(state["evictions"] for state in states)
    total_risk = sum(state["critical_risk"] for state in states)
    total_errors = sum(state["critical_errors"] for state in states)
    auc_pairs = sum(state["auc_pairs"] for state in states)
    return {
        "success": sum(successes) / len(successes),
        "required_recall": sum(recalls) / len(recalls),
        "decision_accuracy": sum(state["correct"] for state in states) / max(1, total_evictions),
        "critical_error_rate": total_errors / max(1, total_risk),
        "episodes_with_critical_error": sum(state["critical_errors"] > 0 for state in states) / len(states),
        "median_first_critical_error": statistics.median(first_positions) if first_positions else None,
        "mean_retained_decoys": sum(retained_decoys) / len(retained_decoys),
        "required_decoy_auc": sum(state["auc_wins"] for state in states) / auc_pairs if auc_pairs else None,
        "mean_safe_margin": sum(all_margins) / len(all_margins) if all_margins else None,
        "positive_safe_margin_rate": sum(margin > 0 for margin in all_margins) / len(all_margins) if all_margins else None,
        "fifo_success": sum(fifo) / len(fifo),
        "oracle_success": sum(oracle) / len(oracle),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--variant", choices=("disconnected", "spoke"), required=True)
    parser.add_argument("--model-seed", type=int, choices=(3, 4, 5), required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model, config = load_model(root, args.model_seed, device)
    rows = []
    for decoy_count in DECOY_COUNTS:
        prepared = [hard_episode(index, decoy_count, args.variant) for index in range(args.episodes)]
        started = time.perf_counter()
        metrics = rollout(model, config, prepared, device)
        row = {
            "variant": args.variant,
            "model_seed": args.model_seed,
            "length": LENGTH,
            "capacity": CAPACITY,
            "hops": HOPS,
            "decoy_count": decoy_count,
            "episodes": args.episodes,
            "seconds": time.perf_counter() - started,
            **metrics,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
