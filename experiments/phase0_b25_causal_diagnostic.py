#!/usr/bin/env python3
"""B2.5: separate visible-graph reasoning from online retention uncertainty.

The same frozen episodes are evaluated in their original order and with all
required path edges moved before the first memory-pressure decision.  A
prefix-only graph policy and a full-graph path solver provide causal and
hindsight references, respectively.  No model is trained by this script.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.phase0_ab import Episode, Record
from experiments.phase0_b21_data import B21Renderer, make_decision_group
from experiments.phase0_b21_model import UtilitySetModel
from experiments.phase0_b21_train import load_config, predict_groups, resolve_seed_config
from experiments.phase0_b23_hard_relations import CAPACITY, hard_episode, rollout


def path_first(episode: Episode) -> Episode:
    required = episode.required_ids
    ordered = [r for r in episode.records if r.uid in required]
    ordered += [r for r in episode.records if r.uid not in required]
    return replace(episode, records=tuple(replace(r, position=i) for i, r in enumerate(ordered)))


def prepressure_mix(episode: Episode, decoy_ids: frozenset[str]) -> Episode:
    """Expose the full path amid eight same-kind negatives before eviction.

    Unlike path_first, no position within the first 12 records uniquely marks
    a required edge.  All four path edges and eight decoys are shuffled with
    an episode-specific seed; the remaining records keep their relative order.
    """
    rng = random.Random(25_000_000 + int(episode.eid[1:]))
    required = [r for r in episode.records if r.uid in episode.required_ids]
    decoys = [r for r in episode.records if r.uid in decoy_ids]
    rng.shuffle(decoys)
    first = [*required, *decoys[: CAPACITY - len(required)]]
    if len(first) != CAPACITY:
        raise ValueError("not enough hard negatives for prepressure mix")
    rng.shuffle(first)
    first_ids = {r.uid for r in first}
    ordered = first + [r for r in episode.records if r.uid not in first_ids]
    return replace(episode, records=tuple(replace(r, position=i) for i, r in enumerate(ordered)))


def completion_at_pressure(episode: Episode, decoy_ids: frozenset[str]) -> Episode:
    """Make the thirteenth edge complete the path at the first eviction.

    Before that decision the memory contains three required edges and nine
    same-template decoys.  The model must evict one of those decoys rather than
    merely retaining every edge in the first capacity-sized prefix.
    """
    rng = random.Random(25_100_000 + int(episode.eid[1:]))
    required = [r for r in episode.records if r.uid in episode.required_ids]
    completion = required[int(episode.eid[1:]) % len(required)]
    decoys = [r for r in episode.records if r.uid in decoy_ids]
    rng.shuffle(decoys)
    first = [r for r in required if r.uid != completion.uid] + decoys[:9]
    if len(first) != CAPACITY:
        raise ValueError("not enough hard negatives for completion probe")
    rng.shuffle(first)
    front = [*first, completion]
    front_ids = {r.uid for r in front}
    ordered = front + [r for r in episode.records if r.uid not in front_ids]
    return replace(episode, records=tuple(replace(r, position=i) for i, r in enumerate(ordered)))


def endpoints(episode: Episode) -> tuple[str, str]:
    # Both endpoints are explicitly announced by the task goal.  Do not infer
    # them from required_ids, which would make this reference policy noncausal.
    start = episode.target_key
    remainder = episode.goal_tokens - {start}
    if len(remainder) != 1:
        raise ValueError("relation goal must identify exactly one destination")
    return start, next(iter(remainder))


def reachable(edges: list[Record], start: str, reverse: bool = False) -> set[str]:
    graph: dict[str, list[str]] = {}
    for r in edges:
        if r.kind != "edge" or len(r.refs) != 2:
            continue
        a, b = r.refs
        if reverse:
            a, b = b, a
        graph.setdefault(a, []).append(b)
    seen = {start}
    stack = [start]
    while stack:
        for nxt in graph.get(stack.pop(), []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def on_visible_path(edges: list[Record], start: str, end: str) -> set[str]:
    forward = reachable(edges, start)
    backward = reachable(edges, end, reverse=True)
    return {
        r.uid for r in edges
        if r.kind == "edge" and len(r.refs) == 2
        and r.refs[0] in forward and r.refs[1] in backward
    }


def full_graph_metrics(episodes: list[Episode]) -> dict[str, float]:
    path_success = []
    arrival_anchored = []
    for episode in episodes:
        start, end = endpoints(episode)
        path_success.append(float(episode.required_ids.issubset(on_visible_path(list(episode.records), start, end))))
        for r in episode.records:
            if r.uid not in episode.required_ids:
                continue
            prefix = [x for x in episode.records if x.position <= r.position]
            from_start = reachable(prefix, start)
            to_end = reachable(prefix, end, reverse=True)
            arrival_anchored.append(float(r.refs[0] in from_start or r.refs[1] in to_end))
    return {
        "full_graph_path_success": sum(path_success) / len(path_success),
        "required_edge_anchored_at_arrival": sum(arrival_anchored) / len(arrival_anchored),
    }


def prefix_graph_rollout(episodes: list[Episode], seed: int) -> dict[str, float]:
    successes, recalls = [], []
    for episode_index, episode in enumerate(episodes):
        start, end = endpoints(episode)
        memory: list[Record] = []
        rng = random.Random(seed + episode_index)
        for candidate in episode.records:
            competing = [*memory, candidate]
            if len(competing) <= CAPACITY:
                memory = competing
                continue
            forward = reachable(competing, start)
            backward = reachable(competing, end, reverse=True)
            scores = []
            for r in competing:
                if r.kind != "edge" or len(r.refs) != 2:
                    scores.append(0)
                elif r.refs[0] in forward and r.refs[1] in backward:
                    scores.append(4)  # Edge on an already visible complete path.
                elif r.refs[0] in forward or r.refs[1] in backward:
                    scores.append(2)  # Partially connected to a goal endpoint.
                else:
                    scores.append(1)  # Unknown edge; retain without hindsight bias.
            minimum = min(scores)
            eligible = [i for i, score in enumerate(scores) if score == minimum]
            remove = rng.choice(eligible)
            memory = [r for i, r in enumerate(competing) if i != remove]
        kept = {r.uid for r in memory}
        successes.append(float(episode.required_ids.issubset(kept)))
        recalls.append(len(episode.required_ids & kept) / len(episode.required_ids))
    return {
        "prefix_graph_success": sum(successes) / len(successes),
        "prefix_graph_required_recall": sum(recalls) / len(recalls),
    }


@torch.no_grad()
def completion_decision_metrics(model, config, episodes: list[Episode], decoy_sets: list[frozenset[str]], device):
    groups = []
    for episode, decoys in zip(episodes, decoy_sets):
        group = make_decision_group(
            episode,
            episode.records[: CAPACITY + 1],
            episode.records[CAPACITY],
            split="b25_completion_probe",
            capacity=CAPACITY,
            hard_negative_count=len(decoys),
            template_family="D",
            behavior_policy="model",
            decision_id=1,
            renderer=B21Renderer("D", f"entity_{episode.eid}"),
        )
        groups.append(group)
    predictions = predict_groups(model, groups, config, device)
    safe = 0
    auc_wins = 0.0
    auc_pairs = 0
    existing_wins = 0.0
    existing_pairs = 0
    candidate_wins = 0.0
    candidate_pairs = 0
    candidate_evictions = 0
    existing_required_evictions = 0
    for episode, decoys, scores in zip(episodes, decoy_sets, predictions):
        competing = episode.records[: CAPACITY + 1]
        selected = min(range(len(competing)), key=lambda j: (scores[j], competing[j].position))
        safe += int(competing[selected].uid in decoys)
        candidate_evictions += int(selected == CAPACITY)
        existing_required_evictions += int(selected != CAPACITY and competing[selected].uid in episode.required_ids)
        required_scores = [score for score, r in zip(scores, competing) if r.uid in episode.required_ids]
        existing_required_scores = [score for score, r in zip(scores[:CAPACITY], competing[:CAPACITY])
                                    if r.uid in episode.required_ids]
        decoy_scores = [score for score, r in zip(scores, competing) if r.uid in decoys]
        for a in required_scores:
            for b in decoy_scores:
                auc_wins += float(a > b) + 0.5 * float(a == b)
                auc_pairs += 1
        for a in existing_required_scores:
            for b in decoy_scores:
                existing_wins += float(a > b) + 0.5 * float(a == b)
                existing_pairs += 1
        for b in decoy_scores:
            a = scores[CAPACITY]
            candidate_wins += float(a > b) + 0.5 * float(a == b)
            candidate_pairs += 1
    return {
        "first_decision_safe_rate": safe / len(episodes),
        "first_decision_required_decoy_auc": auc_wins / auc_pairs,
        "first_decision_existing_auc": existing_wins / existing_pairs,
        "first_decision_candidate_auc": candidate_wins / candidate_pairs,
        "first_decision_candidate_eviction_rate": candidate_evictions / len(episodes),
        "first_decision_existing_required_eviction_rate": existing_required_evictions / len(episodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--decoys", nargs="+", type=int, default=[16, 64])
    parser.add_argument("--variants", nargs="+", default=["disconnected", "spoke"])
    parser.add_argument("--orders", nargs="+", default=["original", "path_first", "prepressure_mix", "completion_at_pressure"],
                        choices=["original", "path_first", "prepressure_mix", "completion_at_pressure"])
    parser.add_argument("--first-only", action="store_true", help="only score the completion-at-pressure first decision")
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
    for variant in args.variants:
        for decoy_count in args.decoys:
            base = [hard_episode(i, decoy_count, variant) for i in range(args.episodes)]
            for order in args.orders:
                if args.first_only and order != "completion_at_pressure":
                    raise ValueError("--first-only requires --orders completion_at_pressure")
                if order == "original":
                    prepared = base
                elif order == "path_first":
                    prepared = [(path_first(e), decoys) for e, decoys in base]
                elif order == "prepressure_mix":
                    prepared = [(prepressure_mix(e, decoys), decoys) for e, decoys in base]
                else:
                    prepared = [(completion_at_pressure(e, decoys), decoys) for e, decoys in base]
                episodes = [e for e, _ in prepared]
                if args.first_only:
                    row = {
                        "model": config["name"],
                        "variant": variant,
                        "decoy_count": decoy_count,
                        "order": order,
                        "episodes": len(episodes),
                        **completion_decision_metrics(model, config, episodes,
                                                      [decoys for _, decoys in prepared], device),
                    }
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
                    continue
                row = {
                    "model": config["name"],
                    "variant": variant,
                    "decoy_count": decoy_count,
                    "order": order,
                    "episodes": len(episodes),
                    **full_graph_metrics(episodes),
                    **prefix_graph_rollout(episodes, seed=26_000_000 + decoy_count),
                    **{f"model_{key}": value for key, value in rollout(model, config, prepared, device).items()
                       if key in {"success", "required_recall", "required_decoy_auc"}},
                }
                if order == "completion_at_pressure":
                    row.update(completion_decision_metrics(model, config, episodes,
                                                           [decoys for _, decoys in prepared], device))
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
