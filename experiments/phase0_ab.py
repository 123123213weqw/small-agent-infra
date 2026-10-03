#!/usr/bin/env python3
"""Phase 0A/0B: symbolic selective-memory validation.

0A asks whether a future oracle can beat FIFO at the same memory capacity.
0B asks whether a small predictor, restricted to decision-time information, can
recover part of that oracle advantage on held-out episodes.

The symbolic reader has deliberately simple 0/1 clause loss.  This isolates
retention policy from language modeling and makes every counterfactual label
auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TASKS = (
    "delayed_query",
    "state_overwrite",
    "long_instruction",
    "completed_intermediate",
    "unresolved_subgoal",
    "relation_chain",
)

KINDS = ("fact", "state", "instruction", "intermediate", "subgoal", "edge", "log")


@dataclass(frozen=True)
class Record:
    uid: str
    position: int
    kind: str
    key: str
    value: str
    refs: tuple[str, ...] = ()
    consumed: bool = False
    unresolved: bool = False


@dataclass(frozen=True)
class Episode:
    eid: str
    task: str
    records: tuple[Record, ...]
    # Each clause is satisfied only when all listed records survive.
    clauses: tuple[tuple[str, ...], ...]
    goal_tokens: frozenset[str]
    target_key: str = ""

    @property
    def required_ids(self) -> frozenset[str]:
        return frozenset(uid for clause in self.clauses for uid in clause)


@dataclass
class RunMetrics:
    success: float
    clause_accuracy: float
    required_recall: float
    eviction_accuracy: float
    eviction_regret: float
    stale_value_rate: float


def _record(eid: str, pos: int, kind: str, key: str, value: str, **kw: object) -> Record:
    return Record(uid=f"{eid}:r{pos}", position=pos, kind=kind, key=key, value=value, **kw)


def _distractor(eid: str, pos: int, rng: random.Random) -> Record:
    kind = rng.choices(
        ["fact", "state", "instruction", "intermediate", "subgoal", "edge", "log"],
        weights=[24, 18, 8, 14, 10, 14, 12],
        k=1,
    )[0]
    key = f"k{rng.randrange(10_000)}"
    value = f"v{rng.randrange(10_000)}"
    if kind == "edge":
        a, b = f"n{rng.randrange(1_000)}", f"n{rng.randrange(1_000)}"
        return _record(eid, pos, kind, a, b, refs=(a, b))
    return _record(
        eid,
        pos,
        kind,
        key,
        value,
        consumed=(kind == "intermediate" and rng.random() < 0.75),
        unresolved=(kind == "subgoal" and rng.random() < 0.20),
    )


def make_episode(index: int, length: int, rng: random.Random, task: str | None = None) -> Episode:
    """Generate one episode without embedding the label in the record type."""
    task = task or rng.choice(TASKS)
    eid = f"e{index}"
    slots: list[Record | None] = [None] * length
    clauses: list[tuple[str, ...]] = []
    goal: set[str] = set()
    target_key = ""

    def put(pos: int, kind: str, key: str, value: str, **kw: object) -> Record:
        if slots[pos] is not None:
            raise ValueError(f"occupied position {pos}")
        rec = _record(eid, pos, kind, key, value, **kw)
        slots[pos] = rec
        return rec

    if task == "delayed_query":
        target_key = f"lookup_{rng.randrange(1_000_000)}"
        # Mix long-delay and recent targets so FIFO is a real baseline rather
        # than a guaranteed-zero straw man.
        pos = rng.randrange(1, length - 1)
        target = put(pos, "fact", target_key, f"secret_{rng.randrange(1_000_000)}")
        clauses.append((target.uid,))
        # Some future queries are not announced. This creates an honest ceiling
        # on predictability while the oracle can still measure their value.
        if rng.random() < 0.50:
            goal.add(target_key)
        else:
            goal.add("remember_for_possible_followup")

    elif task == "state_overwrite":
        target_key = f"state_{rng.randrange(1_000_000)}"
        count = rng.randint(3, 5)
        positions = sorted(rng.sample(range(1, length - 1), count))
        updates = [put(p, "state", target_key, f"value_{j}_{rng.randrange(10_000)}") for j, p in enumerate(positions)]
        clauses.append((updates[-1].uid,))
        goal.add(target_key)

    elif task == "long_instruction":
        target_key = "output_format"
        p1 = rng.randrange(1, max(2, length // 5))
        p2 = rng.randrange(max(p1 + 1, length // 3), max(p1 + 2, 2 * length // 3))
        instruction = put(p1, "instruction", target_key, "json")
        answer_key = f"answer_{rng.randrange(1_000_000)}"
        answer = put(p2, "fact", answer_key, f"result_{rng.randrange(1_000_000)}")
        clauses.extend(((instruction.uid,), (answer.uid,)))
        goal.update((target_key, "json", answer_key))

    elif task == "completed_intermediate":
        # Consumed intermediates are negatives; an unresolved item with the same
        # broad kind is the record that must survive.
        target_key = f"pending_{rng.randrange(1_000_000)}"
        p = rng.randrange(1, length - 1)
        pending = put(p, "intermediate", target_key, "partial_result", consumed=False, unresolved=True)
        clauses.append((pending.uid,))
        goal.add(target_key)
        for q in rng.sample([i for i in range(1, length - 1) if slots[i] is None], k=min(3, max(1, length // 12))):
            put(q, "intermediate", f"tmp_{q}", f"done_{q}", consumed=True, unresolved=False)

    elif task == "unresolved_subgoal":
        target_key = f"subgoal_{rng.randrange(1_000_000)}"
        p1, p2 = sorted(rng.sample(range(1, length - 1), 2))
        subgoal = put(p1, "subgoal", target_key, "open", unresolved=True)
        evidence_key = f"evidence_{rng.randrange(1_000_000)}"
        evidence = put(p2, "fact", evidence_key, "needed_for_final")
        clauses.append((subgoal.uid, evidence.uid))
        goal.update((target_key, evidence_key))

    elif task == "relation_chain":
        start, middle, end = (f"node_{rng.randrange(1_000_000)}" for _ in range(3))
        target_key = start
        positions = sorted(rng.sample(range(1, length - 1), 2))
        edge1 = put(positions[0], "edge", start, middle, refs=(start, middle))
        edge2 = put(positions[1], "edge", middle, end, refs=(middle, end))
        clauses.append((edge1.uid, edge2.uid))
        goal.update((start, end))

    else:
        raise ValueError(task)

    for pos, rec in enumerate(slots):
        if rec is None:
            slots[pos] = _distractor(eid, pos, rng)
    return Episode(
        eid=eid,
        task=task,
        records=tuple(slots),  # type: ignore[arg-type]
        clauses=tuple(clauses),
        goal_tokens=frozenset(goal),
        target_key=target_key,
    )


def make_episodes(count: int, lengths: Sequence[int], seed: int, start_index: int = 0) -> list[Episode]:
    rng = random.Random(seed)
    episodes = []
    for i in range(count):
        # Round-robin task assignment prevents seed-dependent task imbalance.
        task = TASKS[i % len(TASKS)]
        episodes.append(make_episode(start_index + i, rng.choice(lengths), rng, task))
    return episodes


def symbolic_loss(episode: Episode, available_ids: set[str]) -> float:
    if not episode.clauses:
        return 0.0
    failed = sum(not set(clause).issubset(available_ids) for clause in episode.clauses)
    return failed / len(episode.clauses)


def counterfactual_utilities(episode: Episode, records: Sequence[Record], decision_pos: int) -> dict[str, float]:
    """Exact one-record mask labels using only future targets for supervision.

    Future records are included as future context, matching the decision/future
    boundary in the design document. They are never exposed to the predictor.
    """
    future_ids = {r.uid for r in episode.records if r.position > decision_pos}
    full_ids = {r.uid for r in records} | future_ids
    full_loss = symbolic_loss(episode, full_ids)
    return {
        r.uid: symbolic_loss(episode, full_ids - {r.uid}) - full_loss
        for r in records
    }


def _relation_features(record: Record, records: Sequence[Record], goal_tokens: frozenset[str]) -> tuple[float, float]:
    if record.kind != "edge":
        return 0.0, 0.0
    touches = float(bool(set(record.refs) & goal_tokens))
    graph: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        if rec.kind == "edge" and len(rec.refs) == 2:
            a, b = rec.refs
            graph[a].add(b)
            graph[b].add(a)
    if not record.refs:
        return touches, 0.0
    seen = set(record.refs)
    stack = list(record.refs)
    while stack:
        node = stack.pop()
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    connects_goals = float(len(seen & goal_tokens) >= 2)
    return touches, connects_goals


FEATURE_NAMES = (
    *(f"kind={kind}" for kind in KINDS),
    "age_fraction",
    "position_fraction",
    "is_new_candidate",
    "goal_key_match",
    "goal_value_match",
    "goal_ref_match",
    "same_key_count",
    "newer_same_key",
    "older_same_key",
    "consumed",
    "unresolved",
    "edge_touches_goal",
    "edge_component_connects_goals",
    "capacity_pressure",
)


def features(episode: Episode, record: Record, records: Sequence[Record], decision_pos: int, capacity: int) -> np.ndarray:
    same = [r for r in records if r.key == record.key and r.uid != record.uid]
    touches, connects = _relation_features(record, records, episode.goal_tokens)
    # Normalize only by the observed prefix. Using final episode length here
    # would leak information that is generally unavailable at inference.
    denom = max(1, decision_pos)
    result = [float(record.kind == kind) for kind in KINDS]
    result.extend(
        (
            (decision_pos - record.position) / denom,
            record.position / denom,
            float(record.position == decision_pos),
            float(record.key in episode.goal_tokens),
            float(record.value in episode.goal_tokens),
            float(bool(set(record.refs) & episode.goal_tokens)),
            min(1.0, len(same) / 4.0),
            float(any(r.position > record.position for r in same)),
            float(any(r.position < record.position for r in same)),
            float(record.consumed),
            float(record.unresolved),
            touches,
            connects,
            len(records) / max(1, capacity),
        )
    )
    return np.asarray(result, dtype=np.float32)


class UtilityPredictor:
    def __init__(self, seed: int):
        self.model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                batch_size=256,
                learning_rate_init=2e-3,
                max_iter=100,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=8,
                random_state=seed,
            ),
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        # Rare positive labels are oversampled, while targets remain the actual
        # counterfactual utility values used by direct MSE regression.
        pos = np.flatnonzero(y > 0)
        neg = np.flatnonzero(y <= 0)
        if len(pos) == 0:
            raise ValueError("training data contains no positive utility labels")
        rng = np.random.default_rng(12345)
        extra = rng.choice(pos, size=max(0, min(len(neg), 8 * len(pos)) - len(pos)), replace=True)
        idx = np.concatenate((np.arange(len(y)), extra))
        rng.shuffle(idx)
        self.model.fit(x[idx], y[idx])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


def choose_eviction(
    policy: str,
    episode: Episode,
    records: Sequence[Record],
    decision_pos: int,
    capacity: int,
    rng: random.Random,
    predictor: UtilityPredictor | None,
) -> tuple[int, dict[str, float]]:
    utilities = counterfactual_utilities(episode, records, decision_pos)
    if policy == "fifo":
        index = min(range(len(records)), key=lambda i: records[i].position)
    elif policy == "random":
        index = rng.randrange(len(records))
    elif policy == "oracle":
        index = min(range(len(records)), key=lambda i: (utilities[records[i].uid], records[i].position))
    elif policy == "predicted":
        if predictor is None:
            raise ValueError("predicted policy requires a fitted predictor")
        x = np.stack([features(episode, r, records, decision_pos, capacity) for r in records])
        scores = predictor.predict(x)
        index = min(range(len(records)), key=lambda i: (float(scores[i]), records[i].position))
    else:
        raise ValueError(policy)
    return index, utilities


def run_episode(
    episode: Episode,
    capacity: int,
    policy: str,
    seed: int,
    predictor: UtilityPredictor | None = None,
    collect: list[tuple[np.ndarray, float]] | None = None,
    sample_rate: float = 1.0,
) -> RunMetrics:
    rng = random.Random(seed)
    memory: list[Record] = []
    correct_evictions = 0
    total_evictions = 0
    regret = 0.0
    for rec in episode.records:
        competing = memory + [rec]
        if len(competing) <= capacity:
            memory = competing
            continue
        utilities = counterfactual_utilities(episode, competing, rec.position)
        if collect is not None and rng.random() < sample_rate:
            positives = [r for r in competing if utilities[r.uid] > 0]
            negatives = [r for r in competing if utilities[r.uid] <= 0]
            rng.shuffle(negatives)
            selected = positives + negatives[: max(3, len(positives))]
            for item in selected:
                collect.append((features(episode, item, competing, rec.position, capacity), utilities[item.uid]))
        evict_idx, utilities = choose_eviction(policy, episode, competing, rec.position, capacity, rng, predictor)
        minimum = min(utilities.values())
        removed_utility = utilities[competing[evict_idx].uid]
        correct_evictions += int(math.isclose(removed_utility, minimum))
        total_evictions += 1
        regret += removed_utility - minimum
        memory = [r for i, r in enumerate(competing) if i != evict_idx]

    memory_ids = {r.uid for r in memory}
    clause_scores = [float(set(clause).issubset(memory_ids)) for clause in episode.clauses]
    required = episode.required_ids
    recall = len(required & memory_ids) / max(1, len(required))
    stale = 0.0
    if episode.task == "state_overwrite":
        final_required = next(iter(required))
        stale = float(final_required not in memory_ids and any(r.key == episode.target_key for r in memory))
    return RunMetrics(
        success=float(all(clause_scores)),
        clause_accuracy=sum(clause_scores) / max(1, len(clause_scores)),
        required_recall=recall,
        eviction_accuracy=correct_evictions / max(1, total_evictions),
        eviction_regret=regret / max(1, total_evictions),
        stale_value_rate=stale,
    )


def build_training_set(episodes: Sequence[Episode], capacities: Sequence[int], seed: int, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[np.ndarray, float]] = []
    collectors = ("fifo", "random", "oracle")
    for i, episode in enumerate(episodes):
        capacity = capacities[i % len(capacities)]
        policy = collectors[i % len(collectors)]
        run_episode(
            episode,
            capacity,
            policy,
            seed=seed * 1_000_003 + i,
            collect=rows,
            sample_rate=sample_rate,
        )
    x = np.stack([row[0] for row in rows])
    y = np.asarray([row[1] for row in rows], dtype=np.float32)
    return x, y


def aggregate(items: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(items, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def evaluate(
    episodes: Sequence[Episode],
    capacities: Sequence[int],
    predictor: UtilityPredictor,
    seed: int,
    split: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, episode in enumerate(episodes):
        capacity = capacities[i % len(capacities)]
        for policy in ("random", "fifo", "oracle", "predicted"):
            metrics = run_episode(
                episode,
                capacity,
                policy,
                seed=seed * 10_000_019 + i * 17,
                predictor=predictor,
            )
            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "episode": episode.eid,
                    "task": episode.task,
                    "length": len(episode.records),
                    "capacity": capacity,
                    "policy": policy,
                    **asdict(metrics),
                }
            )
    return rows


def summarize(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"groups": {}, "by_task": {}, "decision": {}}
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["split"]), str(row["policy"]))].append(row)
    for (split, policy), values in sorted(groups.items()):
        out: dict[str, dict[str, float]] = {}
        for metric in ("success", "clause_accuracy", "required_recall", "eviction_accuracy", "eviction_regret", "stale_value_rate"):
            mean, std = aggregate([float(v[metric]) for v in values])
            out[metric] = {"mean": mean, "std": std}
        summary["groups"][f"{split}/{policy}"] = out

    for split in sorted({str(r["split"]) for r in rows}):
        for task in TASKS:
            task_scores = {}
            for policy in ("fifo", "oracle", "predicted"):
                values = [
                    float(r["success"])
                    for r in rows
                    if r["split"] == split and r["task"] == task and r["policy"] == policy
                ]
                task_scores[policy] = float(np.mean(values))
            task_scores["oracle_fifo_gap"] = task_scores["oracle"] - task_scores["fifo"]
            summary["by_task"][f"{split}/{task}"] = task_scores

    decisions: dict[str, object] = {}
    for split in sorted({str(r["split"]) for r in rows}):
        score = {
            policy: np.mean([float(r["success"]) for r in rows if r["split"] == split and r["policy"] == policy])
            for policy in ("fifo", "oracle", "predicted")
        }
        gap = float(score["oracle"] - score["fifo"])
        recovered = float((score["predicted"] - score["fifo"]) / gap) if gap > 1e-12 else float("nan")
        task_gaps = [
            summary["by_task"][f"{split}/{task}"]["oracle_fifo_gap"]
            for task in TASKS
        ]
        tasks_with_gap = sum(task_gap >= 0.05 for task_gap in task_gaps)
        seed_gaps: list[float] = []
        seed_recovered: list[float] = []
        for seed in sorted({int(r["seed"]) for r in rows if r["split"] == split}):
            seed_score = {
                policy: float(np.mean([
                    float(r["success"])
                    for r in rows
                    if r["split"] == split and int(r["seed"]) == seed and r["policy"] == policy
                ]))
                for policy in ("fifo", "oracle", "predicted")
            }
            seed_gap = seed_score["oracle"] - seed_score["fifo"]
            seed_gaps.append(seed_gap)
            if seed_gap > 1e-12:
                seed_recovered.append((seed_score["predicted"] - seed_score["fifo"]) / seed_gap)
        _, gap_seed_std = aggregate(seed_gaps)
        _, recovered_seed_std = aggregate(seed_recovered)
        decisions[split] = {
            "fifo_success": float(score["fifo"]),
            "oracle_success": float(score["oracle"]),
            "predicted_success": float(score["predicted"]),
            "oracle_fifo_gap": gap,
            "oracle_fifo_gap_seed_std": gap_seed_std,
            "predicted_gap_recovered": recovered,
            "predicted_gap_recovered_seed_std": recovered_seed_std,
            "tasks_with_oracle_gap_ge_5pp": tasks_with_gap,
            "phase_a_pass_10pp": bool(gap >= 0.10 and tasks_with_gap >= 4),
            "phase_b_pass_50pct": bool(recovered >= 0.50),
        }
    summary["decision"] = decisions
    return summary


def write_outputs(output: Path, rows: Sequence[dict[str, object]], summary: dict[str, object], metadata: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"metadata": metadata, **summary}
    (output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Phase 0A/0B symbolic validation", "", "## Decision summary", ""]
    lines.append("| Split | FIFO | Oracle | Predicted | Oracle−FIFO | Gap recovered | A | B |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|:---:|")
    for split, item in summary["decision"].items():
        lines.append(
            f"| {split} | {item['fifo_success']:.3f} | {item['oracle_success']:.3f} | "
            f"{item['predicted_success']:.3f} | {item['oracle_fifo_gap']:.3f}±{item['oracle_fifo_gap_seed_std']:.3f} | "
            f"{item['predicted_gap_recovered']:.1%}±{item['predicted_gap_recovered_seed_std']:.1%} | "
            f"{'PASS' if item['phase_a_pass_10pp'] else 'FAIL'} | "
            f"{'PASS' if item['phase_b_pass_50pct'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "A requires Oracle−FIFO >= 0.10 and a >= 0.05 gap on at least four task families. "
            "B requires the predictor to recover >= 50% of the aggregate gap.",
            "The oracle alone sees future counterfactual labels; predicted utility receives only decision-time features.",
            "",
            "## Success by task family",
            "",
            "| Split/task | FIFO | Oracle | Predicted | Oracle−FIFO |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, item in summary["by_task"].items():
        lines.append(
            f"| {key} | {item['fifo']:.3f} | {item['oracle']:.3f} | "
            f"{item['predicted']:.3f} | {item['oracle_fifo_gap']:.3f} |"
        )
    lines.append("")
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/phase0_ab"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--train-episodes", type=int, default=600)
    parser.add_argument("--test-episodes", type=int, default=240)
    parser.add_argument("--sample-rate", type=float, default=0.35)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds, args.train_episodes, args.test_episodes = 2, 120, 60

    all_rows: list[dict[str, object]] = []
    train_caps = (4, 8, 12)
    test_caps = (4, 8, 12)
    for seed in range(args.seeds):
        train = make_episodes(args.train_episodes, (32, 48, 64), 1000 + seed, seed * 1_000_000)
        x, y = build_training_set(train, train_caps, seed, args.sample_rate)
        predictor = UtilityPredictor(seed)
        predictor.fit(x, y)

        test_id = make_episodes(args.test_episodes, (48, 64), 20_000 + seed, 10_000_000 + seed * 1_000_000)
        test_ood = make_episodes(args.test_episodes, (96, 128), 30_000 + seed, 20_000_000 + seed * 1_000_000)
        all_rows.extend(evaluate(test_id, test_caps, predictor, seed, "id"))
        all_rows.extend(evaluate(test_ood, test_caps, predictor, seed, "ood_length"))
        print(f"seed={seed} train_rows={len(y)} positives={(y > 0).mean():.3%}", flush=True)

    summary = summarize(all_rows)
    metadata = {
        "seeds": args.seeds,
        "train_episodes_per_seed": args.train_episodes,
        "test_episodes_per_split_per_seed": args.test_episodes,
        "train_lengths": [32, 48, 64],
        "id_lengths": [48, 64],
        "ood_lengths": [96, 128],
        "capacities": list(test_caps),
        "feature_names": list(FEATURE_NAMES),
    }
    write_outputs(args.output, all_rows, summary, metadata)
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
