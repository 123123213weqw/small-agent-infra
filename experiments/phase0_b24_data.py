#!/usr/bin/env python3
"""Build the B2.4 relation-disambiguation training dataset.

The original B2.2 robust groups are retained verbatim.  An equal number of
targeted groups is added in which required and decoy edges coexist in the same
competition set and the total number of relation edges exceeds capacity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.phase0_ab import Episode, Record, _distractor
from experiments.phase0_b21_data import (
    B21Renderer,
    ROBUST_TRAIN_EDGE_TEMPLATES,
    ROBUST_VALIDATION_EDGE_TEMPLATES,
    make_decision_group,
    validate_group,
)


TRAIN_GROUPS = 36_000
VALIDATION_GROUPS = 4_800
LENGTHS = (32, 48, 64)
CAPACITIES = (4, 8, 12)
HOPS = (2, 3, 4)
DECOYS = (4, 8, 12, 16)
VARIANTS = ("disconnected", "spoke", "branch")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_hard_episode(
    episode_index: int,
    *,
    length: int,
    hops: int,
    decoy_count: int,
    variant: str,
    seed: int,
) -> tuple[Episode, frozenset[str]]:
    rng = random.Random(seed)
    eid = f"e{800_000_000 + episode_index}"
    slots: list[Record | None] = [None] * length
    used_nodes: set[str] = set()

    def new_node() -> str:
        while True:
            token = f"node_{rng.randrange(1_000_000)}"
            if token not in used_nodes:
                used_nodes.add(token)
                return token

    chain_nodes = [new_node() for _ in range(hops + 1)]
    positions = list(range(1, length - 1))
    rng.shuffle(positions)
    required_positions = sorted(positions[:hops])
    decoy_positions = positions[hops : hops + decoy_count]
    required_ids: list[str] = []
    for hop, position in enumerate(required_positions):
        a, b = chain_nodes[hop], chain_nodes[hop + 1]
        record = Record(
            uid=f"{eid}:r{position}",
            position=position,
            kind="edge",
            key=a,
            value=b,
            refs=(a, b),
        )
        slots[position] = record
        required_ids.append(record.uid)

    decoy_ids: set[str] = set()
    branch_state: tuple[str, str, int] | None = None
    for decoy_index, position in enumerate(decoy_positions):
        if variant == "disconnected":
            a, b = new_node(), new_node()
        elif variant == "spoke":
            anchor = chain_nodes[decoy_index % len(chain_nodes)]
            leaf = new_node()
            a, b = (anchor, leaf) if decoy_index % 2 == 0 else (leaf, anchor)
        elif variant == "branch":
            # Build short dead-end branches of one to three edges.  A branch
            # touches one true node and never reconnects to the true chain.
            if branch_state is None or branch_state[2] == 0:
                anchor = chain_nodes[(decoy_index // 3) % len(chain_nodes)]
                leaf = new_node()
                remaining = 1 + rng.randrange(3)
                branch_state = (anchor, leaf, remaining)
            a, b, remaining = branch_state
            branch_state = (b, new_node(), remaining - 1) if remaining > 1 else None
        else:
            raise ValueError(variant)
        record = Record(
            uid=f"{eid}:r{position}",
            position=position,
            kind="edge",
            key=a,
            value=b,
            refs=(a, b),
        )
        slots[position] = record
        decoy_ids.add(record.uid)

    for position in range(length):
        if slots[position] is None:
            slots[position] = _distractor(eid, position, rng)
    episode = Episode(
        eid=eid,
        task="relation_chain",
        records=tuple(slots),  # type: ignore[arg-type]
        clauses=(tuple(required_ids),),
        goal_tokens=frozenset((chain_nodes[0], chain_nodes[-1])),
        target_key=chain_nodes[0],
    )
    return episode, frozenset(decoy_ids)


def targeted_groups(
    episode: Episode,
    decoy_ids: frozenset[str],
    *,
    split: str,
    capacity: int,
    family: str,
    maximum: int = 12,
) -> list[dict]:
    renderer = B21Renderer(family, f"entity_{episode.eid}")
    memory: list[Record] = []
    selected: list[dict] = []
    decision = 0
    for candidate in episode.records:
        competing = [*memory, candidate]
        if len(competing) <= capacity:
            memory = competing
            continue
        decision += 1
        group = make_decision_group(
            episode,
            competing,
            candidate,
            split=split,
            capacity=capacity,
            hard_negative_count=len(decoy_ids),
            template_family=family,
            behavior_policy="oracle",
            decision_id=decision,
            renderer=renderer,
        )
        present = {record.uid for record in competing}
        has_required = bool(present & episode.required_ids)
        has_decoy = bool(present & decoy_ids)
        has_positive = any(float(value) > 0 for value in group["utilities"])
        has_safe = any(float(value) == min(group["utilities"]) for value in group["utilities"])
        if has_required and has_decoy and has_positive and has_safe:
            group["hard_relation_group"] = True
            group["decoy_count"] = len(decoy_ids)
            selected.append(group)

        # Oracle behavior keeps every already observed required edge whenever
        # a zero-utility record is available.
        evict = min(
            range(len(competing)),
            key=lambda i: (float(group["utilities"][i]), competing[i].position),
        )
        memory = [record for i, record in enumerate(competing) if i != evict]

    if len(selected) <= maximum:
        return selected
    if maximum == 1:
        return [selected[len(selected) // 2]]
    indices = [round(i * (len(selected) - 1) / (maximum - 1)) for i in range(maximum)]
    return [selected[index] for index in indices]


def build_hard_groups(target: int, split: str, seed: int) -> list[dict]:
    result: list[dict] = []
    episode_index = 0
    while len(result) < target:
        length = LENGTHS[episode_index % len(LENGTHS)]
        capacity = CAPACITIES[(episode_index // len(LENGTHS)) % len(CAPACITIES)]
        hops = HOPS[(episode_index // (len(LENGTHS) * len(CAPACITIES))) % len(HOPS)]
        # Cycle pressure-inducing K values; combinations that fit entirely in
        # memory are skipped so every episode requires edge-vs-edge selection.
        decoy_count = DECOYS[(episode_index // 7) % len(DECOYS)]
        if hops + decoy_count <= capacity:
            decoy_count = next(value for value in DECOYS if hops + value > capacity)
        variant = VARIANTS[(episode_index // 11) % len(VARIANTS)]
        family_count = len(ROBUST_TRAIN_EDGE_TEMPLATES) if split == "train" else len(ROBUST_VALIDATION_EDGE_TEMPLATES)
        family_prefix = "RT" if split == "train" else "RV"
        family = f"{family_prefix}{episode_index % family_count}"
        episode, decoys = make_hard_episode(
            episode_index + (0 if split == "train" else 100_000),
            length=length,
            hops=hops,
            decoy_count=decoy_count,
            variant=variant,
            seed=seed * 10_000_019 + episode_index,
        )
        groups = targeted_groups(
            episode,
            decoys,
            split=split,
            capacity=capacity,
            family=family,
        )
        for group in groups:
            errors = validate_group(group)
            if errors:
                raise ValueError(f"invalid hard group {episode.eid}: {errors}")
        result.extend(groups)
        episode_index += 1
        if episode_index > target * 2 and len(result) < target:
            raise RuntimeError("unable to generate enough targeted hard groups")
    return result[:target]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_line(row) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/b2_2_robust"))
    parser.add_argument("--output", type=Path, default=Path("data/b2_4_relation_disambiguation"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        if not args.force:
            raise FileExistsError(args.output)
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    split_targets = {"train": TRAIN_GROUPS, "validation": VALIDATION_GROUPS}
    manifest = {
        "schema_version": 1,
        "preset": "b2_4_relation_disambiguation",
        "seed": args.seed,
        "source": str(args.source),
        "hard_relation": {
            "lengths": list(LENGTHS),
            "capacities": list(CAPACITIES),
            "hops": list(HOPS),
            "decoys": list(DECOYS),
            "variants": list(VARIANTS),
        },
        "splits": {},
    }
    for split, target in split_targets.items():
        original = read_jsonl(args.source / f"{split}.jsonl")
        hard = build_hard_groups(target, split, args.seed + (0 if split == "train" else 1))
        rows = [*original, *hard]
        random.Random(args.seed + (17 if split == "train" else 23)).shuffle(rows)
        path = args.output / f"{split}.jsonl"
        write_jsonl(path, rows)
        manifest["splits"][split] = {
            "file": path.name,
            "original_groups": len(original),
            "hard_groups": len(hard),
            "groups": len(rows),
            "sha256": sha256(path),
        }

    for split in ("test_id", "test_composition", "test_length", "test_semantic_stress"):
        target = Path("..") / args.source.name / f"{split}.jsonl"
        (args.output / f"{split}.jsonl").symlink_to(target)
        manifest["splits"][split] = {
            "file": f"{split}.jsonl",
            "source": str(args.source / f"{split}.jsonl"),
            "sha256": sha256(args.source / f"{split}.jsonl"),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
