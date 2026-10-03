#!/usr/bin/env python3
"""Build Phase 0B2.1 full-competition-set datasets.

Every JSONL row is one causal eviction decision.  Future records are used only
to compute the counterfactual utility vector and are never serialized as model
inputs.  Dataset files are immutable inputs shared by all six Stage-1 runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.phase0_ab import TASKS, Episode, Record, counterfactual_utilities, make_episodes
from experiments.phase0_b2_text import (
    PARAPHRASE_GOALS,
    PARAPHRASE_TEMPLATES,
    TRAIN_GOALS,
    TRAIN_TEMPLATES,
    harden_episode,
)


SCHEMA_VERSION = 1
CAPACITIES = (4, 8, 12)
HARD_NEGATIVES = (0, 4, 8, 14)
BEHAVIOR_POLICIES = ("fifo", "random", "oracle")
FORBIDDEN_INPUT_FIELDS = {
    "important",
    "consumed",
    "unresolved",
    "future_use",
    "required",
    "final_answer",
    "utility",
    "utilities",
    "oracle",
}


COMPOSITION_TEMPLATES: dict[str, str] = {
    "fact": "事实：{key} 对应的信息为 {value}。",
    "state": "当前 {key} 的有效值为 {value}。",
    "instruction": "交付时，{key} 必须使用 {value}。",
    "intermediate_done": "中间结果 {key}={value} 已经使用完毕，可以清理。",
    "intermediate_open": "请勿丢弃中间结果 {key}={value}，后面的步骤仍然需要。",
    "subgoal_open": "待办事项 {key} 仍未闭环，目前是 {value}。",
    "subgoal_done": "事项 {key} 已经闭环，结果为 {value}。",
    "edge": "关系：实体 {a} 与下一节点 {b} 相连。",
    "log": "运行日志提到 {key}，内容为 {value}。",
}

COMPOSITION_GOALS: dict[str, str] = {
    "delayed_query": "后续查询需要与 {tokens} 有关的信息。",
    "state_overwrite": "任务结束时，只采用 {tokens} 的最新有效值。",
    "long_instruction": "交付答案时必须遵守这些要求：{tokens}。",
    "completed_intermediate": "继续后面步骤仍然需要的计算：{tokens}。",
    "unresolved_subgoal": "推进尚未闭环的事项并保留证据：{tokens}。",
    "relation_chain": "找出 {tokens} 所列对象之间能否连通。",
}

HIDDEN_GOALS: dict[str, str] = {
    "A": "记住稍后查询可能需要的信息。",
    "B": "后续会提出查询，请保留可能相关的记录。",
    "C": "未来问题尚未公布，先维护有用线索。",
    "D": "稍后查询需要当前记录中的某项信息。",
    "E": "尚不知道最终会询问什么，请暂存必要线索。",
}

# Relation-only training variants.  None duplicates the frozen D/E evaluation
# sentence.  Non-relation records in the same episode still use A/B/C through
# the variant's base family, so total data volume and the other five tasks stay
# unchanged.
RELATION_AUG_EDGE_TEMPLATES: tuple[str, ...] = (
    "{a} 指向节点 {b}。",
    "从节点 {a} 可直接走到 {b}。",
    "{b} 是 {a} 的后继节点。",
    "有向边的起点为 {a}，终点为 {b}。",
    "工具 {a} 的输出交给 {b}。",
    "路径片段写作 {a} -> {b}。",
    "连接记录显示 {a} 通往 {b}。",
    "{a} 后面紧接节点 {b}。",
)

RELATION_AUG_GOAL_TEMPLATES: tuple[str, ...] = (
    "判断 {tokens} 中的两个端点是否存在路径。",
    "沿已有关系边检查 {tokens} 能否互相到达。",
    "根据连接记录推导 {tokens} 之间的通路。",
    "确认从 {tokens} 的一个端点能否走到另一个端点。",
    "组合现有边，回答关于 {tokens} 的连通问题。",
    "追踪中间节点并判断 {tokens} 是否可达。",
    "利用路径片段核对 {tokens} 的连接关系。",
    "检查关系图中 {tokens} 对应端点之间是否有链路。",
)

# B2.2 uses a compositional grammar instead of a short list of complete
# sentences.  Training and validation components are deliberately disjoint;
# frozen D/E test renderers remain unchanged.
ROBUST_TRAIN_EDGE_PREFIXES: tuple[str, ...] = (
    "",
    "记录：",
    "工具返回：",
    "当前已知：",
)
ROBUST_TRAIN_EDGE_CLAUSES: tuple[str, ...] = (
    "{a} 指向 {b}。",
    "从 {a} 可以直接到达 {b}。",
    "{b} 是 {a} 的后继节点。",
    "一条有向边从 {a} 通往 {b}。",
)
ROBUST_TRAIN_EDGE_SUFFIXES: tuple[str, ...] = (
    "",
    " 这是一条单步关系。",
    " 此连接记录有效。",
)
ROBUST_TRAIN_GOAL_PREFIXES: tuple[str, ...] = (
    "",
    "请根据已有记录，",
    "结合工具返回，",
    "沿图中的有向边，",
)
ROBUST_TRAIN_GOAL_CLAUSES: tuple[str, ...] = (
    "判断 {tokens} 两个端点是否可达。",
    "检查 {tokens} 之间是否存在完整路径。",
    "推导从 {tokens} 的一个端点能否走到另一个端点。",
    "找出连接 {tokens} 所需的中间关系。",
)

ROBUST_VALIDATION_EDGE_PREFIXES: tuple[str, ...] = (
    "补充说明：",
    "检索结果如下：",
    "来自关系图的证据表明，",
)
ROBUST_VALIDATION_EDGE_CLAUSES: tuple[str, ...] = (
    "节点 {a} 的下一跳是 {b}。",
    "若位于 {a}，下一步可前往 {b}。",
    "以 {a} 为起点的这条边终止于 {b}。",
    "{b} 接收来自 {a} 的输出。",
)
ROBUST_VALIDATION_EDGE_SUFFIXES: tuple[str, ...] = (
    "",
    " 这是局部路径证据。",
)
ROBUST_VALIDATION_GOAL_PREFIXES: tuple[str, ...] = (
    "在不遗漏中间节点的前提下，",
    "使用上述局部证据，",
    "综合所有可用连接，",
)
ROBUST_VALIDATION_GOAL_CLAUSES: tuple[str, ...] = (
    "核对 {tokens} 是否处于同一条可达链上。",
    "判断能否沿有向关系从 {tokens} 的一端抵达另一端。",
    "寻找一条把 {tokens} 两端串联起来的路径。",
)


def _compose_templates(
    prefixes: Sequence[str], clauses: Sequence[str], suffixes: Sequence[str]
) -> tuple[str, ...]:
    return tuple(
        f"{prefix}{clause}{suffix}"
        for prefix in prefixes
        for clause in clauses
        for suffix in suffixes
    )


ROBUST_TRAIN_EDGE_TEMPLATES = _compose_templates(
    ROBUST_TRAIN_EDGE_PREFIXES,
    ROBUST_TRAIN_EDGE_CLAUSES,
    ROBUST_TRAIN_EDGE_SUFFIXES,
)
ROBUST_VALIDATION_EDGE_TEMPLATES = _compose_templates(
    ROBUST_VALIDATION_EDGE_PREFIXES,
    ROBUST_VALIDATION_EDGE_CLAUSES,
    ROBUST_VALIDATION_EDGE_SUFFIXES,
)
ROBUST_TRAIN_GOAL_TEMPLATES = tuple(
    f"{prefix}{clause}"
    for prefix in ROBUST_TRAIN_GOAL_PREFIXES
    for clause in ROBUST_TRAIN_GOAL_CLAUSES
)
ROBUST_VALIDATION_GOAL_TEMPLATES = tuple(
    f"{prefix}{clause}"
    for prefix in ROBUST_VALIDATION_GOAL_PREFIXES
    for clause in ROBUST_VALIDATION_GOAL_CLAUSES
)

# These are task vocabulary, not episode-specific entities.
FIXED_TOKENS = {
    "remember_for_possible_followup",
    "output_format",
    "json",
    "partial_result",
    "open",
    "closed",
    "needed_for_final",
}


def composition_unseen_characters() -> dict[str, str]:
    """Return D-template characters absent from A/B/C literal vocabulary."""
    seen: set[str] = set()
    for table in (TRAIN_TEMPLATES, TRAIN_GOALS):
        for templates in table.values():
            for template in templates:
                seen.update(re.sub(r"\{[^}]+\}", "", template))
    for table in (PARAPHRASE_TEMPLATES, PARAPHRASE_GOALS):
        for templates in table.values():
            # Family C uses the first paraphrase; the second is reserved for E.
            seen.update(re.sub(r"\{[^}]+\}", "", templates[0]))
    ignored = set("，。：、=-> ")
    combined = {
        **COMPOSITION_TEMPLATES,
        **{f"goal:{key}": value for key, value in COMPOSITION_GOALS.items()},
    }
    result: dict[str, str] = {}
    for key, template in combined.items():
        unseen = sorted(set(re.sub(r"\{[^}]+\}", "", template)) - seen - ignored)
        if unseen:
            result[key] = "".join(unseen)
    return result


@dataclass(frozen=True)
class SplitSpec:
    name: str
    episodes: int
    lengths: tuple[int, ...]
    template_families: tuple[str, ...]


@dataclass(frozen=True)
class PreparedEpisode:
    episode: Episode
    capacity: int
    hard_negative_count: int
    template_family: str
    behavior_policy: str
    renderer: B21Renderer
    query_visibility: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dynamic_token(token: str) -> bool:
    if token in FIXED_TOKENS:
        return False
    return bool(re.search(r"\d", token) or token.startswith(("lookup_", "state_", "answer_", "pending_", "subgoal_", "node_", "decoy_")))


class B21Renderer:
    """Render one template family with an episode-unique neutral namespace."""

    def __init__(self, family: str, namespace: str):
        relation_variant: int | None = None
        robust_split: str | None = None
        if family.startswith("R") and family[1:].isdigit():
            relation_variant = int(family[1:])
            if relation_variant >= len(RELATION_AUG_EDGE_TEMPLATES):
                raise ValueError(f"unknown relation template family: {family}")
        elif family.startswith("RT") and family[2:].isdigit():
            robust_split = "train"
            relation_variant = int(family[2:])
            if relation_variant >= len(ROBUST_TRAIN_EDGE_TEMPLATES):
                raise ValueError(f"unknown robust train family: {family}")
        elif family.startswith("RV") and family[2:].isdigit():
            robust_split = "validation"
            relation_variant = int(family[2:])
            if relation_variant >= len(ROBUST_VALIDATION_EDGE_TEMPLATES):
                raise ValueError(f"unknown robust validation family: {family}")
        elif family not in {"A", "B", "C", "D", "E"}:
            raise ValueError(f"unknown template family: {family}")
        self.family = family
        self.namespace = namespace
        self.relation_variant = relation_variant
        self.robust_split = robust_split
        self.base_family = (
            ("A", "B", "C")[relation_variant % 3]
            if relation_variant is not None
            else family
        )

    def entity(self, token: str) -> str:
        return f"{self.namespace}_{token}" if _dynamic_token(token) else token

    def entity_ids(self, episode: Episode) -> set[str]:
        values: set[str] = set()
        for record in episode.records:
            for token in (record.key, record.value, *record.refs):
                if _dynamic_token(token):
                    values.add(self.entity(token))
        for token in episode.goal_tokens:
            if _dynamic_token(token):
                values.add(self.entity(token))
        return values

    def _record_key(self, record: Record) -> str:
        if record.kind == "intermediate":
            return "intermediate_done" if record.consumed else "intermediate_open"
        if record.kind == "subgoal":
            return "subgoal_open" if record.unresolved else "subgoal_done"
        return record.kind

    def _template(self, key: str) -> str:
        if key == "edge" and self.robust_split == "train":
            return ROBUST_TRAIN_EDGE_TEMPLATES[self.relation_variant or 0]
        if key == "edge" and self.robust_split == "validation":
            return ROBUST_VALIDATION_EDGE_TEMPLATES[self.relation_variant or 0]
        if key == "edge" and self.relation_variant is not None:
            return RELATION_AUG_EDGE_TEMPLATES[self.relation_variant]
        if self.base_family == "A":
            return TRAIN_TEMPLATES[key][0]
        if self.base_family == "B":
            return TRAIN_TEMPLATES[key][1]
        if self.base_family == "C":
            return PARAPHRASE_TEMPLATES[key][0]
        if self.base_family == "E":
            return PARAPHRASE_TEMPLATES[key][1]
        return COMPOSITION_TEMPLATES[key]

    def record(self, record: Record) -> str:
        template = self._template(self._record_key(record))
        if record.kind == "edge":
            a, b = record.refs
            return template.format(a=self.entity(a), b=self.entity(b))
        return template.format(key=self.entity(record.key), value=self.entity(record.value))

    def goal(self, episode: Episode) -> tuple[str, str]:
        hidden = episode.task == "delayed_query" and "remember_for_possible_followup" in episode.goal_tokens
        if hidden:
            return HIDDEN_GOALS[self.base_family], "hidden_query"
        tokens = "、".join(sorted(self.entity(token) for token in episode.goal_tokens))
        if episode.task == "relation_chain" and self.robust_split == "train":
            template = ROBUST_TRAIN_GOAL_TEMPLATES[
                (self.relation_variant or 0) % len(ROBUST_TRAIN_GOAL_TEMPLATES)
            ]
        elif episode.task == "relation_chain" and self.robust_split == "validation":
            template = ROBUST_VALIDATION_GOAL_TEMPLATES[
                (self.relation_variant or 0) % len(ROBUST_VALIDATION_GOAL_TEMPLATES)
            ]
        elif episode.task == "relation_chain" and self.relation_variant is not None:
            template = RELATION_AUG_GOAL_TEMPLATES[self.relation_variant]
        elif self.base_family == "A":
            template = TRAIN_GOALS[episode.task][0]
        elif self.base_family == "B":
            template = TRAIN_GOALS[episode.task][1]
        elif self.base_family == "C":
            template = PARAPHRASE_GOALS[episode.task][0]
        elif self.base_family == "E":
            template = PARAPHRASE_GOALS[episode.task][1]
        else:
            template = COMPOSITION_GOALS[episode.task]
        return template.format(tokens=tokens), "announced_query"


def record_source(record: Record) -> str:
    if record.kind == "instruction":
        return "user"
    if record.kind in {"intermediate", "subgoal"}:
        return "assistant"
    return "tool"


def group_model_input(group: dict[str, Any]) -> dict[str, Any]:
    """Return exactly the fields later models may consume, excluding labels."""
    return {
        "goal": group["goal"],
        "recent_context": group["recent_context"],
        "records": [
            {
                "text": item["text"],
                "event_index": item["event_index"],
                "relative_age": item["relative_age"],
                "is_candidate": item["is_candidate"],
                "source": item["source"],
            }
            for item in group["records"]
        ],
        "candidate_index": group["candidate_index"],
    }


def _choose_eviction(
    policy: str,
    competing: Sequence[Record],
    utilities: dict[str, float],
    rng: random.Random,
) -> int:
    if policy == "fifo":
        return min(range(len(competing)), key=lambda i: competing[i].position)
    if policy == "random":
        return rng.randrange(len(competing))
    if policy == "oracle":
        return min(
            range(len(competing)),
            key=lambda i: (utilities[competing[i].uid], competing[i].position),
        )
    raise ValueError(policy)


def _sample_groups(groups: Sequence[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if len(groups) <= maximum:
        return list(groups)
    if maximum <= 1:
        return [groups[len(groups) // 2]]
    indices = [round(i * (len(groups) - 1) / (maximum - 1)) for i in range(maximum)]
    return [groups[index] for index in indices]


def make_decision_group(
    episode: Episode,
    competing: Sequence[Record],
    candidate: Record,
    *,
    split: str,
    capacity: int,
    hard_negative_count: int,
    template_family: str,
    behavior_policy: str,
    decision_id: int,
    renderer: B21Renderer | None = None,
) -> dict[str, Any]:
    """Serialize one already-observed competition set plus training labels."""
    renderer = renderer or B21Renderer(template_family, f"entity_{episode.eid}")
    goal, query_visibility = renderer.goal(episode)
    utilities_by_uid = counterfactual_utilities(episode, competing, candidate.position)
    utilities = [float(utilities_by_uid[item.uid]) for item in competing]
    minimum = min(utilities)
    oracle_indices = [i for i, value in enumerate(utilities) if math.isclose(value, minimum)]
    recent_records = [item for item in episode.records if item.position < candidate.position][-2:]
    records = [
        {
            "uid": item.uid,
            "text": renderer.record(item),
            "event_index": item.position,
            "relative_age": candidate.position - item.position,
            "is_candidate": item.uid == candidate.uid,
            "source": record_source(item),
        }
        for item in competing
    ]
    group: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "episode_id": episode.eid,
        "decision_id": decision_id,
        "decision_position": candidate.position,
        "task": episode.task,
        "query_visibility": query_visibility,
        "template_family": template_family,
        "capacity": capacity,
        "hard_negative_count": hard_negative_count,
        "behavior_policy": behavior_policy,
        "goal": goal,
        "recent_context": [
            {"text": renderer.record(item), "event_index": item.position}
            for item in recent_records
        ],
        "records": records,
        "candidate_index": len(records) - 1,
        "utilities": utilities,
        "oracle_eviction_indices": oracle_indices,
    }
    group["input_sha256"] = _sha256_bytes(_json_line(group_model_input(group)).encode("utf-8"))
    return group


def collect_episode_groups(
    episode: Episode,
    *,
    split: str,
    capacity: int,
    hard_negative_count: int,
    template_family: str,
    behavior_policy: str,
    seed: int,
    max_groups: int = 12,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Roll out one behavior policy and return sampled complete decisions."""
    episode = harden_episode(episode, seed + 17, count=hard_negative_count)
    renderer = B21Renderer(template_family, f"entity_{episode.eid}")
    rng = random.Random(seed)
    memory: list[Record] = []
    all_groups: list[dict[str, Any]] = []
    decision = 0

    for candidate in episode.records:
        competing = memory + [candidate]
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
            hard_negative_count=hard_negative_count,
            template_family=template_family,
            behavior_policy=behavior_policy,
            decision_id=decision,
            renderer=renderer,
        )
        all_groups.append(group)

        utilities_by_uid = {
            item.uid: float(group["utilities"][i]) for i, item in enumerate(competing)
        }
        eviction = _choose_eviction(behavior_policy, competing, utilities_by_uid, rng)
        memory = [item for i, item in enumerate(competing) if i != eviction]

    return _sample_groups(all_groups, max_groups), renderer.entity_ids(episode)


def validate_group(group: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    records = group["records"]
    if len(records) != int(group["capacity"]) + 1:
        errors.append("record count is not capacity + 1")
    if len(group["utilities"]) != len(records):
        errors.append("utility vector length mismatch")
    if not (0 <= int(group["candidate_index"]) < len(records)):
        errors.append("candidate index out of range")
    elif not records[int(group["candidate_index"])]["is_candidate"]:
        errors.append("candidate index does not point to candidate")
    if sum(bool(item["is_candidate"]) for item in records) != 1:
        errors.append("decision must contain exactly one candidate")
    if any(int(item["event_index"]) > int(group["decision_position"]) for item in records):
        errors.append("future record leaked into competition set")
    if any(
        int(item["event_index"]) >= int(group["decision_position"])
        for item in group["recent_context"]
    ):
        errors.append("future/current record leaked into recent context")
    if not group["oracle_eviction_indices"]:
        errors.append("oracle eviction set is empty")
    minimum = min(float(value) for value in group["utilities"])
    expected = {
        i for i, value in enumerate(group["utilities"]) if math.isclose(float(value), minimum)
    }
    if set(group["oracle_eviction_indices"]) != expected:
        errors.append("oracle eviction indices do not match minimum utility")
    input_view = group_model_input(group)
    if group["input_sha256"] != _sha256_bytes(_json_line(input_view).encode("utf-8")):
        errors.append("input hash mismatch")
    serialized_input = _json_line(input_view)
    for field in FORBIDDEN_INPUT_FIELDS:
        if f'"{field}"' in serialized_input:
            errors.append(f"forbidden model-input field: {field}")
    return errors


def _split_specs(preset: str) -> tuple[SplitSpec, ...]:
    base_preset = preset.removesuffix("_relation_aug").removesuffix("_relation_robust")
    if base_preset == "audit":
        train, validation, test = 100, 24, 24
    elif base_preset == "stage1":
        train, validation, test = 3000, 400, 600
    else:
        raise ValueError(preset)
    return (
        SplitSpec("train", train, (32, 48, 64), ("A", "B", "C")),
        SplitSpec("validation", validation, (32, 48, 64), ("A", "B", "C")),
        SplitSpec("test_id", test, (48, 64), ("A", "B", "C")),
        SplitSpec("test_composition", test, (48, 64), ("D",)),
        SplitSpec("test_length", test, (96, 128), ("D",)),
        SplitSpec("test_semantic_stress", test, (48, 64), ("E",)),
    )


def _assignment(
    index: int,
    spec: SplitSpec,
    *,
    relation_augmented: bool = False,
    relation_robust: bool = False,
) -> tuple[int, int, str, str]:
    task_index = index % len(TASKS)
    occurrence = index // len(TASKS)
    capacity = CAPACITIES[(task_index + occurrence) % len(CAPACITIES)]
    hard_negative_count = HARD_NEGATIVES[(task_index + 2 * occurrence) % len(HARD_NEGATIVES)]
    family = spec.template_families[(task_index + occurrence) % len(spec.template_families)]
    if relation_robust and spec.name in {"train", "validation"} and TASKS[task_index] == "relation_chain":
        if spec.name == "train":
            family = f"RT{occurrence % len(ROBUST_TRAIN_EDGE_TEMPLATES)}"
        else:
            family = f"RV{occurrence % len(ROBUST_VALIDATION_EDGE_TEMPLATES)}"
    elif relation_augmented and spec.name in {"train", "validation"} and TASKS[task_index] == "relation_chain":
        family = f"R{occurrence % len(RELATION_AUG_EDGE_TEMPLATES)}"
    behavior = BEHAVIOR_POLICIES[(task_index + occurrence) % len(BEHAVIOR_POLICIES)]
    return capacity, hard_negative_count, family, behavior


def materialize_split_episodes(preset: str, seed: int, split_name: str) -> list[PreparedEpisode]:
    """Recreate exactly the episodes used by a serialized dataset split."""
    specs = _split_specs(preset)
    matches = [(index, spec) for index, spec in enumerate(specs) if spec.name == split_name]
    if not matches:
        raise KeyError(split_name)
    split_index, spec = matches[0]
    episodes = make_episodes(
        spec.episodes,
        spec.lengths,
        seed=seed + 10_000 * (split_index + 1),
        start_index=(split_index + 1) * 100_000_000,
    )
    prepared: list[PreparedEpisode] = []
    for i, episode in enumerate(episodes):
        capacity, hard_count, family, behavior = _assignment(
            i,
            spec,
            relation_augmented=preset.endswith("_relation_aug"),
            relation_robust=preset.endswith("_relation_robust"),
        )
        episode_seed = seed * 1_000_003 + split_index * 100_003 + i
        hardened = harden_episode(episode, episode_seed + 17, count=hard_count)
        renderer = B21Renderer(family, f"entity_{hardened.eid}")
        _, visibility = renderer.goal(hardened)
        prepared.append(
            PreparedEpisode(
                episode=hardened,
                capacity=capacity,
                hard_negative_count=hard_count,
                template_family=family,
                behavior_policy=behavior,
                renderer=renderer,
                query_visibility=visibility,
            )
        )
    return prepared


def _write_jsonl(path: Path, groups: Iterable[dict[str, Any]]) -> int:
    count = 0
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(_json_line(group) + "\n")
            count += 1
    temporary.replace(path)
    return count


def _hash_strings(values: Iterable[str]) -> str:
    return _sha256_bytes("\n".join(sorted(values)).encode("utf-8"))


def build_dataset(
    output: Path,
    preset: str,
    seed: int,
    force: bool = False,
    specs: Sequence[SplitSpec] | None = None,
) -> dict[str, Any]:
    unseen = composition_unseen_characters()
    if unseen:
        raise ValueError(f"composition template D introduces unseen characters: {unseen}")
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(f"output directory is not empty: {output}; pass --force")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator": "experiments/phase0_b21_data.py",
        "generator_sha256": _sha256_file(Path(__file__)),
        "preset": preset,
        "seed": seed,
        "max_decision_groups_per_episode": 12,
        "capacities": list(CAPACITIES),
        "hard_negative_counts": list(HARD_NEGATIVES),
        "tasks": list(TASKS),
        "training_template_families": (
            ["A", "B", "C", *[f"RT{i}" for i in range(len(ROBUST_TRAIN_EDGE_TEMPLATES))]]
            if preset.endswith("_relation_robust")
            else (
                ["A", "B", "C", *[f"R{i}" for i in range(len(RELATION_AUG_EDGE_TEMPLATES))]]
                if preset.endswith("_relation_aug")
                else ["A", "B", "C"]
            )
        ),
        "composition_template_family": "D",
        "composition_unseen_characters": unseen,
        "semantic_stress_template_family": "E",
        "splits": {},
    }
    all_entities: dict[str, set[str]] = {}
    all_episode_ids: dict[str, set[str]] = {}

    for split_index, spec in enumerate(specs or _split_specs(preset)):
        episodes = make_episodes(
            spec.episodes,
            spec.lengths,
            seed=seed + 10_000 * (split_index + 1),
            start_index=(split_index + 1) * 100_000_000,
        )
        groups: list[dict[str, Any]] = []
        entities: set[str] = set()
        task_counts: Counter[str] = Counter()
        capacity_counts: Counter[str] = Counter()
        hard_negative_counts: Counter[str] = Counter()
        template_counts: Counter[str] = Counter()
        visibility_counts: Counter[str] = Counter()
        behavior_counts: Counter[str] = Counter()
        informative_decisions = 0
        positive_records = 0
        total_records = 0
        oracle_tie_total = 0

        for i, episode in enumerate(episodes):
            capacity, hard_count, family, behavior = _assignment(
                i,
                spec,
                relation_augmented=preset.endswith("_relation_aug"),
                relation_robust=preset.endswith("_relation_robust"),
            )
            episode_groups, episode_entities = collect_episode_groups(
                episode,
                split=spec.name,
                capacity=capacity,
                hard_negative_count=hard_count,
                template_family=family,
                behavior_policy=behavior,
                seed=seed * 1_000_003 + split_index * 100_003 + i,
            )
            for group in episode_groups:
                errors = validate_group(group)
                if errors:
                    raise ValueError(f"invalid group {spec.name}/{episode.eid}: {errors}")
                informative_decisions += int(any(float(value) > 0 for value in group["utilities"]))
                positive_records += sum(float(value) > 0 for value in group["utilities"])
                total_records += len(group["records"])
                oracle_tie_total += len(group["oracle_eviction_indices"])
            groups.extend(episode_groups)
            entities.update(episode_entities)
            task_counts[episode.task] += 1
            capacity_counts[str(capacity)] += 1
            hard_negative_counts[str(hard_count)] += 1
            template_counts[family] += 1
            behavior_counts[behavior] += 1
            if episode_groups:
                visibility_counts[episode_groups[0]["query_visibility"]] += 1

        path = output / f"{spec.name}.jsonl"
        decision_count = _write_jsonl(path, groups)
        episode_ids = {episode.eid for episode in episodes}
        all_entities[spec.name] = entities
        all_episode_ids[spec.name] = episode_ids
        manifest["splits"][spec.name] = {
            "file": path.name,
            "sha256": _sha256_file(path),
            "episodes": len(episodes),
            "decisions": decision_count,
            "lengths": list(spec.lengths),
            "task_counts": dict(sorted(task_counts.items())),
            "capacity_counts": dict(sorted(capacity_counts.items())),
            "hard_negative_counts": dict(sorted(hard_negative_counts.items())),
            "template_counts": dict(sorted(template_counts.items())),
            "query_visibility_counts": dict(sorted(visibility_counts.items())),
            "behavior_policy_counts": dict(sorted(behavior_counts.items())),
            "informative_decisions": informative_decisions,
            "informative_decision_rate": informative_decisions / max(1, decision_count),
            "positive_records": positive_records,
            "total_records": total_records,
            "positive_record_rate": positive_records / max(1, total_records),
            "mean_oracle_tie_count": oracle_tie_total / max(1, decision_count),
            "episode_ids_sha256": _hash_strings(episode_ids),
            "entity_count": len(entities),
            "entity_ids_sha256": _hash_strings(entities),
        }

    split_names = list(all_episode_ids)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            if all_episode_ids[left] & all_episode_ids[right]:
                raise ValueError(f"episode leakage between {left} and {right}")
            if all_entities[left] & all_entities[right]:
                overlap = sorted(all_entities[left] & all_entities[right])[:5]
                raise ValueError(f"entity leakage between {left} and {right}: {overlap}")

    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_audit_report(output, manifest)
    return manifest


def write_audit_report(output: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Phase 0B2.1 数据审计",
        "",
        f"- preset：`{manifest['preset']}`",
        f"- seed：`{manifest['seed']}`",
        f"- schema：`{manifest['schema_version']}`",
        "- episode 与动态实体已经按 split 隔离；",
        "- 每个决策组已经执行因果边界、候选数量、标签和输入字段检查。",
        "",
        "| split | episode | 决策组 | 有信息决策 | 正效用记录 | 模板 | 查询可见性 | SHA-256 |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for name, item in manifest["splits"].items():
        templates = ", ".join(f"{key}:{value}" for key, value in item["template_counts"].items())
        visibility = ", ".join(
            f"{key}:{value}" for key, value in item["query_visibility_counts"].items()
        )
        lines.append(
            f"| {name} | {item['episodes']} | {item['decisions']} | "
            f"{item['informative_decision_rate']:.1%} | {item['positive_record_rate']:.1%} | {templates} | "
            f"{visibility} | `{item['sha256'][:12]}` |"
        )
    lines.append("")
    (output / "AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/b2_1_stage1_audit"))
    parser.add_argument(
        "--preset",
        choices=(
            "audit",
            "stage1",
            "audit_relation_aug",
            "stage1_relation_aug",
            "audit_relation_robust",
            "stage1_relation_robust",
        ),
        default="audit",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = build_dataset(args.output, args.preset, args.seed, force=args.force)
    print(json.dumps(manifest["splits"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
