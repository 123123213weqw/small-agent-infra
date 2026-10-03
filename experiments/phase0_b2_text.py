#!/usr/bin/env python3
"""Phase 0B2: learn counterfactual utility from text, not structure flags.

The model is a small byte-level Transformer trained from scratch. Its input is
limited to information available at the eviction decision: the task goal, one
candidate record, recent observed events, and a bounded view of competing
memory. Future records are used only to construct training labels and the
Future Oracle baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.phase0_ab import (
    TASKS,
    Episode,
    Record,
    RunMetrics,
    UtilityPredictor,
    build_training_set,
    counterfactual_utilities,
    make_episodes,
    run_episode,
)
from experiments.experiment_visualization import (
    create_summary_writer,
    log_evaluation_to_tensorboard,
    write_eviction_artifacts,
)


TRAIN_TEMPLATES: dict[str, tuple[str, ...]] = {
    "fact": ("事实：{key} 等于 {value}。", "观察结果：{key} = {value}。"),
    "state": ("状态更新：{key} 现在是 {value}。", "当前 {key} 已变成 {value}。"),
    "instruction": ("要求：{key} 必须使用 {value}。", "任务约束：{key} 设为 {value}。"),
    "intermediate_done": ("中间结果 {key}={value} 已经使用完毕。", "临时计算 {key}={value} 已消费。"),
    "intermediate_open": ("中间结果 {key}={value} 后续仍然需要。", "临时计算 {key}={value} 尚未使用。"),
    "subgoal_open": ("子目标 {key} 仍未解决，状态为 {value}。", "待办事项 {key} 仍处于 {value}。"),
    "subgoal_done": ("子目标 {key} 已完成，结果为 {value}。", "事项 {key} 已处理：{value}。"),
    "edge": ("关系：{a} 指向 {b}。", "已知连接 {a} -> {b}。"),
    "log": ("运行日志：{key}，内容 {value}。", "调试记录 {key}: {value}。"),
}

PARAPHRASE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "fact": ("可以确认，{value} 是 {key} 对应的信息。", "条目 {key} 所记录的内容为 {value}。"),
    "state": ("把变量 {key} 的有效值改写为 {value}。", "对于 {key}，最新读数为 {value}。"),
    "instruction": ("交付时务必令 {key} 符合 {value}。", "最终产物需要遵循 {key}:{value}。"),
    "intermediate_done": ("{key} 得到 {value}，且已服务于下一步骤，可以清理。", "{key} 的 {value} 已经被后续过程采用。"),
    "intermediate_open": ("请勿丢弃 {key} 的 {value}，后面的步骤还会引用它。", "{key} 产生 {value}，目前尚未投入使用。"),
    "subgoal_open": ("关于 {key} 的工作仍在等待处理，目前标记 {value}。", "{key} 还没有闭环，其记录是 {value}。"),
    "subgoal_done": ("{key} 已经闭环，留下结果 {value}。", "无需继续处理 {key}，结果为 {value}。"),
    "edge": ("实体 {a} 与下一节点 {b} 相连。", "从 {a} 可以到达 {b}。"),
    "log": ("诊断输出提到 {key}，附带值 {value}。", "一条普通追踪信息：{key}/{value}。"),
}

TRAIN_GOALS: dict[str, tuple[str, ...]] = {
    "delayed_query": ("记住稍后查询所需的信息：{tokens}。", "后续需要回答与 {tokens} 有关的问题。"),
    "state_overwrite": ("跟踪 {tokens}，最终只采用最新有效状态。", "任务结束时报告 {tokens} 的当前值。"),
    "long_instruction": ("完成任务并遵守这些要求：{tokens}。", "保留答案以及交付约束：{tokens}。"),
    "completed_intermediate": ("继续尚未结束的计算：{tokens}。", "不要丢失仍待使用的结果：{tokens}。"),
    "unresolved_subgoal": ("完成目标及其必要证据：{tokens}。", "推进未完成事项，关注 {tokens}。"),
    "relation_chain": ("找出这些实体之间的关系：{tokens}。", "回答关于连接路径的问题：{tokens}。"),
}

PARAPHRASE_GOALS: dict[str, tuple[str, ...]] = {
    "delayed_query": ("未来的追问可能涉及 {tokens}，请保留相关线索。", "稍后核对 {tokens} 时需要准确作答。"),
    "state_overwrite": ("在所有改动结束后，给出 {tokens} 的最终版本。", "只认最后一次写入，并返回 {tokens}。"),
    "long_instruction": ("交付结果时同时满足 {tokens} 所描述的规范。", "产出内容及格式要求都与 {tokens} 有关。"),
    "completed_intermediate": ("后续步骤仍会引用 {tokens} 对应的尚未消费结果。", "请延续仍未收尾的计算 {tokens}。"),
    "unresolved_subgoal": ("解决 {tokens} 涉及的待办并结合所需依据。", "最终结论依赖尚未闭环的 {tokens}。"),
    "relation_chain": ("推导 {tokens} 所列对象之间能否连通。", "沿已有边寻找 {tokens} 之间的链路。"),
}


def _stable_choice(options: Sequence[str], key: str) -> str:
    return options[sum(key.encode("utf-8")) % len(options)]


class TextRenderer:
    """Turn structured simulator state into text without exposing flags."""

    def __init__(self, family: str):
        if family not in {"train", "paraphrase"}:
            raise ValueError(family)
        self.family = family
        self.record_templates = TRAIN_TEMPLATES if family == "train" else PARAPHRASE_TEMPLATES
        self.goal_templates = TRAIN_GOALS if family == "train" else PARAPHRASE_GOALS

    def goal(self, episode: Episode) -> str:
        tokens = "、".join(sorted(episode.goal_tokens))
        template = _stable_choice(self.goal_templates[episode.task], episode.eid)
        return template.format(tokens=tokens)

    def record(self, record: Record) -> str:
        if record.kind == "intermediate":
            key = "intermediate_done" if record.consumed else "intermediate_open"
        elif record.kind == "subgoal":
            key = "subgoal_open" if record.unresolved else "subgoal_done"
        else:
            key = record.kind
        template = _stable_choice(self.record_templates[key], record.uid)
        if record.kind == "edge":
            a, b = record.refs
            return template.format(a=a, b=b)
        return template.format(key=record.key, value=record.value)

    def model_input(
        self,
        episode: Episode,
        candidate: Record,
        competing: Sequence[Record],
        decision_pos: int,
    ) -> str:
        # All referenced records have already arrived. The bounded view makes
        # input cost independent of total episode length.
        recent = [r for r in episode.records if r.position < decision_pos][-2:]
        others = [r for r in competing if r.uid != candidate.uid]
        # Include records nearest the candidate plus oldest records; both are
        # causal and help expose overwrite and relation interactions.
        selected: list[Record] = []
        for rec in (others[:2] + others[-2:]):
            if rec.uid not in {x.uid for x in selected}:
                selected.append(rec)
        parts = [
            "[目标] " + self.goal(episode),
            "[候选记录] " + self.record(candidate),
            "[近期上下文] " + " ".join(self.record(r) for r in recent),
            "[其他竞争记录] " + " ".join(self.record(r) for r in selected),
        ]
        return "\n".join(parts)


@dataclass(frozen=True)
class TextExample:
    text: str
    utility: float


def harden_episode(episode: Episode, seed: int, count: int = 14) -> Episode:
    """Add goal-overlapping but semantically disposable hard negatives.

    Exact identifier matching should not solve B2. Required records are never
    changed; only irrelevant records are rewritten into plausible competitors.
    """
    rng = random.Random(seed)
    required = episode.required_ids
    available = [r.position for r in episode.records if r.uid not in required]
    rng.shuffle(available)
    positions = available[: min(count, len(available))]
    records = list(episode.records)
    goal_tokens = sorted(episode.goal_tokens)
    target = episode.target_key or (goal_tokens[0] if goal_tokens else "task_item")
    for j, pos in enumerate(positions):
        old = records[pos]
        if episode.task == "relation_chain":
            endpoints = [x for x in goal_tokens if x.startswith("node_")]
            if len(endpoints) >= 2:
                decoy = f"decoy_{episode.eid}_{j}"
                a, b = (endpoints[0], decoy) if j % 2 == 0 else (decoy, endpoints[-1])
                records[pos] = Record(old.uid, pos, "edge", a, b, refs=(a, b))
                continue
        if episode.task == "unresolved_subgoal" and j % 2 == 0:
            records[pos] = Record(old.uid, pos, "subgoal", target, "closed", unresolved=False)
        else:
            key = goal_tokens[j % len(goal_tokens)] if goal_tokens else target
            records[pos] = Record(
                old.uid,
                pos,
                "intermediate",
                key,
                f"obsolete_{j}",
                consumed=True,
                unresolved=False,
            )
    return Episode(
        eid=episode.eid,
        task=episode.task,
        records=tuple(records),
        clauses=episode.clauses,
        goal_tokens=episode.goal_tokens,
        target_key=episode.target_key,
    )


def harden_episodes(episodes: Sequence[Episode], seed: int) -> list[Episode]:
    return [harden_episode(ep, seed * 1_000_003 + i) for i, ep in enumerate(episodes)]


def collect_text_examples(
    episodes: Sequence[Episode],
    capacities: Sequence[int],
    renderer: TextRenderer,
    seed: int,
    sample_rate: float,
) -> list[TextExample]:
    examples: list[TextExample] = []
    collectors = ("fifo", "random", "oracle")
    for i, episode in enumerate(episodes):
        rng = random.Random(seed * 1_000_003 + i)
        capacity = capacities[i % len(capacities)]
        policy = collectors[i % len(collectors)]
        memory: list[Record] = []
        for rec in episode.records:
            competing = memory + [rec]
            if len(competing) <= capacity:
                memory = competing
                continue
            utilities = counterfactual_utilities(episode, competing, rec.position)
            if rng.random() < sample_rate:
                positive = [r for r in competing if utilities[r.uid] > 0]
                negative = [r for r in competing if utilities[r.uid] <= 0]
                rng.shuffle(negative)
                for item in positive + negative[: max(3, len(positive))]:
                    examples.append(
                        TextExample(
                            renderer.model_input(episode, item, competing, rec.position),
                            utilities[item.uid],
                        )
                    )
            if policy == "fifo":
                idx = min(range(len(competing)), key=lambda j: competing[j].position)
            elif policy == "random":
                idx = rng.randrange(len(competing))
            else:
                idx = min(range(len(competing)), key=lambda j: (utilities[competing[j].uid], competing[j].position))
            memory = [r for j, r in enumerate(competing) if j != idx]
    return examples


PAD, CLS = 256, 257
VOCAB_SIZE = 258


def encode_bytes(text: str, max_bytes: int) -> tuple[torch.Tensor, torch.Tensor]:
    raw = list(text.encode("utf-8"))[: max_bytes - 1]
    ids = [CLS] + raw
    mask = [1] * len(ids)
    padding = max_bytes - len(ids)
    ids.extend([PAD] * padding)
    mask.extend([0] * padding)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


class TextDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, examples: Sequence[TextExample], max_bytes: int):
        self.examples = examples
        self.max_bytes = max_bytes

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.examples[index]
        ids, mask = encode_bytes(item.text, self.max_bytes)
        return ids, mask, torch.tensor(item.utility, dtype=torch.float32)


class ByteUtilityTransformer(nn.Module):
    def __init__(
        self,
        max_bytes: int,
        d_model: int,
        layers: int,
        heads: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.max_bytes = max_bytes
        self.token = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.position = nn.Embedding(max_bytes, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(d_model))
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        x = self.token(ids) + self.position(pos)
        x = self.encoder(x, src_key_padding_mask=~mask)
        return torch.sigmoid(self.head(x[:, 0]).squeeze(-1))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def validation_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for ids, mask, targets in loader:
        ids, mask, targets = ids.to(device), mask.to(device), targets.to(device)
        pred = model(ids, mask)
        total += torch.square(pred - targets).sum().item()
        count += len(targets)
    return total / max(1, count)


def train_model(
    model: ByteUtilityTransformer,
    train_examples: Sequence[TextExample],
    val_examples: Sequence[TextExample],
    args: argparse.Namespace,
    device: torch.device,
    writer: object | None = None,
) -> list[dict[str, float]]:
    train_loader = DataLoader(
        TextDataset(train_examples, args.max_bytes),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TextDataset(val_examples, args.max_bytes),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    positive = sum(x.utility > 0 for x in train_examples)
    negative = len(train_examples) - positive
    positive_weight = min(8.0, negative / max(1, positive))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float]] = []
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(args.epochs):
        model.train()
        total, count = 0.0, 0
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        for step, (ids, mask, targets) in enumerate(train_loader):
            ids, mask, targets = ids.to(device), mask.to(device), targets.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                pred = model(ids, mask)
                weights = torch.where(targets > 0, positive_weight, 1.0)
                loss = (weights * torch.square(pred - targets)).mean()
            scaler.scale(loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total += loss.item() * len(targets)
            count += len(targets)
        val = validation_loss(model, val_loader, device)
        row = {
            "epoch": float(epoch + 1),
            "train_weighted_mse": total / max(1, count),
            "val_mse": val,
            "seconds": time.time() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if writer is not None:
            writer.add_scalar("training/train_weighted_mse", row["train_weighted_mse"], epoch + 1)
            writer.add_scalar("training/validation_mse", row["val_mse"], epoch + 1)
            writer.add_scalar("training/epoch_seconds", row["seconds"], epoch + 1)
            writer.add_scalar("training/learning_rate", optimizer.param_groups[0]["lr"], epoch + 1)
            writer.flush()
        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


class TextScorer:
    def __init__(self, model: ByteUtilityTransformer, renderer: TextRenderer, max_bytes: int, device: torch.device):
        self.model, self.renderer, self.max_bytes, self.device = model, renderer, max_bytes, device

    @torch.no_grad()
    def scores(self, episode: Episode, records: Sequence[Record], decision_pos: int) -> list[float]:
        encoded = [
            encode_bytes(self.renderer.model_input(episode, rec, records, decision_pos), self.max_bytes)
            for rec in records
        ]
        ids = torch.stack([x[0] for x in encoded]).to(self.device)
        mask = torch.stack([x[1] for x in encoded]).to(self.device)
        self.model.eval()
        return self.model(ids, mask).float().cpu().tolist()


def lexical_scores(renderer: TextRenderer, episode: Episode, records: Sequence[Record], decision_pos: int) -> list[float]:
    # Deliberately strong but non-semantic baseline: exact ASCII identifier
    # overlap. Hard negatives share these identifiers, forcing semantic rerank.
    goal = set(re.findall(r"[a-z0-9_]+", renderer.goal(episode).lower()))
    scores = []
    for rec in records:
        tokens = set(re.findall(r"[a-z0-9_]+", renderer.record(rec).lower()))
        scores.append(len(goal & tokens) / max(1, len(goal | tokens)))
    return scores


def run_scored_episode(
    episode: Episode,
    capacity: int,
    score_fn: Callable[[Episode, Sequence[Record], int], Sequence[float]],
    *,
    renderer: TextRenderer | None = None,
    split: str = "",
    policy: str = "text",
    trace: list[dict[str, object]] | None = None,
) -> RunMetrics:
    memory: list[Record] = []
    correct, evictions, regret = 0, 0, 0.0
    episode_traces: list[dict[str, object]] = []
    for rec in episode.records:
        competing = memory + [rec]
        if len(competing) <= capacity:
            memory = competing
            continue
        utilities = counterfactual_utilities(episode, competing, rec.position)
        scores = score_fn(episode, competing, rec.position)
        idx = min(range(len(competing)), key=lambda j: (float(scores[j]), competing[j].position))
        min_u = min(utilities.values())
        removed_u = utilities[competing[idx].uid]
        correct += int(math.isclose(removed_u, min_u))
        evictions += 1
        regret += removed_u - min_u
        if trace is not None:
            if renderer is None:
                raise ValueError("renderer is required when collecting an eviction trace")
            oracle_ids = {item.uid for item in competing if math.isclose(utilities[item.uid], min_u)}
            episode_traces.append(
                {
                    "split": split,
                    "policy": policy,
                    "episode": episode.eid,
                    "task": episode.task,
                    "capacity": capacity,
                    "decision": evictions,
                    "decision_position": rec.position,
                    "goal": renderer.goal(episode),
                    "candidate_uid": rec.uid,
                    "evicted_uid": competing[idx].uid,
                    "oracle_evictions": sorted(oracle_ids),
                    "regret": float(removed_u - min_u),
                    "records": [
                        {
                            "uid": item.uid,
                            "text": renderer.record(item),
                            "position": item.position,
                            "age": rec.position - item.position,
                            "is_candidate": item.uid == rec.uid,
                            "utility": float(utilities[item.uid]),
                            "score": float(scores[j]),
                            "evicted": j == idx,
                            "oracle": item.uid in oracle_ids,
                        }
                        for j, item in enumerate(competing)
                    ],
                }
            )
        memory = [r for j, r in enumerate(competing) if j != idx]
    ids = {r.uid for r in memory}
    clauses = [float(set(c).issubset(ids)) for c in episode.clauses]
    required = episode.required_ids
    stale = 0.0
    if episode.task == "state_overwrite":
        final_required = next(iter(required))
        stale = float(final_required not in ids and any(r.key == episode.target_key for r in memory))
    metrics = RunMetrics(
        success=float(all(clauses)),
        clause_accuracy=sum(clauses) / max(1, len(clauses)),
        required_recall=len(required & ids) / max(1, len(required)),
        eviction_accuracy=correct / max(1, evictions),
        eviction_regret=regret / max(1, evictions),
        stale_value_rate=stale,
    )
    if trace is not None:
        for item in episode_traces:
            item["episode_success"] = bool(metrics.success)
            item["episode_required_recall"] = float(metrics.required_recall)
        trace.extend(episode_traces)
    return metrics


def evaluate_split(
    episodes: Sequence[Episode],
    capacities: Sequence[int],
    split: str,
    seed: int,
    renderer: TextRenderer,
    text_scorer: TextScorer,
    structured: UtilityPredictor,
    traces: list[dict[str, object]] | None = None,
    trace_episodes: int = 0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, episode in enumerate(episodes):
        capacity = capacities[i % len(capacities)]
        text_trace = traces if traces is not None and i < trace_episodes else None
        baselines = {
            "random": run_episode(episode, capacity, "random", seed * 1_000_003 + i),
            "fifo": run_episode(episode, capacity, "fifo", seed * 1_000_003 + i),
            "oracle": run_episode(episode, capacity, "oracle", seed * 1_000_003 + i),
            "structured": run_episode(episode, capacity, "predicted", seed * 1_000_003 + i, predictor=structured),
            "lexical": run_scored_episode(
                episode, capacity, lambda ep, records, pos: lexical_scores(renderer, ep, records, pos)
            ),
            "text": run_scored_episode(
                episode,
                capacity,
                text_scorer.scores,
                renderer=renderer,
                split=split,
                policy="text",
                trace=text_trace,
            ),
        }
        for policy, metric in baselines.items():
            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "episode": episode.eid,
                    "task": episode.task,
                    "length": len(episode.records),
                    "capacity": capacity,
                    "policy": policy,
                    **asdict(metric),
                }
            )
        if (i + 1) % 20 == 0:
            print(f"eval {split}: {i + 1}/{len(episodes)}", flush=True)
    return rows


def summarize(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"scores": {}, "by_task": {}, "decision": {}}
    splits = sorted({str(r["split"]) for r in rows})
    policies = ("random", "fifo", "lexical", "text", "structured", "oracle")
    for split in splits:
        for policy in policies:
            values = [float(r["success"]) for r in rows if r["split"] == split and r["policy"] == policy]
            result["scores"][f"{split}/{policy}"] = float(np.mean(values))
        fifo = result["scores"][f"{split}/fifo"]
        oracle = result["scores"][f"{split}/oracle"]
        text = result["scores"][f"{split}/text"]
        lexical = result["scores"][f"{split}/lexical"]
        gap = oracle - fifo
        recovered = (text - fifo) / gap if gap > 1e-12 else float("nan")
        result["decision"][split] = {
            "oracle_fifo_gap": gap,
            "text_gap_recovered": recovered,
            "text_minus_lexical": text - lexical,
            "pass_50pct_and_beat_lexical": bool(recovered >= 0.50 and text - lexical >= 0.05),
        }
        for task in TASKS:
            item = {}
            for policy in ("fifo", "text", "structured", "oracle"):
                values = [
                    float(r["success"])
                    for r in rows
                    if r["split"] == split and r["task"] == task and r["policy"] == policy
                ]
                item[policy] = float(np.mean(values))
            result["by_task"][f"{split}/{task}"] = item
    return result


def write_outputs(
    output: Path,
    rows: Sequence[dict[str, object]],
    summary: dict[str, object],
    metadata: dict[str, object],
    history: Sequence[dict[str, float]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"metadata": metadata, "history": list(history), **summary}
    (output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Phase 0B2：纯文本效用预测结果",
        "",
        "文本模型不接收未来记录，也不接收模拟器中的结构化真值字段。",
        "",
        "| 测试集 | Random | FIFO | Lexical | Text | 结构化基线 | Oracle | Text 恢复比例 | 结论 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for split, decision in summary["decision"].items():
        score = summary["scores"]
        lines.append(
            f"| {split} | {score[f'{split}/random']:.3f} | {score[f'{split}/fifo']:.3f} | "
            f"{score[f'{split}/lexical']:.3f} | {score[f'{split}/text']:.3f} | "
            f"{score[f'{split}/structured']:.3f} | {score[f'{split}/oracle']:.3f} | "
            f"{decision['text_gap_recovered']:.1%} | "
            f"{'通过' if decision['pass_50pct_and_beat_lexical'] else '失败'} |"
        )
    lines.extend(["", "## 分任务成功率", "", "| 测试集/任务 | FIFO | Text | 结构化基线 | Oracle |", "|---|---:|---:|---:|---:|"])
    for key, item in summary["by_task"].items():
        lines.append(f"| {key} | {item['fifo']:.3f} | {item['text']:.3f} | {item['structured']:.3f} | {item['oracle']:.3f} |")
    lines.append("")
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/phase0_b2_text"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-episodes", type=int, default=1200)
    parser.add_argument("--eval-episodes", type=int, default=120)
    parser.add_argument("--sample-rate", type=float, default=0.30)
    parser.add_argument("--max-bytes", type=int, default=384)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="TensorBoard 日志目录；默认写到 <output>/tensorboard。",
    )
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument(
        "--trace-episodes",
        type=int,
        default=3,
        help="每个测试集保存多少个 episode 的逐步记忆淘汰记录。",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.train_episodes = 180
        args.eval_episodes = 24
        args.sample_rate = 0.25
        args.max_bytes = 256
        args.d_model = 128
        args.layers = 3
        args.heads = 4
        args.ff_dim = 512
        args.batch_size = 64
        args.grad_accum = 1
        args.epochs = 2
        args.workers = 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = args.tensorboard_dir or (args.output / "tensorboard")
    writer = create_summary_writer(tensorboard_dir, enabled=not args.no_tensorboard)
    capacities = (4, 8, 12)
    train_episodes = harden_episodes(
        make_episodes(args.train_episodes, (32, 48, 64), 41_000 + args.seed, args.seed * 10_000_000),
        81_000 + args.seed,
    )
    renderer_train = TextRenderer("train")
    examples = collect_text_examples(train_episodes, capacities, renderer_train, args.seed, args.sample_rate)
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    split = int(0.90 * len(examples))
    train_examples, val_examples = examples[:split], examples[split:]

    model = ByteUtilityTransformer(
        args.max_bytes, args.d_model, args.layers, args.heads, args.ff_dim, args.dropout
    ).to(device)
    print(
        json.dumps(
            {
                "device": str(device),
                "parameters": parameter_count(model),
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
                "positive_rate": sum(x.utility > 0 for x in examples) / max(1, len(examples)),
            }
        ),
        flush=True,
    )
    history = train_model(model, train_examples, val_examples, args, device, writer)

    # Structured baseline is trained on the same episode family but receives the
    # simulator fields that B2 deliberately hides from the text model.
    sx, sy = build_training_set(train_episodes, capacities, args.seed, args.sample_rate)
    structured = UtilityPredictor(args.seed)
    structured.fit(sx, sy)

    evaluations = (
        (
            "id",
            harden_episodes(make_episodes(args.eval_episodes, (48, 64), 51_000 + args.seed, 100_000_000), 91_000 + args.seed),
            TextRenderer("train"),
        ),
        (
            "paraphrase",
            harden_episodes(make_episodes(args.eval_episodes, (48, 64), 61_000 + args.seed, 200_000_000), 101_000 + args.seed),
            TextRenderer("paraphrase"),
        ),
        (
            "ood_length_paraphrase",
            harden_episodes(make_episodes(args.eval_episodes, (96, 128), 71_000 + args.seed, 300_000_000), 111_000 + args.seed),
            TextRenderer("paraphrase"),
        ),
    )
    rows: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    for name, episodes, renderer in evaluations:
        scorer = TextScorer(model, renderer, args.max_bytes, device)
        rows.extend(
            evaluate_split(
                episodes,
                capacities,
                name,
                args.seed,
                renderer,
                scorer,
                structured,
                traces,
                args.trace_episodes,
            )
        )
    summary = summarize(rows)
    metadata = {
        "seed": args.seed,
        "device": str(device),
        "parameters": parameter_count(model),
        "train_episodes": args.train_episodes,
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "eval_episodes_per_split": args.eval_episodes,
        "capacities": list(capacities),
        "model": {
            "max_bytes": args.max_bytes,
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
            "ff_dim": args.ff_dim,
        },
        "training": {
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "sample_rate": args.sample_rate,
        },
        "hard_negatives_per_episode": 14,
        "tensorboard_dir": str(tensorboard_dir),
        "trace_episodes_per_split": args.trace_episodes,
    }
    write_outputs(args.output, rows, summary, metadata, history)
    write_eviction_artifacts(args.output, traces)
    log_evaluation_to_tensorboard(writer, rows, summary, metadata, traces)
    torch.save({"model": model.state_dict(), "metadata": metadata}, args.output / "model.pt")
    if writer is not None:
        writer.close()
    print(json.dumps(summary["decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
