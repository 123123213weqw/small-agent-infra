#!/usr/bin/env python3
"""Train and evaluate one Phase 0B2.1 Stage-1 configuration."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.experiment_visualization import create_summary_writer, write_eviction_artifacts
from experiments.phase0_ab import RunMetrics, run_episode
from experiments.phase0_b21_data import (
    PreparedEpisode,
    make_decision_group,
    materialize_split_episodes,
)
from experiments.phase0_b21_model import (
    DecisionGroupDataset,
    GroupCollator,
    UtilitySetModel,
    move_batch,
    parameter_count,
    utility_losses,
)


TEST_SPLITS = ("test_id", "test_composition", "test_length", "test_semantic_stress")


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "name",
        "architecture",
        "d_model",
        "heads",
        "ff_dim",
        "record_layers",
        "pointwise_layers",
        "set_layers",
        "batch_size",
        "epochs",
        "learning_rate",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def resolve_seed_config(config: dict[str, Any]) -> dict[str, Any]:
    """Separate optimization randomness from frozen data/evaluation sampling."""
    legacy = int(config.get("seed", 0))
    return {
        **config,
        "model_seed": int(config.get("model_seed", legacy)),
        "data_seed": int(config.get("data_seed", legacy)),
        "evaluation_seed": int(config.get("evaluation_seed", legacy)),
    }


def collator_for(config: dict[str, Any]) -> GroupCollator:
    return GroupCollator(
        config["architecture"],
        record_bytes=int(config.get("record_bytes", 256)),
        context_bytes=int(config.get("context_bytes", 512)),
        pointwise_bytes=int(config.get("pointwise_bytes", 384)),
    )


def loader_for(
    dataset: DecisionGroupDataset,
    config: dict[str, Any],
    *,
    shuffle: bool,
    batch_size: int | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(config["model_seed"]))
    return DataLoader(
        dataset,
        batch_size=batch_size or int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config.get("workers", 4)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config.get("workers", 4)) > 0,
        collate_fn=collator_for(config),
        generator=generator,
    )


@torch.no_grad()
def validation_losses(
    model: UtilitySetModel,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = defaultdict(float)
    groups = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        predictions = model(batch)
        losses = utility_losses(predictions, batch, config)
        size = int(batch["set_mask"].shape[0])
        for name, loss in losses.items():
            totals[name] += float(loss.item()) * size
        groups += size
    return {name: value / max(1, groups) for name, value in totals.items()}


def save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(
    model: UtilitySetModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    checkpoint: Path,
    writer: Any | None,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    epochs = int(config["epochs"])
    accumulation = int(config.get("gradient_accumulation", 1))
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs * updates_per_epoch),
        eta_min=float(config.get("minimum_learning_rate", 3e-5)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float]] = []
    best = float("inf")
    best_selection = float("-inf")
    update = 0

    for epoch in range(epochs):
        model.train()
        totals = defaultdict(float)
        seen = 0
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for step, raw_batch in enumerate(train_loader):
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                predictions = model(batch)
                losses = utility_losses(predictions, batch, config)
                loss = losses["total"] / accumulation
            scaler.scale(loss).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update += 1
                if writer is not None and update % int(config.get("log_every_updates", 25)) == 0:
                    writer.add_scalar("training/learning_rate", optimizer.param_groups[0]["lr"], update)
            size = int(batch["set_mask"].shape[0])
            for name, value in losses.items():
                totals[name] += float(value.detach().item()) * size
            seen += size

        validation = validation_losses(model, validation_loader, config, device)
        row: dict[str, float] = {
            "epoch": float(epoch + 1),
            **{f"train_{name}": value / max(1, seen) for name, value in totals.items()},
            **{f"validation_{name}": value for name, value in validation.items()},
            "seconds": time.perf_counter() - started,
            "updates": float(update),
        }
        selection_mode = str(config.get("checkpoint_selection", "validation_total"))
        selection_value = -float(validation["total"])
        if selection_mode == "relation_rollout":
            selection_rows, _, _ = evaluate_rollout(
                model,
                str(config.get("selection_rollout_split", "validation")),
                config,
                device,
                int(config.get("selection_rollout_maximum_episodes", 0)) or None,
                preset=str(config.get("selection_rollout_preset", "stage1")),
                task_filter="relation_chain",
            )
            model_rows = [item for item in selection_rows if item["policy"] == "model"]
            if not model_rows:
                raise RuntimeError("relation rollout checkpoint selection produced no rows")
            relation_success = float(np.mean([float(item["success"]) for item in model_rows]))
            relation_recall = float(
                np.mean([float(item["required_recall"]) for item in model_rows])
            )
            relation_regret = float(
                np.mean([float(item["eviction_regret"]) for item in model_rows])
            )
            row.update(
                {
                    "validation_relation_rollout_success": relation_success,
                    "validation_relation_required_recall": relation_recall,
                    "validation_relation_eviction_regret": relation_regret,
                }
            )
            selection_value = relation_success
        elif selection_mode != "validation_total":
            raise ValueError(f"unknown checkpoint selection mode: {selection_mode}")
        history.append(row)
        print(json.dumps(row), flush=True)
        if writer is not None:
            for name, value in row.items():
                if name not in {"epoch", "updates"}:
                    writer.add_scalar(f"training/{name}", value, epoch + 1)
            writer.flush()
        improved = selection_value > best_selection + 1e-12
        tied_but_lower_loss = (
            abs(selection_value - best_selection) <= 1e-12
            and validation["total"] < best
        )
        if improved or tied_but_lower_loss:
            best = validation["total"]
            best_selection = selection_value
            save_checkpoint_atomic(
                checkpoint,
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch + 1,
                    "validation_total": best,
                    "selection_mode": selection_mode,
                    "selection_value": best_selection,
                },
            )
    return history


def _pairwise_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    true_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
    pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)
    valid = true_diff > 1e-8
    return int(((pred_diff > 0) & valid).sum().item()), int(valid.sum().item())


@torch.no_grad()
def evaluate_offline(
    model: UtilitySetModel,
    dataset: DecisionGroupDataset,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    loader = loader_for(
        dataset,
        config,
        shuffle=False,
        batch_size=int(config.get("evaluation_batch_size", config["batch_size"])),
    )
    model.eval()
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
    for raw_batch in loader:
        groups = raw_batch["groups"]
        batch = move_batch(raw_batch, device)
        predictions = model(batch).float().cpu()
        targets = raw_batch["utilities"]
        masks = raw_batch["set_mask"]
        for b, group in enumerate(groups):
            count = int(masks[b].sum().item())
            pred = predictions[b, :count]
            target = targets[b, :count]
            selected = min(
                range(count),
                key=lambda i: (float(pred[i]), int(group["records"][i]["event_index"])),
            )
            minimum = float(target.min().item())
            pair_correct, pair_total = _pairwise_accuracy(pred, target)
            rows.append(
                {
                    "split": group["split"],
                    "episode": group["episode_id"],
                    "decision": group["decision_id"],
                    "task": group["task"],
                    "query_visibility": group["query_visibility"],
                    "capacity": group["capacity"],
                    "hard_negative_count": group["hard_negative_count"],
                    "mse": float(torch.square(pred - target).mean().item()),
                    "pairwise_correct": pair_correct,
                    "pairwise_total": pair_total,
                    "argmin_correct": int(selected in group["oracle_eviction_indices"]),
                    "regret": float(target[selected].item() - minimum),
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    pair_correct = sum(int(row["pairwise_correct"]) for row in rows)
    pair_total = sum(int(row["pairwise_total"]) for row in rows)
    summary = {
        "groups": float(len(rows)),
        "mse": float(np.mean([float(row["mse"]) for row in rows])),
        "pairwise_accuracy": pair_correct / max(1, pair_total),
        "argmin_accuracy": float(np.mean([float(row["argmin_correct"]) for row in rows])),
        "regret": float(np.mean([float(row["regret"]) for row in rows])),
        "seconds": seconds,
        "groups_per_second": len(rows) / max(1e-9, seconds),
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        ),
    }
    return rows, summary


@torch.no_grad()
def predict_groups(
    model: UtilitySetModel,
    groups: Sequence[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
) -> list[list[float]]:
    collator = collator_for(config)
    size = int(config.get("evaluation_batch_size", config["batch_size"]))
    result: list[list[float]] = []
    model.eval()
    for start in range(0, len(groups), size):
        selected = groups[start : start + size]
        batch = move_batch(collator(selected), device)
        predictions = model(batch).float().cpu()
        for row, group in zip(predictions, selected):
            result.append(row[: len(group["records"])].tolist())
    return result


def _final_metrics(prepared: PreparedEpisode, state: dict[str, Any]) -> RunMetrics:
    episode = prepared.episode
    ids = {record.uid for record in state["memory"]}
    clauses = [float(set(clause).issubset(ids)) for clause in episode.clauses]
    required = episode.required_ids
    stale = 0.0
    if episode.task == "state_overwrite":
        final_required = next(iter(required))
        stale = float(final_required not in ids and any(r.key == episode.target_key for r in state["memory"]))
    return RunMetrics(
        success=float(all(clauses)),
        clause_accuracy=sum(clauses) / max(1, len(clauses)),
        required_recall=len(required & ids) / max(1, len(required)),
        eviction_accuracy=state["correct"] / max(1, state["evictions"]),
        eviction_regret=state["regret"] / max(1, state["evictions"]),
        stale_value_rate=stale,
    )


def _trace_decision(
    group: dict[str, Any],
    scores: Sequence[float],
    evicted: int,
    episode_success: bool,
) -> dict[str, Any]:
    oracle = set(group["oracle_eviction_indices"])
    minimum = min(group["utilities"])
    return {
        "split": group["split"],
        "policy": "model",
        "episode": group["episode_id"],
        "task": group["task"],
        "capacity": group["capacity"],
        "decision": group["decision_id"],
        "decision_position": group["decision_position"],
        "goal": group["goal"],
        "candidate_uid": group["records"][group["candidate_index"]]["uid"],
        "evicted_uid": group["records"][evicted]["uid"],
        "oracle_evictions": [group["records"][i]["uid"] for i in sorted(oracle)],
        "regret": float(group["utilities"][evicted] - minimum),
        "episode_success": episode_success,
        "records": [
            {
                "uid": record["uid"],
                "text": record["text"],
                "position": record["event_index"],
                "age": record["relative_age"],
                "is_candidate": record["is_candidate"],
                "utility": float(group["utilities"][i]),
                "score": float(scores[i]),
                "evicted": i == evicted,
                "oracle": i in oracle,
            }
            for i, record in enumerate(group["records"])
        ],
    }


@torch.no_grad()
def evaluate_rollout(
    model: UtilitySetModel,
    split: str,
    config: dict[str, Any],
    device: torch.device,
    maximum_episodes: int | None = None,
    *,
    preset: str = "stage1",
    task_filter: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, Any]], dict[str, float]]:
    prepared = materialize_split_episodes(
        preset, int(config["evaluation_seed"]), split
    )
    if task_filter is not None:
        prepared = [item for item in prepared if item.episode.task == task_filter]
    if maximum_episodes is not None:
        prepared = prepared[:maximum_episodes]
    states = [
        {"memory": [], "correct": 0, "evictions": 0, "regret": 0.0, "trace": []}
        for _ in prepared
    ]
    maximum_length = max(len(item.episode.records) for item in prepared)
    scoring_seconds = 0.0
    scored_groups = 0

    for position in range(maximum_length):
        pending: list[tuple[int, dict[str, Any], Sequence[Any]]] = []
        for index, (item, state) in enumerate(zip(prepared, states)):
            if position >= len(item.episode.records):
                continue
            candidate = item.episode.records[position]
            competing = [*state["memory"], candidate]
            if len(competing) <= item.capacity:
                state["memory"] = competing
                continue
            group = make_decision_group(
                item.episode,
                competing,
                candidate,
                split=split,
                capacity=item.capacity,
                hard_negative_count=item.hard_negative_count,
                template_family=item.template_family,
                behavior_policy="model",
                decision_id=state["evictions"] + 1,
                renderer=item.renderer,
            )
            pending.append((index, group, competing))

        if not pending:
            continue
        groups = [group for _, group, _ in pending]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        scores = predict_groups(model, groups, config, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        scoring_seconds += time.perf_counter() - started
        scored_groups += len(groups)
        for (state_index, group, competing), predicted in zip(pending, scores):
            state = states[state_index]
            selected = min(
                range(len(competing)),
                key=lambda i: (float(predicted[i]), int(group["records"][i]["event_index"])),
            )
            minimum = min(group["utilities"])
            removed = float(group["utilities"][selected])
            state["correct"] += int(selected in group["oracle_eviction_indices"])
            state["evictions"] += 1
            state["regret"] += removed - minimum
            if state_index < int(config.get("trace_episodes", 1)):
                state["trace"].append((group, predicted, selected))
            state["memory"] = [record for i, record in enumerate(competing) if i != selected]

    rows: list[dict[str, object]] = []
    traces: list[dict[str, Any]] = []
    for i, (item, state) in enumerate(zip(prepared, states)):
        model_metrics = _final_metrics(item, state)
        common = {
            "split": split,
            "episode": item.episode.eid,
            "task": item.episode.task,
            "query_visibility": item.query_visibility,
            "capacity": item.capacity,
            "hard_negative_count": item.hard_negative_count,
        }
        rows.append({**common, "policy": "model", **asdict(model_metrics)})
        fifo = run_episode(item.episode, item.capacity, "fifo", seed=i)
        oracle = run_episode(item.episode, item.capacity, "oracle", seed=i)
        rows.append({**common, "policy": "fifo", **asdict(fifo)})
        rows.append({**common, "policy": "oracle", **asdict(oracle)})
        if i < int(config.get("trace_episodes", 1)):
            for group, predicted, selected in state["trace"]:
                traces.append(
                    _trace_decision(group, predicted, selected, bool(model_metrics.success))
                )

    summary = {
        "scored_groups": float(scored_groups),
        "scoring_seconds": scoring_seconds,
        "groups_per_second": scored_groups / max(1e-9, scoring_seconds),
    }
    return rows, traces, summary


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: Sequence[dict[str, object]], keys: Sequence[str], metrics: Sequence[str]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    result: dict[str, Any] = {}
    for group_key, items in groups.items():
        name = "/".join(group_key)
        result[name] = {
            metric: float(np.mean([float(item[metric]) for item in items])) for metric in metrics
        }
        result[name]["count"] = len(items)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/b2_1_stage1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/b2_1_stage1"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    config = resolve_seed_config(load_config(args.config))
    if args.smoke:
        config = {
            **config,
            "epochs": 1,
            "batch_size": 4,
            "evaluation_batch_size": 8,
            "workers": 2,
            "maximum_train_groups": 192,
            "maximum_validation_groups": 96,
            "maximum_test_groups": 96,
            "maximum_rollout_episodes": 12,
        }
    seed_everything(int(config["model_seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("B2.1 training requires CUDA")
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.data_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_seed = int(manifest["seed"])
        if manifest_seed != int(config["data_seed"]):
            raise ValueError(
                f"data seed mismatch: config={config['data_seed']} manifest={manifest_seed}"
            )
    (args.output / "config.resolved.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    writer = create_summary_writer(args.output / "tensorboard", enabled=not args.no_tensorboard)

    train_data = DecisionGroupDataset(args.data_dir / "train.jsonl")
    validation_data = DecisionGroupDataset(args.data_dir / "validation.jsonl")
    if config.get("maximum_train_groups"):
        train_data.groups = train_data.groups[: int(config["maximum_train_groups"])]
    if config.get("maximum_validation_groups"):
        validation_data.groups = validation_data.groups[: int(config["maximum_validation_groups"])]
    train_loader = loader_for(train_data, config, shuffle=True)
    validation_loader = loader_for(validation_data, config, shuffle=False)

    model = UtilitySetModel(config).to(device)
    parameters = parameter_count(model)
    print(
        json.dumps(
            {
                "name": config["name"],
                "device": str(device),
                "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "parameters": parameters,
                "train_groups": len(train_data),
                "validation_groups": len(validation_data),
            }
        ),
        flush=True,
    )
    if writer is not None:
        writer.add_text("run/config", f"```json\n{json.dumps(config, indent=2, ensure_ascii=False)}\n```")
        writer.add_scalar("run/parameters", parameters, 0)

    checkpoint = args.checkpoint_dir / config["name"] / "best.pt"
    history = train(model, train_loader, validation_loader, config, device, checkpoint, writer)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])

    offline_rows: list[dict[str, object]] = []
    offline_summary: dict[str, Any] = {}
    rollout_rows: list[dict[str, object]] = []
    rollout_summary: dict[str, Any] = {}
    traces: list[dict[str, Any]] = []
    for split in TEST_SPLITS:
        dataset = DecisionGroupDataset(args.data_dir / f"{split}.jsonl")
        if config.get("maximum_test_groups"):
            dataset.groups = dataset.groups[: int(config["maximum_test_groups"])]
        split_rows, split_summary = evaluate_offline(model, dataset, config, device)
        offline_rows.extend(split_rows)
        offline_summary[split] = split_summary
        print(json.dumps({"offline": split, **split_summary}), flush=True)

        maximum_rollout = config.get("maximum_rollout_episodes")
        split_rollout, split_traces, speed = evaluate_rollout(
            model,
            split,
            config,
            device,
            int(maximum_rollout) if maximum_rollout else None,
            preset=str(config.get("rollout_preset", "stage1")),
        )
        rollout_rows.extend(split_rollout)
        traces.extend(split_traces)
        rollout_summary[split] = speed
        print(json.dumps({"rollout": split, **speed}), flush=True)

    offline_aggregates = aggregate_rows(
        offline_rows,
        ("split",),
        ("mse", "argmin_correct", "regret"),
    )
    # Pairwise accuracy must aggregate counts, not per-group NaNs/means.
    for split in TEST_SPLITS:
        subset = [row for row in offline_rows if row["split"] == split]
        correct = sum(int(row["pairwise_correct"]) for row in subset)
        total = sum(int(row["pairwise_total"]) for row in subset)
        offline_aggregates[split]["pairwise_accuracy"] = correct / max(1, total)

    rollout_aggregates = aggregate_rows(
        rollout_rows,
        ("split", "policy"),
        (
            "success",
            "clause_accuracy",
            "required_recall",
            "eviction_accuracy",
            "eviction_regret",
            "stale_value_rate",
        ),
    )
    rollout_by_task = aggregate_rows(
        [row for row in rollout_rows if row["policy"] == "model"],
        ("split", "task"),
        ("success", "required_recall", "eviction_regret", "stale_value_rate"),
    )
    rollout_by_visibility = aggregate_rows(
        [row for row in rollout_rows if row["policy"] == "model"],
        ("split", "query_visibility"),
        ("success", "required_recall", "eviction_regret"),
    )
    summary = {
        "name": config["name"],
        "architecture": config["architecture"],
        "parameters": parameters,
        "best_epoch": int(saved["epoch"]),
        "best_validation_total": float(saved["validation_total"]),
        "checkpoint_selection": str(saved.get("selection_mode", "validation_total")),
        "best_selection_value": float(saved.get("selection_value", -saved["validation_total"])),
        "history": history,
        "offline": offline_aggregates,
        "offline_efficiency": offline_summary,
        "rollout": rollout_aggregates,
        "rollout_by_task": rollout_by_task,
        "rollout_by_query_visibility": rollout_by_visibility,
        "rollout_efficiency": rollout_summary,
    }
    write_csv(args.output / "offline_metrics.csv", offline_rows)
    write_csv(args.output / "rollout_metrics.csv", rollout_rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_eviction_artifacts(args.output, traces)
    if writer is not None:
        for split, item in offline_aggregates.items():
            for metric, value in item.items():
                if metric != "count":
                    writer.add_scalar(f"offline/{split}/{metric}", value, 0)
        for key, item in rollout_aggregates.items():
            for metric, value in item.items():
                if metric != "count":
                    writer.add_scalar(f"rollout/{key}/{metric}", value, 0)
        writer.flush()
        writer.close()
    print(json.dumps({"complete": config["name"], "summary": str(args.output / 'summary.json')}), flush=True)


if __name__ == "__main__":
    main()
