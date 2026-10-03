#!/usr/bin/env python3
"""Aggregate the frozen relation-data-only C experiment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Sequence


SPLITS = ("test_id", "test_composition", "test_length", "test_semantic_stress")
TASKS = (
    "delayed_query",
    "state_overwrite",
    "long_instruction",
    "completed_intermediate",
    "unresolved_subgoal",
    "relation_chain",
)
NON_RELATION_TASKS = TASKS[:-1]


def mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs(
    stage1_dir: Path,
    ac_seed_dir: Path,
    augmented_dir: Path,
) -> dict[int, dict[str, dict[str, Any]]]:
    return {
        0: {
            "baseline": load_summary(stage1_dir / "C_joint_rank" / "summary.json"),
            "augmented": load_summary(augmented_dir / "C_seed0" / "summary.json"),
        },
        1: {
            "baseline": load_summary(ac_seed_dir / "C_seed1" / "summary.json"),
            "augmented": load_summary(augmented_dir / "C_seed1" / "summary.json"),
        },
        2: {
            "baseline": load_summary(ac_seed_dir / "C_seed2" / "summary.json"),
            "augmented": load_summary(augmented_dir / "C_seed2" / "summary.json"),
        },
    }


def rollout(run: dict[str, Any], split: str) -> dict[str, float]:
    return run["rollout"][f"{split}/model"]


def task_success(run: dict[str, Any], task: str) -> float:
    return run["rollout_by_task"][f"test_composition/{task}"]["success"]


def average_regret(run: dict[str, Any]) -> float:
    return mean([rollout(run, split)["eviction_regret"] for split in SPLITS])


def build_report(
    runs: dict[int, dict[str, dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, object]], str]:
    seeds = sorted(runs)
    rows: list[dict[str, object]] = []
    seed_rows = []
    for seed in seeds:
        baseline = runs[seed]["baseline"]
        augmented = runs[seed]["augmented"]
        relation_before = task_success(baseline, "relation_chain")
        relation_after = task_success(augmented, "relation_chain")
        composition_before = rollout(baseline, "test_composition")["success"]
        composition_after = rollout(augmented, "test_composition")["success"]
        regret_before = average_regret(baseline)
        regret_after = average_regret(augmented)
        seed_rows.append(
            {
                "seed": seed,
                "relation_before": relation_before,
                "relation_after": relation_after,
                "relation_gain": relation_after - relation_before,
                "composition_before": composition_before,
                "composition_after": composition_after,
                "composition_gain": composition_after - composition_before,
                "regret_before": regret_before,
                "regret_after": regret_after,
            }
        )
        for variant in ("baseline", "augmented"):
            for split in SPLITS:
                item = rollout(runs[seed][variant], split)
                rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "split": split,
                        "success": item["success"],
                        "required_recall": item["required_recall"],
                        "eviction_regret": item["eviction_regret"],
                        "stale_value_rate": item["stale_value_rate"],
                    }
                )

    task_rows = []
    for task in TASKS:
        before = [task_success(runs[seed]["baseline"], task) for seed in seeds]
        after = [task_success(runs[seed]["augmented"], task) for seed in seeds]
        task_rows.append(
            {
                "task": task,
                "baseline_mean": mean(before),
                "augmented_mean": mean(after),
                "gain": mean(after) - mean(before),
            }
        )

    split_rows = []
    for split in SPLITS:
        before = [rollout(runs[seed]["baseline"], split)["success"] for seed in seeds]
        after = [rollout(runs[seed]["augmented"], split)["success"] for seed in seeds]
        split_rows.append(
            {
                "split": split,
                "baseline_mean": mean(before),
                "baseline_std": sample_std(before),
                "augmented_mean": mean(after),
                "augmented_std": sample_std(after),
                "gain": mean(after) - mean(before),
            }
        )

    relation_values = [float(item["relation_after"]) for item in seed_rows]
    relation_mean = mean(relation_values)
    baseline_composition = mean([float(item["composition_before"]) for item in seed_rows])
    augmented_composition = mean([float(item["composition_after"]) for item in seed_rows])
    non_relation = [item for item in task_rows if item["task"] in NON_RELATION_TASKS]
    within_2pp = sum(float(item["gain"]) >= -0.02 for item in non_relation)
    no_drop_over_5pp = all(float(item["gain"]) >= -0.05 for item in non_relation)
    regret_before = mean([float(item["regret_before"]) for item in seed_rows])
    regret_after = mean([float(item["regret_after"]) for item in seed_rows])
    regret_increase = (
        (regret_after - regret_before) / regret_before if regret_before > 1e-12 else 0.0
    )
    parameter_ratios = [
        runs[seed]["augmented"]["parameters"] / runs[seed]["baseline"]["parameters"]
        for seed in seeds
    ]
    parameter_ratio = mean(parameter_ratios)

    gates = {
        "relation_mean_at_least_50pct": {
            "value": relation_mean,
            "pass": relation_mean >= 0.50,
        },
        "every_seed_relation_at_least_35pct": {
            "values": relation_values,
            "pass": all(value >= 0.35 for value in relation_values),
        },
        "composition_within_2pp": {
            "baseline": baseline_composition,
            "augmented": augmented_composition,
            "pass": augmented_composition >= baseline_composition - 0.02,
        },
        "other_tasks_preserved": {
            "within_2pp": within_2pp,
            "no_drop_over_5pp": no_drop_over_5pp,
            "pass": within_2pp >= 4 and no_drop_over_5pp,
        },
        "regret_increase_at_most_10pct": {
            "value": regret_increase,
            "pass": regret_increase <= 0.10,
        },
        "parameters_unchanged": {
            "ratio": parameter_ratio,
            "pass": abs(parameter_ratio - 1.0) <= 1e-12,
        },
    }
    passed = all(item["pass"] for item in gates.values())
    summary = {
        "decision": "relation_data_fix_passed" if passed else "relation_data_fix_failed",
        "pass": passed,
        "gates": gates,
        "seed_comparisons": seed_rows,
        "split_success": split_rows,
        "composition_tasks": task_rows,
        "mean_rollout_regret": {"baseline": regret_before, "augmented": regret_after},
    }

    lines = [
        "# Phase 0B2.1 关系链数据增强结果",
        "",
        "本实验保持 C 的模型、损失、训练规模和评估集不变，只增加训练与验证中的关系表达多样性。",
        "",
        "## 逐 Seed 关系链结果",
        "",
        "| Seed | 原 C 关系链 | 数据增强 C | 增益 | 原 C 总组合 | 数据增强 C 总组合 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in seed_rows:
        lines.append(
            f"| {item['seed']} | {item['relation_before']:.3f} | {item['relation_after']:.3f} | "
            f"{item['relation_gain']:+.3f} | {item['composition_before']:.3f} | "
            f"{item['composition_after']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 四测试集成功率",
            "",
            "| 测试集 | 原 C 均值±标准差 | 数据增强 C 均值±标准差 | 增益 |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in split_rows:
        lines.append(
            f"| {item['split']} | {item['baseline_mean']:.3f}±{item['baseline_std']:.3f} | "
            f"{item['augmented_mean']:.3f}±{item['augmented_std']:.3f} | {item['gain']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## 组合测试分任务",
            "",
            "| 任务 | 原 C | 数据增强 C | 增益 |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in task_rows:
        lines.append(
            f"| {item['task']} | {item['baseline_mean']:.3f} | "
            f"{item['augmented_mean']:.3f} | {item['gain']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## 冻结门槛",
            "",
            "| 条件 | 数值 | 结论 |",
            "|---|---:|:---:|",
            f"| 关系链三种子均值至少 50% | {relation_mean:.1%} | {'通过' if gates['relation_mean_at_least_50pct']['pass'] else '失败'} |",
            f"| 每个 Seed 至少 35% | {', '.join(f'{value:.1%}' for value in relation_values)} | {'通过' if gates['every_seed_relation_at_least_35pct']['pass'] else '失败'} |",
            f"| 总组合成功率最多下降 2pp | {baseline_composition:.1%} → {augmented_composition:.1%} | {'通过' if gates['composition_within_2pp']['pass'] else '失败'} |",
            f"| 其他任务保持 | {within_2pp}/5 在 2pp 内；最大下降不超过 5pp={no_drop_over_5pp} | {'通过' if gates['other_tasks_preserved']['pass'] else '失败'} |",
            f"| 平均 regret 增幅不超过 10% | {regret_increase:+.1%} | {'通过' if gates['regret_increase_at_most_10pct']['pass'] else '失败'} |",
            f"| 参数量不变 | {parameter_ratio:.4f} | {'通过' if gates['parameters_unchanged']['pass'] else '失败'} |",
            "",
            f"**数据增强结论：{'通过' if passed else '失败'}。**",
            "",
        ]
    )
    return summary, rows, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-dir", type=Path, default=Path("runs/b2_1_stage1"))
    parser.add_argument("--ac-seed-dir", type=Path, default=Path("runs/b2_1_ac_seeds"))
    parser.add_argument("--augmented-dir", type=Path, default=Path("runs/b2_1_relation_data"))
    parser.add_argument("--output", type=Path, default=Path("results/phase0_b21_relation_data"))
    args = parser.parse_args()
    runs = load_runs(args.stage1_dir, args.ac_seed_dir, args.augmented_dir)
    summary, rows, markdown = build_report(runs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "RESULTS.md").write_text(markdown, encoding="utf-8")
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
