#!/usr/bin/env python3
"""Aggregate the pre-registered three-seed A/C confirmation."""

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


def mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs(stage1_dir: Path, seed_dir: Path) -> dict[int, dict[str, dict[str, Any]]]:
    return {
        0: {
            "A": load_summary(stage1_dir / "A_pointwise" / "summary.json"),
            "C": load_summary(stage1_dir / "C_joint_rank" / "summary.json"),
        },
        1: {
            "A": load_summary(seed_dir / "A_seed1" / "summary.json"),
            "C": load_summary(seed_dir / "C_seed1" / "summary.json"),
        },
        2: {
            "A": load_summary(seed_dir / "A_seed2" / "summary.json"),
            "C": load_summary(seed_dir / "C_seed2" / "summary.json"),
        },
    }


def rollout(run: dict[str, Any], split: str) -> dict[str, float]:
    return run["rollout"][f"{split}/model"]


def average_regret(run: dict[str, Any]) -> float:
    return mean([rollout(run, split)["eviction_regret"] for split in SPLITS])


def build_report(
    runs: dict[int, dict[str, dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, object]], str]:
    rows: list[dict[str, object]] = []
    seed_comparisons: list[dict[str, float | int | bool]] = []
    for seed in sorted(runs):
        a, c = runs[seed]["A"], runs[seed]["C"]
        a_regret, c_regret = average_regret(a), average_regret(c)
        comp_a = rollout(a, "test_composition")["success"]
        comp_c = rollout(c, "test_composition")["success"]
        reduction = (a_regret - c_regret) / a_regret if a_regret > 1e-12 else 0.0
        seed_comparisons.append(
            {
                "seed": seed,
                "composition_A": comp_a,
                "composition_C": comp_c,
                "composition_gain": comp_c - comp_a,
                "mean_regret_A": a_regret,
                "mean_regret_C": c_regret,
                "regret_reduction": reduction,
                "both_better": comp_c > comp_a and c_regret < a_regret,
                "no_catastrophic_regression": comp_c + 1e-12 >= comp_a and reduction >= -0.10,
            }
        )
        for model in ("A", "C"):
            for split in SPLITS:
                item = rollout(runs[seed][model], split)
                rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "split": split,
                        "success": item["success"],
                        "required_recall": item["required_recall"],
                        "eviction_regret": item["eviction_regret"],
                        "stale_value_rate": item["stale_value_rate"],
                    }
                )

    mean_comp_gain = mean([float(item["composition_gain"]) for item in seed_comparisons])
    mean_a_regret = mean([float(item["mean_regret_A"]) for item in seed_comparisons])
    mean_c_regret = mean([float(item["mean_regret_C"]) for item in seed_comparisons])
    mean_regret_reduction = (
        (mean_a_regret - mean_c_regret) / mean_a_regret if mean_a_regret > 1e-12 else 0.0
    )
    both_better = sum(bool(item["both_better"]) for item in seed_comparisons)
    no_catastrophic = all(bool(item["no_catastrophic_regression"]) for item in seed_comparisons)

    task_rows = []
    for task in TASKS:
        a_values = [
            runs[seed]["A"]["rollout_by_task"][f"test_composition/{task}"]["success"]
            for seed in sorted(runs)
        ]
        c_values = [
            runs[seed]["C"]["rollout_by_task"][f"test_composition/{task}"]["success"]
            for seed in sorted(runs)
        ]
        task_rows.append(
            {
                "task": task,
                "A_mean": mean(a_values),
                "C_mean": mean(c_values),
                "C_minus_A": mean(c_values) - mean(a_values),
            }
        )
    non_decreasing_tasks = sum(
        float(item["C_mean"]) + 1e-12 >= float(item["A_mean"]) for item in task_rows
    )

    parameter_ratios = []
    throughput_ratios = []
    memory_ratios = []
    for seed in sorted(runs):
        a, c = runs[seed]["A"], runs[seed]["C"]
        parameter_ratios.append(c["parameters"] / a["parameters"])
        a_eff = a["offline_efficiency"]["test_composition"]
        c_eff = c["offline_efficiency"]["test_composition"]
        throughput_ratios.append(c_eff["groups_per_second"] / a_eff["groups_per_second"])
        memory_ratios.append(c_eff["peak_memory_mib"] / a_eff["peak_memory_mib"])
    parameter_ratio = mean(parameter_ratios)
    throughput_ratio = mean(throughput_ratios)
    memory_ratio = mean(memory_ratios)

    gates = {
        "mean_composition_gain_at_least_5pp": {
            "value": mean_comp_gain,
            "pass": mean_comp_gain >= 0.05,
        },
        "mean_regret_reduction_at_least_20pct": {
            "value": mean_regret_reduction,
            "pass": mean_regret_reduction >= 0.20,
        },
        "at_least_4_of_6_tasks_non_decreasing": {
            "value": non_decreasing_tasks,
            "pass": non_decreasing_tasks >= 4,
        },
        "at_least_2_seeds_better_on_both": {
            "value": both_better,
            "pass": both_better >= 2,
        },
        "no_catastrophic_seed": {"value": no_catastrophic, "pass": no_catastrophic},
        "efficiency_acceptable": {
            "parameter_ratio_C_over_A": parameter_ratio,
            "throughput_ratio_C_over_A": throughput_ratio,
            "memory_ratio_C_over_A": memory_ratio,
            "pass": (
                abs(parameter_ratio - 1.0) <= 0.01
                and throughput_ratio >= 0.50
                and memory_ratio <= 1.25
            ),
        },
    }
    passed = all(item["pass"] for item in gates.values())

    split_rows = []
    for split in SPLITS:
        a_values = [rollout(runs[seed]["A"], split)["success"] for seed in sorted(runs)]
        c_values = [rollout(runs[seed]["C"], split)["success"] for seed in sorted(runs)]
        split_rows.append(
            {
                "split": split,
                "A_mean": mean(a_values),
                "A_std": sample_std(a_values),
                "C_mean": mean(c_values),
                "C_std": sample_std(c_values),
                "C_minus_A": mean([c - a for a, c in zip(a_values, c_values)]),
            }
        )

    summary = {
        "decision": "pass_b2_1_seed_confirmation" if passed else "stop_and_diagnose",
        "pass": passed,
        "gates": gates,
        "seed_comparisons": seed_comparisons,
        "split_success": split_rows,
        "composition_tasks": task_rows,
        "mean_rollout_regret": {"A": mean_a_regret, "C": mean_c_regret},
    }

    lines = [
        "# Phase 0B2.1 A/C 多种子确认结果",
        "",
        "C 是 Seed 0 诊断后选出的候选。本报告按随后冻结的 A/C 多种子计划，使用相同 data seed 和 evaluation seed，只改变 model seed。",
        "",
        "## 逐 Seed 配对结果",
        "",
        "| Seed | A 组合成功率 | C 组合成功率 | C−A | A 平均 regret | C 平均 regret | regret 降幅 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in seed_comparisons:
        lines.append(
            f"| {item['seed']} | {item['composition_A']:.3f} | {item['composition_C']:.3f} | "
            f"{item['composition_gain']:+.3f} | {item['mean_regret_A']:.6f} | "
            f"{item['mean_regret_C']:.6f} | {item['regret_reduction']:+.1%} |"
        )
    lines.extend(
        [
            "",
            "## 三种子成功率",
            "",
            "| 测试集 | A 均值±标准差 | C 均值±标准差 | 配对均值差 C−A |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in split_rows:
        lines.append(
            f"| {item['split']} | {item['A_mean']:.3f}±{item['A_std']:.3f} | "
            f"{item['C_mean']:.3f}±{item['C_std']:.3f} | {item['C_minus_A']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## 组合任务",
            "",
            "| 任务 | A 三种子均值 | C 三种子均值 | C−A |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in task_rows:
        lines.append(
            f"| {item['task']} | {item['A_mean']:.3f} | {item['C_mean']:.3f} | {item['C_minus_A']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## 冻结门槛",
            "",
            "| 条件 | 数值 | 结论 |",
            "|---|---:|:---:|",
            f"| 平均组合成功率增益 | {mean_comp_gain:+.1%} | {'通过' if gates['mean_composition_gain_at_least_5pp']['pass'] else '失败'} |",
            f"| 平均 regret 降幅 | {mean_regret_reduction:+.1%} | {'通过' if gates['mean_regret_reduction_at_least_20pct']['pass'] else '失败'} |",
            f"| 不下降任务数 | {non_decreasing_tasks}/6 | {'通过' if gates['at_least_4_of_6_tasks_non_decreasing']['pass'] else '失败'} |",
            f"| 同时改善成功率和 regret 的 Seed | {both_better}/3 | {'通过' if gates['at_least_2_seeds_better_on_both']['pass'] else '失败'} |",
            f"| 无灾难性 Seed | {no_catastrophic} | {'通过' if no_catastrophic else '失败'} |",
            f"| C/A 吞吐比；显存比；参数比 | {throughput_ratio:.2f}；{memory_ratio:.2f}；{parameter_ratio:.4f} | {'通过' if gates['efficiency_acceptable']['pass'] else '失败'} |",
            "",
            f"**多种子确认结论：{'通过' if passed else '停止并诊断'}。**",
            "",
        ]
    )
    return summary, rows, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-dir", type=Path, default=Path("runs/b2_1_stage1"))
    parser.add_argument("--seed-dir", type=Path, default=Path("runs/b2_1_ac_seeds"))
    parser.add_argument("--output", type=Path, default=Path("results/phase0_b21_ac_seeds"))
    args = parser.parse_args()
    runs = load_runs(args.stage1_dir, args.seed_dir)
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
