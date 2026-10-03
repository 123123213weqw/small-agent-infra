#!/usr/bin/env python3
"""Prepare a fixed regression/holdout split and compare v1/v2 audit outputs.

The 400 source excerpts and model predictions stay outside the public repo.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random


def sample_key(source_file: str, row_idx: int) -> tuple[str, int]:
    return source_file, int(row_idx)


def task_key(task: dict) -> tuple[str, int]:
    data = task["data"]
    return sample_key(data["source_file"], data["source_row_idx"])


def prediction_key(prediction: dict) -> tuple[str, int]:
    source = prediction["source"]
    return sample_key(source["data.source_file"], source["data.source_row_idx"])


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_tasks(path: Path) -> dict[tuple[str, int], dict]:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("tasks must be a JSON array")
    mapping = {task_key(task): task for task in tasks}
    if len(mapping) != len(tasks):
        raise ValueError("duplicate task source IDs")
    return mapping


def load_cases(path: Path) -> dict[tuple[str, int], str]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    mapping = {
        sample_key(case["source_file"], case["source_row_idx"]): case["expected_decision"]
        for case in cases
    }
    if len(mapping) != len(cases):
        raise ValueError("duplicate regression case IDs")
    return mapping


def load_predictions(path: Path, tasks: dict[tuple[str, int], dict]) -> dict[tuple[str, int], dict]:
    predictions = {}
    versions = set()
    for result in read_jsonl(path):
        key = prediction_key(result)
        if key not in tasks:
            raise ValueError(f"unknown prediction source ID: {key}")
        expected_source_id = json.dumps([key[0], key[1]], ensure_ascii=False, separators=(",", ":"))
        if result.get("source_id") != expected_source_id:
            raise ValueError(f"source_id mismatch: {key}")
        if result["text_sha256"] != text_hash(tasks[key]["data"]["text"]):
            raise ValueError(f"text hash mismatch: {key}")
        if result.get("status") == "ok":
            versions.add((result["prompt_sha256"], result["schema_sha256"], result["model"], result.get("temperature")))
            predictions[key] = result
    if len(versions) > 1:
        raise ValueError(f"mixed prompt/schema/model versions in {path}")
    return predictions


def prepare(
    tasks: dict[tuple[str, int], dict],
    cases: dict[tuple[str, int], str],
    dev_keys: set[tuple[str, int]],
    per_source: int,
    seed: int,
) -> tuple[list[dict], dict]:
    if not cases.keys() <= tasks.keys():
        raise ValueError("regression case missing from tasks")
    if not dev_keys <= tasks.keys():
        raise ValueError("dev sample missing from tasks")
    if not cases.keys() <= dev_keys:
        raise ValueError("regression cases must be part of the development sample")
    if per_source < 1:
        raise ValueError("per_source must be >= 1")
    regression = [tasks[key] for key in sorted(cases)]
    sources = sorted({source for source, _ in tasks})
    rng = random.Random(seed)
    selected = []
    for source in sources:
        candidates = sorted(key for key in tasks if key[0] == source and key not in dev_keys)
        if len(candidates) < per_source:
            raise ValueError(f"not enough holdout tasks for {source}")
        selected.extend(sorted(rng.sample(candidates, per_source)))
    holdout = {
        "seed": seed,
        "per_source": per_source,
        "dev_count": len(dev_keys),
        "ids": [{"source_file": source, "source_row_idx": row} for source, row in selected],
    }
    return regression, holdout


def report(
    tasks: dict[tuple[str, int], dict],
    cases: dict[tuple[str, int], str],
    v1: dict[tuple[str, int], dict],
    v2: dict[tuple[str, int], dict],
) -> tuple[str, list[dict]]:
    missing = sorted(tasks.keys() - v2.keys())
    extra = sorted(v2.keys() - tasks.keys())
    if extra:
        raise ValueError(f"v2 has unknown IDs: {extra[:3]}")
    changes = []
    def run_settings(predictions: dict[tuple[str, int], dict]) -> str:
        if not predictions:
            return "无成功结果"
        first = next(iter(predictions.values()))
        temperature = first.get("temperature")
        return (
            f"model={first['model']}，prompt_sha256={first['prompt_sha256'][:12]}…，"
            f"temperature={'API 默认' if temperature is None else temperature}"
        )

    lines = [
        "# 数据审计 v2：400 条对比报告",
        "",
        f"- 输入：{len(tasks)} 条；v1 成功：{len(v1)}；v2 成功：{len(v2)}；v2 缺失：{len(missing)}。",
        f"- v1 配置：{run_settings(v1)}。",
        f"- v2 配置：{run_settings(v2)}。",
        "- 这是模型版本对比，不是准确率估计；13 条回归样本参与制定 v2，不能作独立验证。",
        "",
        "## 13 条回归样本",
        "",
    ]
    passed = 0
    for key, expected in sorted(cases.items()):
        actual = v2.get(key, {}).get("annotation", {}).get("decision", "缺失")
        passed += actual == expected
        lines.append(f"- `{key[0]}#{key[1]}`：期望 {expected}；v2 {actual}；{'通过' if actual == expected else '未通过'}。")
    lines.extend(["", f"回归通过：**{passed}/{len(cases)}**。", "", "## 各来源比较", ""])
    lines.append("| 来源 | 总数 | v1 保留/待复核/剔除 | v2 保留/待复核/剔除 | 去留变化 |")
    lines.append("| --- | ---: | --- | --- | ---: |")
    by_source: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in tasks:
        by_source[key[0]].append(key)
    for source, keys in sorted(by_source.items()):
        c1 = Counter(v1[key]["annotation"]["decision"] for key in keys if key in v1)
        c2 = Counter(v2[key]["annotation"]["decision"] for key in keys if key in v2)
        changed = sum(
            v1[key]["annotation"]["decision"] != v2[key]["annotation"]["decision"]
            for key in keys if key in v1 and key in v2
        )
        def counts(counter: Counter) -> str:
            return "/".join(str(counter[label]) for label in ("保留", "待复核", "剔除"))
        lines.append(f"| {source} | {len(keys)} | {counts(c1)} | {counts(c2)} | {changed} |")
    for key in sorted(tasks):
        if key not in v1 or key not in v2:
            continue
        old = v1[key]["annotation"]
        new = v2[key]["annotation"]
        if old["decision"] != new["decision"]:
            changes.append({
                "source_file": key[0],
                "source_row_idx": key[1],
                "text_sha256": text_hash(tasks[key]["data"]["text"]),
                "v1": old,
                "v2": new,
                "regression_expected": cases.get(key),
            })
    lines.extend(["", f"去留变化总数：**{len(changes)}**。逐条见同目录的差异 JSONL。", ""])
    if missing:
        lines.extend(["## 未完成 ID", ""])
        lines.extend(f"- `{source}#{row}`" for source, row in missing)
    return "\n".join(lines) + "\n", changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--tasks", type=Path, required=True)
    p_prepare.add_argument("--cases", type=Path, required=True)
    p_prepare.add_argument("--dev-sample", type=Path, required=True)
    p_prepare.add_argument("--regression-output", type=Path, required=True)
    p_prepare.add_argument("--holdout-output", type=Path, required=True)
    p_prepare.add_argument("--per-source", type=int, default=10)
    p_prepare.add_argument("--seed", type=int, default=20260924)
    p_report = sub.add_parser("report")
    p_report.add_argument("--tasks", type=Path, required=True)
    p_report.add_argument("--cases", type=Path, required=True)
    p_report.add_argument("--v1", type=Path, required=True)
    p_report.add_argument("--v2", type=Path, required=True)
    p_report.add_argument("--report-output", type=Path, required=True)
    p_report.add_argument("--changes-output", type=Path, required=True)
    args = parser.parse_args()
    tasks = load_tasks(args.tasks)
    cases = load_cases(args.cases)
    if args.command == "prepare":
        dev = read_jsonl(args.dev_sample)
        dev_keys = {sample_key(row["source_file"], row["source_row_idx"]) for row in dev}
        if len(dev_keys) != len(dev):
            raise ValueError("duplicate dev sample IDs")
        regression, holdout = prepare(tasks, cases, dev_keys, args.per_source, args.seed)
        args.regression_output.parent.mkdir(parents=True, exist_ok=True)
        args.holdout_output.parent.mkdir(parents=True, exist_ok=True)
        args.regression_output.write_text(json.dumps(regression, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.holdout_output.write_text(json.dumps(holdout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.regression_output.chmod(0o600)
        print(f"regression={len(regression)} holdout={len(holdout['ids'])} dev={len(dev_keys)}")
    else:
        v1 = load_predictions(args.v1, tasks)
        v2 = load_predictions(args.v2, tasks)
        markdown, changes = report(tasks, cases, v1, v2)
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.changes_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(markdown, encoding="utf-8")
        args.changes_output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in changes), encoding="utf-8")
        print(f"v2={len(v2)}/{len(tasks)} changed={len(changes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
