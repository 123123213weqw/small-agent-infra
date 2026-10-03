#!/usr/bin/env python3
"""Deterministically route frozen data-audit excerpts using existing scores.

This is a dry curation pipeline: it never changes the input corpus, makes API
calls, or emits a training-text file. All outputs are provisional IDs/signals.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


ROUTES = ("train", "repair", "drop", "source_check", "review")
DECISIONS = {"保留", "待复核", "剔除"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_from_task(task: dict[str, Any]) -> tuple[str, int]:
    data = task["data"]
    return str(data["source_file"]), int(data["source_row_idx"])


def key_from_prediction(prediction: dict[str, Any]) -> tuple[str, int]:
    source = prediction["source"]
    return str(source["data.source_file"]), int(source["data.source_row_idx"])


def public_id(key: tuple[str, int]) -> str:
    return f"{key[0]}#{key[1]}"


def load_tasks(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("tasks must be a JSON array")
    result = {}
    for task in tasks:
        key = key_from_task(task)
        text = task["data"].get("text")
        if key in result or not isinstance(text, str) or not text.strip():
            raise ValueError(f"duplicate ID or empty text: {public_id(key)}")
        result[key] = task
    return result


def load_predictions(path: Path, tasks: dict[tuple[str, int], dict[str, Any]]) -> tuple[dict, dict]:
    results = {}
    settings = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            prediction = json.loads(line)
            key = key_from_prediction(prediction)
            if key not in tasks:
                raise ValueError(f"{path.name}:{line_number}: unknown ID {public_id(key)}")
            if prediction.get("status") != "ok":
                raise ValueError(f"{path.name}:{line_number}: non-ok status {public_id(key)}")
            expected_id = json.dumps([key[0], key[1]], ensure_ascii=False, separators=(",", ":"))
            if prediction.get("source_id") != expected_id:
                raise ValueError(f"{path.name}:{line_number}: source ID mismatch {public_id(key)}")
            if prediction.get("text_sha256") != sha256_text(tasks[key]["data"]["text"]):
                raise ValueError(f"{path.name}:{line_number}: text SHA mismatch {public_id(key)}")
            decision = prediction.get("annotation", {}).get("decision")
            if decision not in DECISIONS:
                raise ValueError(f"{path.name}:{line_number}: invalid decision {public_id(key)}")
            if key in results:
                raise ValueError(f"{path.name}:{line_number}: duplicate score {public_id(key)}")
            results[key] = prediction
            settings.add((prediction["model"], prediction["prompt_sha256"],
                          prediction["schema_sha256"], prediction.get("temperature")))
    if set(results) != set(tasks):
        missing = set(tasks) - set(results)
        raise ValueError(f"{path.name}: missing {len(missing)} scores")
    if len(settings) != 1:
        raise ValueError(f"{path.name}: mixed scoring settings")
    model, prompt_hash, schema_hash, temperature = next(iter(settings))
    return results, {
        "model": model,
        "prompt_sha256": prompt_hash,
        "schema_sha256": schema_hash,
        "temperature": temperature,
    }


def load_policy(path: Path) -> tuple[dict[str, Any], list[tuple[str, re.Pattern[str]]]]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(policy.get("version"), str) or not policy["version"]:
        raise ValueError("policy version required")
    if not isinstance(policy.get("source_review_markers"), list):
        raise ValueError("source_review_markers must be a list")
    if not isinstance(policy.get("extraction_damage_patterns"), list):
        raise ValueError("extraction_damage_patterns must be a list")
    patterns = []
    for item in policy["extraction_damage_patterns"]:
        patterns.append((item["id"], re.compile(item["regex"], re.IGNORECASE)))
    return policy, patterns


def classify(
    key: tuple[str, int],
    task: dict[str, Any],
    v1: dict[str, Any],
    v2: dict[str, Any],
    policy: dict[str, Any],
    patterns: list[tuple[str, re.Pattern[str]]],
    duplicate_of: str | None,
    policy_hash: str,
) -> dict[str, Any]:
    data = task["data"]
    text = data["text"]
    a1, a2 = v1["annotation"], v2["annotation"]
    source_markers = [marker for marker in policy["source_review_markers"]
                      if marker.lower() in text.lower()]
    damage_flags = [name for name, pattern in patterns if pattern.search(text)]
    model_source_issue = "隐私或版权风险" in set(a1.get("issues", [])) | set(a2.get("issues", []))
    source_flags = [f"visible:{marker}" for marker in source_markers]
    if model_source_issue:
        source_flags.append("model_flag:隐私或版权风险")
    d1, d2 = a1["decision"], a2["decision"]
    if duplicate_of is not None:
        route, rule = "drop", "exact_duplicate"
        evidence = f"与 {duplicate_of} 正文 SHA-256 相同"
    elif source_flags:
        route, rule = "source_check", "explicit_source_signal"
        evidence = "；".join(source_flags)
    elif damage_flags:
        route, rule = "repair", "visible_extraction_damage"
        evidence = "；".join(damage_flags)
    elif d1 == d2 == "剔除":
        route, rule = "drop", "both_scorers_reject"
        evidence = "v1 与 v2 均判剔除；仅是待核验的丢弃候选"
    elif d1 == d2 == "保留":
        route, rule = "train", "both_scorers_keep"
        evidence = "v1 与 v2 均判保留；仅是训练候选，未做来源许可确认"
    else:
        route, rule = "review", "scorer_disagreement_or_abstention"
        evidence = f"v1={d1}；v2={d2}"
    return {
        "id": public_id(key),
        "source_file": key[0],
        "source_row_idx": key[1],
        "source_dataset": data.get("source_dataset"),
        "text_sha256": sha256_text(text),
        "extraction": {"status": "damaged" if damage_flags else "no_high_precision_damage_detected",
                       "flags": damage_flags},
        "content": {"v1_decision": d1, "v2_decision": d2,
                    "status": "useful_candidate" if d1 == d2 == "保留" else
                              "low_candidate" if d1 == d2 == "剔除" else "disputed"},
        "provenance": {"status": "explicit_signal" if source_flags else "unknown",
                       "flags": source_flags},
        "route": route,
        "rule": rule,
        "evidence": evidence,
        "duplicate_of": duplicate_of,
        "pipeline_version": policy["version"],
        "policy_sha256": policy_hash,
    }


def build_records(
    tasks: dict[tuple[str, int], dict[str, Any]],
    v1: dict[tuple[str, int], dict[str, Any]],
    v2: dict[tuple[str, int], dict[str, Any]],
    policy: dict[str, Any],
    patterns: list[tuple[str, re.Pattern[str]]],
) -> list[dict[str, Any]]:
    policy_hash = sha256_text(canonical_json(policy))
    canonical_by_text = {}
    records = []
    for key in sorted(tasks):
        text_digest = sha256_text(tasks[key]["data"]["text"])
        duplicate_of = canonical_by_text.get(text_digest)
        if duplicate_of is None:
            canonical_by_text[text_digest] = public_id(key)
        records.append(classify(key, tasks[key], v1[key], v2[key], policy, patterns,
                                duplicate_of, policy_hash))
    return records


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def as_jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(canonical_json(record) + "\n" for record in records)


def build_report(records: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    routes = Counter(record["route"] for record in records)
    sources = sorted({record["source_file"] for record in records})
    by_source = defaultdict(Counter)
    for record in records:
        by_source[record["source_file"]][record["route"]] += 1
    lines = [
        "# 400 条语料离线分流：v0",
        "",
        "**这是候选清单，不是最终训练集。** 两次评分来自同一个模型体系，达成一致不等于正确；",
        "`train` 仅代表内容候选，尚未完成来源许可核对。输入是摘录，不代表完整原文。",
        "",
        f"- 输入：{len(records)} 条；策略：`{manifest['policy_version']}`。",
        "- 本轮没有调用模型、没有修改原始数据或 Label Studio。",
        "- `drop` 仅写 ID 清单，不删除原始记录；`review` 是评分分歧，不是提取修复。",
        "",
        "| 来源 | train 候选 | repair 提取 | drop 候选 | source_check 来源 | review 分歧 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in sources:
        counts = by_source[source]
        lines.append("| " + source + " | " + " | ".join(str(counts[route]) for route in ROUTES) + " |")
    lines.extend(["", "## 总数", ""])
    for route in ROUTES:
        lines.append(f"- `{route}`：{routes[route]}")
    lines.extend(["", "## 优先检查", ""])
    for route in ("source_check", "repair", "review"):
        ids = [record["id"] for record in records if record["route"] == route]
        lines.append(f"- `{route}` 前 10 条：" + ("、".join(f"`{item}`" for item in ids[:10]) if ids else "无"))
    lines.extend(["", "## 可复现性", "",
                  f"- 输入哈希、模型设置和策略哈希见 `manifest.json`。",
                  "- 输出按来源文件和原始行号排序；不包含当前运行时间或随机数。", ""])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_tasks(args.tasks)
    v1, v1_settings = load_predictions(args.v1, tasks)
    v2, v2_settings = load_predictions(args.v2, tasks)
    policy, patterns = load_policy(args.policy)
    records = build_records(tasks, v1, v2, policy, patterns)
    manifest = {
        "pipeline_version": "curation-pipeline-v0",
        "policy_version": policy["version"],
        "policy_sha256": sha256_text(canonical_json(policy)),
        "input_sha256": {name: sha256_file(path) for name, path in
                         (("tasks", args.tasks), ("v1", args.v1), ("v2", args.v2), ("policy", args.policy))},
        "scoring_settings": {"v1": v1_settings, "v2": v2_settings},
        "records": len(records),
        "routes": dict(sorted(Counter(record["route"] for record in records).items())),
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(args.run_dir / "records.jsonl", as_jsonl(records))
    for route in ROUTES:
        ids = [{"id": record["id"], "source_file": record["source_file"],
                "source_row_idx": record["source_row_idx"], "text_sha256": record["text_sha256"]}
               for record in records if record["route"] == route]
        atomic_write(args.run_dir / f"{route}_ids.jsonl", as_jsonl(ids))
    atomic_write(args.run_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    atomic_write(args.run_dir / "report.md", build_report(records, manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = run(args)
    print(f"records={manifest['records']} routes={canonical_json(manifest['routes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
