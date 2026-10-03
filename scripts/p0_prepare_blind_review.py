#!/usr/bin/env python3
"""Prepare blind Label Studio tasks for paired-judge disagreements and controls."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


DOMAINS = ("general_zh", "general_en", "math_en", "code_python", "code_shell", "code_other")


def strict_keep(annotation: dict) -> bool:
    return (
        annotation.get("decision") == "keep"
        and annotation.get("confidence", 0) >= 0.85
        and annotation.get("quality", 0) >= 3
        and annotation.get("completeness", 0) >= 4
        and annotation.get("educational_value", 0) >= 3
        and annotation.get("format_integrity", 0) >= 4
        and annotation.get("recommended_bucket") != "drop"
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_valid(path: Path) -> dict[str, dict]:
    output = {}
    for row in read_jsonl(path):
        if row.get("status") == "ok":
            output[row["annotation"]["document_id"]] = row
    return output


def rank(seed: str, document_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{document_id}".encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--deepseek", type=Path, required=True)
    p.add_argument("--qwen", type=Path, required=True)
    p.add_argument("--tasks-output", type=Path, required=True)
    p.add_argument("--manifest-output", type=Path, required=True)
    p.add_argument("--seed", default="p0-500m-blind-review36-v1")
    args = p.parse_args()
    if args.tasks_output.exists() or args.manifest_output.exists():
        raise ValueError("output exists")

    inputs = {row["document_id"]: row for row in read_jsonl(args.input)}
    deepseek = latest_valid(args.deepseek)
    qwen = latest_valid(args.qwen)
    disagreements = []
    agreements = defaultdict(list)
    for document_id, source in inputs.items():
        d = deepseek.get(document_id)
        q = qwen.get(document_id)
        if not d or not q:
            continue
        deep_keep = strict_keep(d["annotation"])
        qwen_keep = strict_keep(q["annotation"])
        item = {
            "document_id": document_id,
            "domain": source["current_bucket"],
            "deepseek_strict": "keep" if deep_keep else "drop",
            "qwen_strict": "keep" if qwen_keep else "drop",
        }
        if deep_keep != qwen_keep:
            item["selection_type"] = "strict_disagreement"
            disagreements.append(item)
        else:
            item["selection_type"] = "agreement_control"
            agreements[source["current_bucket"]].append(item)
    if len(disagreements) != 30:
        raise ValueError(f"expected 30 strict disagreements, found {len(disagreements)}")
    selected = list(disagreements)
    for domain in DOMAINS:
        pool = sorted(agreements[domain], key=lambda x: rank(args.seed, x["document_id"]))
        if not pool:
            raise ValueError(f"no agreement control for {domain}")
        selected.append(pool[0])
    selected.sort(key=lambda x: rank(args.seed + ":order", x["document_id"]))

    tasks = []
    for item in selected:
        source = inputs[item["document_id"]]
        payload = json.loads(source["review_payload"])
        tasks.append({"data": {
            "document_id": item["document_id"],
            "domain": item["domain"],
            "text": payload["document_text"],
        }})
    args.tasks_output.parent.mkdir(parents=True, exist_ok=True)
    task_text = json.dumps(tasks, ensure_ascii=False, indent=2) + "\n"
    args.tasks_output.write_text(task_text, encoding="utf-8")
    report = {
        "stage": "blind_human_review_queue_not_quality_approved",
        "seed": args.seed,
        "task_count": len(tasks),
        "strict_disagreements": len(disagreements),
        "agreement_controls": len(DOMAINS),
        "tasks_sha256": hashlib.sha256(task_text.encode()).hexdigest(),
        "comparison_kept_separate_from_tasks": selected,
    }
    args.manifest_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "comparison_kept_separate_from_tasks"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
