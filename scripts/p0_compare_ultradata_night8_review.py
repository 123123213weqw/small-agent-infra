#!/usr/bin/env python3
"""Summarize blind DeepSeek review of the 200 Qwen-kept night8 documents."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


RESULT_FILES = ("deepseek_review200.results.jsonl", "retry7.results.jsonl", "retry4.results.jsonl")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", type=Path, required=True)
    args = ap.parse_args()
    root = args.directory
    input_path = root / "deepseek_review200.input.jsonl"
    manifest = json.loads((root / "deepseek_review200.input.manifest.json").read_text())
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != manifest["input_sha256"]:
        raise ValueError("blind review input hash mismatch")
    inputs = {row["document_id"]: row for row in read_jsonl(input_path)}
    selected = {row["document_id"]: row for row in manifest["items"]}
    if len(inputs) != 200 or set(inputs) != set(selected):
        raise ValueError("input/manifest identity mismatch")
    results, attempts = {}, Counter()
    for filename in RESULT_FILES:
        for row in read_jsonl(root / filename):
            doc_id, = json.loads(row["source_id"])
            if doc_id not in inputs:
                raise ValueError("result outside blind sample")
            attempts[row["status"]] += 1
            if row["status"] == "ok":
                if doc_id in results or row["annotation"]["document_id"] != doc_id:
                    raise ValueError("duplicate/mismatched valid result")
                results[doc_id] = row
    missing = sorted(set(inputs) - set(results))
    by_source_random = defaultdict(Counter)
    by_source_targeted = defaultdict(Counter)
    by_status = defaultdict(Counter)
    drop_reasons_random = Counter()
    drop_reasons_targeted = Counter()
    disagreements = []
    for doc_id, item in selected.items():
        if doc_id not in results:
            continue
        annotation = results[doc_id]["annotation"]
        decision = annotation["decision"]
        source = item["source_id"]
        targeted = item["selection_stratum"].endswith("_targeted_flag")
        (by_source_targeted if targeted else by_source_random)[source][decision] += 1
        by_status[item["postfilter_status"]][decision] += 1
        if decision == "drop":
            (drop_reasons_targeted if targeted else drop_reasons_random).update(annotation["reason_codes"])
            disagreements.append({
                "document_id": doc_id, "source_id": source,
                "selection_stratum": item["selection_stratum"],
                "postfilter_status": item["postfilter_status"],
                "quality_flags": item["quality_flags"],
                "qwen_quality": item["qwen_quality"],
                "qwen_confidence": item["qwen_confidence"],
                "deepseek_quality": annotation["quality"],
                "deepseek_confidence": annotation["confidence"],
                "deepseek_reason_codes": annotation["reason_codes"],
                "deepseek_evidence": annotation["evidence"],
            })
    convert = lambda d: {key: dict(sorted(value.items())) for key, value in sorted(d.items())}
    report = {
        "stage": "blind_second_model_review_not_ground_truth_or_training_approval",
        "selected": len(inputs), "deepseek_ok": len(results), "missing_ids": missing,
        "attempts": dict(attempts),
        "by_source_random": convert(by_source_random),
        "by_source_targeted": convert(by_source_targeted),
        "by_postfilter_status": convert(by_status),
        "drop_reasons_random": dict(drop_reasons_random.most_common()),
        "drop_reasons_targeted": dict(drop_reasons_targeted.most_common()),
        "disagreements": sorted(disagreements, key=lambda row: row["document_id"]),
    }
    (root / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 夜间 Qwen keep × DeepSeek 盲法复核（200 条）", "",
             "**DeepSeek 未见 Qwen 标签或规则标记；此结果不是人工真值，也不是训练批准。**", "",
             f"有效 {len(results)}/{len(inputs)}；成功前总失败尝试 {attempts['error']}。", "",
             "| 来源 | 随机 keep 中 DS keep | 随机 keep 中 DS drop | 定向标记中 DS keep | 定向标记中 DS drop |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for source in sorted(by_source_random):
        a, b = by_source_random[source], by_source_targeted[source]
        lines.append(f"| {source} | {a['keep']} | {a['drop']} | {b['keep']} | {b['drop']} |")
    lines.extend(["", "随机样本每来源仅 40 条，定向标记每来源 10 条；两类不能混为总体比例。", "",
                  "## DeepSeek 驳回 Qwen keep 的原因", "",
                  "随机样本：" + json.dumps(dict(drop_reasons_random.most_common()), ensure_ascii=False), "",
                  "定向标记：" + json.dumps(dict(drop_reasons_targeted.most_common()), ensure_ascii=False), ""])
    (root / "comparison.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "selected", "deepseek_ok", "attempts", "by_source_random", "by_source_targeted",
        "by_postfilter_status", "drop_reasons_random", "drop_reasons_targeted")},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
