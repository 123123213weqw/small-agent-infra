#!/usr/bin/env python3
"""Compare blind DeepSeek/Qwen data-quality judgments, preserving sampling strata."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    args = p.parse_args()
    input_path = args.directory / "deepseek_review180.input.jsonl"
    manifest = json.loads((args.directory / "deepseek_review180.input.manifest.json").read_text())
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != manifest["input_sha256"]:
        raise ValueError("blind review input checksum mismatch")
    inputs = {x["document_id"]: x for x in read_jsonl(input_path)}
    selected = {x["document_id"]: x for x in manifest["items"]}
    if len(inputs) != len(selected) or set(inputs) != set(selected):
        raise ValueError("input and manifest identities differ")
    results = {}
    attempts = Counter()
    for row in read_jsonl(args.directory / "deepseek_review180.results.jsonl"):
        doc_id, = json.loads(row["source_id"])
        if doc_id not in inputs:
            raise ValueError("result ID not in blind sample")
        attempts[row["status"]] += 1
        if row["status"] == "ok":
            if row["annotation"]["document_id"] != doc_id:
                raise ValueError("DeepSeek annotation ID mismatch")
            results[doc_id] = row
    missing = sorted(set(inputs) - set(results))
    by_source = defaultdict(Counter)
    by_stratum = defaultdict(Counter)
    deepseek_reasons_on_qwen_keep_drop = Counter()
    disagreements = []
    for doc_id, row in inputs.items():
        if doc_id not in results:
            continue
        item = selected[doc_id]
        qwen = item["qwen_annotation"]
        deepseek = results[doc_id]["annotation"]
        source = row["source_id"]
        stratum = item["selection_stratum"]
        pair = qwen["decision"] + "→" + deepseek["decision"]
        by_source[source][pair] += 1
        by_stratum[stratum][pair] += 1
        if qwen["decision"] != deepseek["decision"]:
            if qwen["decision"] == "keep":
                deepseek_reasons_on_qwen_keep_drop.update(deepseek["reason_codes"])
            disagreements.append({
                "document_id": doc_id, "source_id": source, "stratum": stratum,
                "upstream_score": row["upstream_score"], "text_chars": row["text_chars"],
                "qwen_decision": qwen["decision"], "qwen_quality": qwen["quality"],
                "qwen_confidence": qwen["confidence"], "qwen_reason_codes": qwen["reason_codes"],
                "qwen_evidence": qwen["evidence"],
                "deepseek_decision": deepseek["decision"], "deepseek_quality": deepseek["quality"],
                "deepseek_confidence": deepseek["confidence"], "deepseek_reason_codes": deepseek["reason_codes"],
                "deepseek_evidence": deepseek["evidence"],
            })
    def converted(values):
        return {key: dict(sorted(counts.items())) for key, counts in sorted(values.items())}
    report = {
        "stage": "blind_dual_model_sample_comparison_not_training_approved",
        "selected": len(inputs), "deepseek_ok": len(results), "missing_ids": missing,
        "attempts": dict(attempts), "by_source": converted(by_source),
        "by_stratum": converted(by_stratum),
        "deepseek_reasons_on_qwen_keep_drop": dict(deepseek_reasons_on_qwen_keep_drop.most_common()),
        "disagreements": disagreements,
    }
    (args.directory / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 新来源 Qwen × DeepSeek 独立复核（180 条）", "",
        "**仅是分层审计样本，非训练批准。DeepSeek 未看到 Qwen 的答案。**", "",
        f"DeepSeek 成功 {len(results)}/{len(inputs)}；失败或缺失 {len(missing)}。", "",
        "| 来源 | 双方留 | Qwen 留/DeepSeek 丢 | Qwen 丢/DeepSeek 留 | 双方丢 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for source, counts in sorted(by_source.items()):
        lines.append(f"| {source} | {counts['keep→keep']} | {counts['keep→drop']} | {counts['drop→keep']} | {counts['drop→drop']} |")
    lines.extend(["", "## 分层结果", "", "| 分层 | 双方留 | Qwen 留/DeepSeek 丢 | Qwen 丢/DeepSeek 留 | 双方丢 |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for stratum, counts in sorted(by_stratum.items()):
        lines.append(f"| {stratum} | {counts['keep→keep']} | {counts['keep→drop']} | {counts['drop→keep']} | {counts['drop→drop']} |")
    lines.extend(["", "## Qwen 留、DeepSeek 丢：原因码", "",
                  json.dumps(dict(deepseek_reasons_on_qwen_keep_drop.most_common()), ensure_ascii=False), "",
                  "分层有意使 Qwen keep/drop 接近均衡；此表不能当作原始语料总体误判率。\n"])
    (args.directory / "comparison.zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("selected", "deepseek_ok", "attempts", "by_source", "by_stratum", "deepseek_reasons_on_qwen_keep_drop")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
