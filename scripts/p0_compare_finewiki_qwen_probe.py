#!/usr/bin/env python3
"""Verify Qwen quality-probe results; summarize only scored sample, no pool promotion."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    context = args.root / "input/context_split_v1"
    context_manifest_path = context / "manifest.json"
    context_manifest = json.loads(context_manifest_path.read_text())
    if context_manifest["stage"] != "finewiki_zh_qwen_context_split_audit_only":
        raise ValueError("unexpected context manifest")
    scoreable_path = context / "scoreable.input.jsonl"
    if sha256_file(scoreable_path) != context_manifest["scoreable_sha256"]:
        raise ValueError("scoreable input checksum mismatch")
    inputs = {}
    for line in scoreable_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["document_id"] in inputs:
            raise ValueError("duplicate input document ID")
        inputs[row["document_id"]] = row
    results_path = args.root / "results/qwen_v1.results.jsonl"
    results = defaultdict(list)
    for line in results_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        source_ids = json.loads(row["source_id"])
        if len(source_ids) != 1 or source_ids[0] not in inputs:
            raise ValueError("unknown result document ID")
        results[source_ids[0]].append(row)
    summary = Counter()
    by_stratum = defaultdict(Counter)
    reasons = Counter()
    for doc_id, source in inputs.items():
        rows = results.get(doc_id, [])
        good = [r for r in rows if r["status"] == "ok"]
        if len(good) > 1:
            raise ValueError(f"multiple valid results: {doc_id}")
        key = source["stratum"]
        by_stratum[key]["scoreable"] += 1
        if not good:
            summary["unscored"] += 1
            by_stratum[key]["unscored"] += 1
            continue
        row = good[0]
        annotation = row["annotation"]
        if annotation["document_id"] != doc_id:
            raise ValueError(f"result document ID mismatch: {doc_id}")
        decision = annotation["decision"]
        if decision not in ("keep", "drop"):
            raise ValueError(f"invalid decision: {decision}")
        summary["scored"] += 1
        summary[decision] += 1
        by_stratum[key][decision] += 1
        for reason in annotation["reason_codes"]:
            reasons[reason] += 1
    if sum(v for k, v in summary.items() if k in ("scored", "unscored")) != len(inputs):
        raise ValueError("result accounting mismatch")
    report = {
        "stage": "finewiki_zh_qwen_probe_summary_audit_only",
        "context_manifest_sha256": sha256_file(context_manifest_path),
        "results_sha256": sha256_file(results_path),
        "scoreable_documents": len(inputs),
        "summary": dict(summary),
        "by_stratum": {k: dict(v) for k, v in sorted(by_stratum.items())},
        "reason_codes": dict(reasons),
        "sample_overlong_documents": sum(v for k, v in context_manifest["counts"].items() if k.startswith("overlong:")),
        "training_eligible": False,
        "limitations": ["Qwen label is not ground truth", "overlong sample is unscored", "strata were sampled equally, not by population", "no per-record license approval"],
    }
    output = args.root / "results/summary_v1.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
