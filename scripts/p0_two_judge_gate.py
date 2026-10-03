#!/usr/bin/env python3
"""Conservative quality gate: keep only documents passing both independent judges.

This writes audit decisions, not training data. Missing/invalid model responses fail closed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def strict_keep(a: dict) -> bool:
    return (
        a.get("decision") == "keep"
        and a.get("confidence", 0) >= 0.85
        and a.get("quality", 0) >= 3
        and a.get("completeness", 0) >= 4
        and a.get("educational_value", 0) >= 3
        and a.get("format_integrity", 0) >= 4
        and a.get("recommended_bucket") != "drop"
    )


def index_results(path: Path, expected_ids: set[str]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        # source_id is present even for schema-invalid attempts.
        source_id = row.get("source_id")
        try:
            document_id, = json.loads(source_id)
        except (TypeError, ValueError):
            raise ValueError(f"malformed source_id in {path}") from None
        if document_id not in expected_ids:
            raise ValueError(f"unknown document {document_id} in {path}")
        grouped[document_id].append(row)
    result = {}
    for document_id, attempts in grouped.items():
        valid = [row for row in attempts if row.get("status") == "ok"]
        # Never silently choose between conflicting valid attempts.
        if len(valid) > 1 and any(
            row.get("annotation") != valid[0].get("annotation") for row in valid[1:]
        ):
            raise ValueError(f"conflicting valid results for {document_id} in {path}")
        if valid:
            row = valid[-1]
            if row.get("annotation", {}).get("document_id") != document_id:
                raise ValueError(f"annotation ID mismatch for {document_id} in {path}")
            result[document_id] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--deepseek", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")

    docs = read_jsonl(args.input)
    input_by_id = {row["document_id"]: row for row in docs}
    if len(input_by_id) != len(docs):
        raise ValueError("duplicate input document_id")
    ids = set(input_by_id)
    deepseek = index_results(args.deepseek, ids)
    qwen = index_results(args.qwen, ids)

    records = []
    for document_id, source in input_by_id.items():
        d, q = deepseek.get(document_id), qwen.get(document_id)
        expected_hash = hashlib.sha256(source["review_payload"].encode("utf-8")).hexdigest()
        if d is None or q is None:
            reason = "missing_or_invalid_judgment"
        elif (d.get("prompt_sha256"), d.get("schema_sha256"), d.get("text_sha256")) != (
            q.get("prompt_sha256"), q.get("schema_sha256"), q.get("text_sha256")
        ) or d.get("text_sha256") != expected_hash:
            reason = "judge_input_or_rubric_mismatch"
        elif strict_keep(d["annotation"]) and strict_keep(q["annotation"]):
            if d["annotation"]["recommended_bucket"] == q["annotation"]["recommended_bucket"]:
                reason = "both_strict_keep"
            else:
                reason = "bucket_disagreement"
        elif strict_keep(d["annotation"]) != strict_keep(q["annotation"]):
            reason = "strict_disagreement"
        else:
            reason = "both_not_strict_keep"
        records.append({
            "document_id": document_id,
            "domain": source["current_bucket"],
            "target_bucket": d["annotation"]["recommended_bucket"] if reason == "both_strict_keep" else None,
            "route": "candidate_keep" if reason == "both_strict_keep" else "exclude_from_training",
            "reason": reason,
        })

    summary = {
        "policy": "two_judge_strict_intersection_and_bucket_agreement_v2",
        "note": "Sample-only quality gate; no ground-truth precision claim or bulk-corpus approval.",
        "total": len(records),
        "by_reason": dict(sorted(Counter(r["reason"] for r in records).items())),
        "by_domain_and_route": {
            domain: dict(sorted(Counter(r["route"] for r in records if r["domain"] == domain).items()))
            for domain in sorted({r["domain"] for r in records})
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
