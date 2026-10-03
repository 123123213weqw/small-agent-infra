#!/usr/bin/env python3
"""Summarize a paired wave-1 quality probe without approving training data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from p0_two_judge_gate import index_results, strict_keep


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--deepseek", type=Path, required=True)
    p.add_argument("--qwen", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    rows = read_jsonl(args.input)
    ids = {r["document_id"] for r in rows}
    if len(ids) != len(rows):
        raise ValueError("duplicate sample IDs")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    d = index_results(args.deepseek, ids)
    q = index_results(args.qwen, ids)
    decisions = []
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        doc_id = row["document_id"]
        a, b = d.get(doc_id), q.get(doc_id)
        if a is None or b is None:
            reason = "missing_or_invalid_judgment"
        elif (a.get("prompt_sha256"), a.get("schema_sha256"), a.get("text_sha256")) != (
            b.get("prompt_sha256"), b.get("schema_sha256"), b.get("text_sha256")
        ):
            reason = "judge_input_or_rubric_mismatch"
        elif strict_keep(a["annotation"]) and strict_keep(b["annotation"]):
            if a["annotation"]["recommended_bucket"] == b["annotation"]["recommended_bucket"]:
                reason = "both_strict_keep"
            else:
                reason = "bucket_disagreement"
        elif strict_keep(a["annotation"]) != strict_keep(b["annotation"]):
            reason = "strict_disagreement"
        else:
            reason = "both_not_strict_keep"
        item = {
            "document_id": doc_id,
            "domain": row["current_bucket"],
            "length_bin": row["length_bin"],
            "own_tokens": row["own_token_count"],
            "reason": reason,
            "target_bucket": a["annotation"]["recommended_bucket"] if reason == "both_strict_keep" else None,
        }
        decisions.append(item)
        by_stratum[f"{item['domain']}/length_{item['length_bin']}"].append(item)

    stratum_report = {}
    domain_estimated_kept = Counter()
    domain_population = Counter()
    for key, sampled in sorted(by_stratum.items()):
        population = manifest["strata"][key]
        n = len(sampled)
        weighted_kept = sum(x["own_tokens"] for x in sampled if x["reason"] == "both_strict_keep")
        estimated_kept = population["documents"] * weighted_kept / n
        domain = population["domain"]
        domain_estimated_kept[domain] += estimated_kept
        domain_population[domain] += population["own_tokens"]
        stratum_report[key] = {
            "sampled_documents": n,
            "sampled_own_tokens": sum(x["own_tokens"] for x in sampled),
            "kept_documents": sum(x["reason"] == "both_strict_keep" for x in sampled),
            "kept_own_tokens": weighted_kept,
            "population_documents": population["documents"],
            "population_own_tokens": population["own_tokens"],
            "diagnostic_estimated_kept_own_tokens": round(estimated_kept),
        }
    by_domain = {}
    for domain in sorted(domain_population):
        items = [x for x in decisions if x["domain"] == domain]
        sample_total = sum(x["own_tokens"] for x in items)
        sample_keep = sum(x["own_tokens"] for x in items if x["reason"] == "both_strict_keep")
        by_domain[domain] = {
            "sampled_documents": len(items),
            "sampled_own_tokens": sample_total,
            "sampled_kept_documents": sum(x["reason"] == "both_strict_keep" for x in items),
            "sampled_kept_own_tokens": sample_keep,
            "sampled_token_retention": round(sample_keep / sample_total, 4) if sample_total else None,
            "population_own_tokens_within_20k_chars": domain_population[domain],
            "diagnostic_estimated_token_retention_within_20k_chars": round(
                domain_estimated_kept[domain] / domain_population[domain], 4
            ),
            "overlong_unscored_own_tokens": manifest["overlong_train_rows_by_domain"].get(domain, {}).get("own_tokens", 0),
        }
    summary = {
        "stage": "small_stratified_probe_not_training_approved",
        "caution": "Five or fewer documents per stratum: diagnostic estimates have high variance. Overlong documents are excluded.",
        "sampled_documents": len(rows),
        "by_reason": dict(sorted(Counter(x["reason"] for x in decisions).items())),
        "by_domain": by_domain,
        "by_stratum": stratum_report,
        "usage": {
            "deepseek": dict(sum_usage(d.values())),
            "qwen": dict(sum_usage(q.values())),
        },
        "decisions": decisions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("decisions", "by_stratum")}, ensure_ascii=False))


def sum_usage(rows: object) -> Counter:
    counter = Counter()
    for row in rows:
        for key, value in row.get("usage", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                counter[key] += value
    return counter


if __name__ == "__main__":
    main()
