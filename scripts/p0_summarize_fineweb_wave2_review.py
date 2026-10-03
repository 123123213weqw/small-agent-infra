#!/usr/bin/env python3
"""Summarize Wave2 DeepSeek audit without promoting model labels to truth."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(directory: Path) -> dict:
    report = json.loads((directory / "REPORT.json").read_text())
    input_path = directory / "deepseek_review.input.jsonl"
    if sha256_file(input_path) != report["outputs_sha256"][input_path.name]:
        raise ValueError("review input hash differs from frozen selection")
    selected = {}
    for line in input_path.open(encoding="utf-8"):
        row = json.loads(line)
        doc = row["document_id"]
        if doc in selected:
            raise ValueError("duplicate review input document")
        selected[doc] = row
    if len(selected) != report["total_review_rows"]:
        raise ValueError("review input count differs from selection report")
    valid = {}
    errors = Counter()
    invalid = Counter()
    use = Counter()
    result_rows = 0
    results = directory / "deepseek_full.results.jsonl"
    if results.exists():
        for line in results.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            result_rows += 1
            try:
                doc, = json.loads(row["source_id"])
            except (KeyError, TypeError, ValueError):
                invalid["bad_source_id"] += 1
                continue
            if doc not in selected:
                invalid["not_in_frozen_sample"] += 1
                continue
            usage = row.get("usage") or {}
            for field in ("prompt_tokens", "prompt_cache_hit_tokens",
                          "prompt_cache_miss_tokens", "completion_tokens"):
                use[field] += int(usage.get(field, 0))
            if row.get("status") != "ok":
                errors[str(row.get("error", "unknown"))[:120]] += 1
                continue
            annotation = row.get("annotation") or {}
            if (annotation.get("document_id") != doc
                    or annotation.get("decision") not in {"keep", "drop"}):
                invalid["bad_annotation"] += 1
                continue
            if doc in valid and valid[doc] != annotation:
                invalid["conflicting_valid_retry"] += 1
            valid[doc] = annotation

    group_docs = defaultdict(list)
    for doc, source in selected.items():
        group_docs[(source["sampling_design"], source["sampling_group"])].append(doc)
    per_group = {}
    weighted_kept_docs = 0.0
    weighted_kept_docs_upper = 0.0
    weighted_all_docs = 0.0
    weighted_kept_tokens = 0.0
    weighted_kept_tokens_upper = 0.0
    weighted_all_tokens = 0.0
    full_stratified = True
    for (design, name), docs in sorted(group_docs.items()):
        completed = [doc for doc in docs if doc in valid]
        kept = [doc for doc in completed if valid[doc]["decision"] == "keep"]
        dropped = [doc for doc in completed if valid[doc]["decision"] == "drop"]
        reason_counts = Counter(code for doc in dropped
                                for code in valid[doc].get("reason_codes", []))
        stats = {"sampled": len(docs), "valid": len(completed),
                 "pending": len(docs) - len(completed), "keep": len(kept),
                 "drop": len(dropped), "reason_codes_for_drops": dict(reason_counts)}
        if design == "stratified":
            population = report["stratum_population_rows"][name]
            stats["population_rows"] = population
            stats["population_tokens"] = report["stratum_population_tokens"][name]
            if len(completed) != len(docs):
                full_stratified = False
            weight = population / len(docs)
            unknown = [doc for doc in docs if doc not in valid]
            weighted_kept_docs += weight * len(kept)
            weighted_kept_docs_upper += weight * (len(kept) + len(unknown))
            weighted_all_docs += weight * len(docs)
            weighted_kept_tokens += weight * sum(
                selected[doc]["tokenizer_v1_tokens"] for doc in kept)
            weighted_kept_tokens_upper += weight * sum(
                selected[doc]["tokenizer_v1_tokens"] for doc in kept + unknown)
            weighted_all_tokens += weight * sum(
                selected[doc]["tokenizer_v1_tokens"] for doc in docs)
        per_group[design + ":" + name] = stats
    expected_strata = set(report["stratum_population_rows"])
    actual_strata = {name for design, name in group_docs if design == "stratified"}
    if actual_strata != expected_strata:
        raise ValueError("missing or extra stratified sample groups")

    # Conservative peak-time bookkeeping, not a billing receipt. Current
    # DeepSeek account balance checks in the capped runner are a separate guard.
    miss = use["prompt_cache_miss_tokens"]
    hit = use["prompt_cache_hit_tokens"]
    if miss == hit == 0:
        miss = use["prompt_tokens"]
    conservative_peak_usd = (miss * .30 + hit * .006
                             + use["completion_tokens"] * 1.20) / 1_000_000
    out = {"status": "model_audit_not_training_approval",
           "selection_report_sha256": sha256_file(directory / "REPORT.json"),
           "results_sha256": sha256_file(results) if results.exists() else None,
           "sampled": len(selected), "valid_unique": len(valid),
           "pending_unique": len(selected) - len(valid), "result_rows_including_retries": result_rows,
           "errors_by_message": dict(errors), "invalid_by_reason": dict(invalid),
           "token_usage_recorded": dict(use),
           "conservative_peak_logged_usd": round(conservative_peak_usd, 4),
           "per_group": per_group,
           "stratified_estimate_status": "complete" if full_stratified else "pending_coverage"}
    out["model_label_bounds_not_ground_truth"] = {
        "post_domain_population_rows": report["post_domain_rows"],
        "document_weighted_keep_lower_unresolved_drop": weighted_kept_docs / weighted_all_docs,
        "document_weighted_keep_upper_unresolved_keep": weighted_kept_docs_upper / weighted_all_docs,
        "sample_token_weighted_keep_lower_unresolved_drop": weighted_kept_tokens / weighted_all_tokens,
        "sample_token_weighted_keep_upper_unresolved_keep": weighted_kept_tokens_upper / weighted_all_tokens,
        "note": "Targeted-risk supplement excluded. Bounds handle missing model labels only; sampling error and model misclassification remain."}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.review_dir.resolve())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in
                      ("status", "sampled", "valid_unique", "pending_unique",
                       "conservative_peak_logged_usd", "stratified_estimate_status")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
