#!/usr/bin/env python3
"""Freeze a fail-closed go/no-go decision for the audited night8 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.report_dir
    inputs = {
        "postfilter": root / "summary.json",
        "token_count": root / "candidate_token_summary.json",
        "near_dedup": root / "candidate_near_dedup_summary.json",
        "eval_screen": root / "eval_screen_v2/eval_exact_phrase_summary.json",
        "web_source_mix": root / "web_source_summary.json",
    }
    data = {key: read(path) for key, path in inputs.items()}
    if data["postfilter"]["qwen_keep"] != data["token_count"]["documents"]:
        raise ValueError("Qwen/token count mismatch")
    candidates = data["token_count"]["documents_by_postfilter_status"]["quality_candidate_license_pending"]
    if candidates != data["near_dedup"]["candidates_scanned"]:
        raise ValueError("near dedup candidate count mismatch")
    if candidates != data["eval_screen"]["corpus_documents_scanned"]["new_candidate"]:
        raise ValueError("eval screen candidate count mismatch")
    if data["web_source_mix"]["total_mapped"] != 23388 + 13578:
        raise ValueError("web source mapping mismatch")

    near_hits = sum(
        x["count"] for x in data["near_dedup"]["findings"]
        if x["disposition"] == "near_duplicate"
    )
    decision = {
        "stage": "night8_go_no_go_v1",
        "decision": "NO_GO_FORMAL_PRETRAINING",
        "training_eligible": False,
        "train_manifest_created": False,
        "validation_manifest_created": False,
        "source_rights_gate": {
            "ultradata_code_l2_py": "pending_each_source_repository_license",
            "ultradata_math_l2_preview": "pending_seed_source_terms_and_traceability",
            "ultrafineweb_en": "pending_upstream_fineweb_terms_and_provenance_record",
            "ultrafineweb_zh": "pending_eight_subsource_terms",
        },
        "quality_gate": "not_passed_qwen_keep_has_substantial_blind_deepseek_disagreement",
        "near_dedup_gate": "audit_done_near_matches_must_be_excluded_before_any_release",
        "eval_decontamination_gate": "partial_screen_only_at_least_one_clear_wave1_arc_easy_test_overlap",
        "qwen_keep_documents": data["postfilter"]["qwen_keep"],
        "qwen_keep_text_tokens": data["token_count"]["text_tokens_excluding_eos"],
        "unflagged_exact_unique_documents": candidates,
        "unflagged_exact_unique_text_tokens": data["token_count"]["tokens_by_postfilter_status"]["quality_candidate_license_pending"],
        "near_duplicate_audit_hits": near_hits,
        "sources_sha256": {key: sha256(path) for key, path in inputs.items()},
        "release_rule": "No candidate document may enter a formal training manifest until source, quality, dedup and evaluation gates pass independently.",
    }
    output = root / "go_no_go_decision.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
