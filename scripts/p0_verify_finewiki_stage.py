#!/usr/bin/env python3
"""Independent read-only integrity checks for each FineWiki audit pipeline stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


STAGES = {
    "raw_download": ("raw_manifest.json", "raw_download_verified_audit_only"),
    "rule_clean": ("clean_rule_v1/manifest.json", "finewiki_zh_rule_clean_audit_only"),
    "token_count": ("clean_rule_v1/token_count_manifest.json", "finewiki_zh_token_count_audit_only"),
    "quality_sample": ("audit/stratified_quality_probe_v1/manifest.json", "finewiki_zh_stratified_quality_probe_audit_only"),
    "empty_language_scan": ("audit/empty_language_annotations_v1.json", "finewiki_zh_empty_language_annotation_audit_only"),
    "empty_language_tokens": ("audit/empty_language_token_count_v1.json", "finewiki_zh_extraction_artifact_token_audit_only"),
    "wave1_overlap": ("audit/wave1_overlap_v1/manifest.json", "finewiki_zh_vs_wave1_exact_and_family_overlap_audit_only"),
    "unflagged_index": ("audit/unflagged_index_v1/manifest.json", "finewiki_zh_unflagged_index_audit_only"),
    "residual_scan": ("audit/residual_extraction_v1.json", "finewiki_zh_residual_extraction_audit_only"),
    "residual_overlay": ("audit/residual_flags_v1/manifest.json", "finewiki_zh_residual_flag_overlay_audit_only"),
    "ceval_fetch": ("audit/heldout_ceval_val_v1/manifest.json", "ceval_validation_frozen_for_decontamination_audit_only"),
    "ceval_exact": ("audit/ceval_validation_exact_screen_v1/manifest.json", "finewiki_priority_ceval_validation_exact_question_screen_only"),
    "ceval_short": ("audit/ceval_validation_short_screen_v1/manifest.json", "finewiki_priority_ceval_short_stem_plus_options_screen_only"),
    "full_near_dedup": ("audit/full_near_dedup_v1/manifest.json", "finewiki_priority_full_near_dedup_audit_only"),
    "title_triage": ("audit/near_hit_title_triage_v1/manifest.json", "finewiki_near_hit_title_aware_triage_audit_only"),
    "strict_near_policy": ("audit/strict_near_policy_v1/manifest.json", "finewiki_strict_near_policy_overlay_audit_only"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha: str, expected_bytes: int | None = None) -> None:
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"byte count mismatch: {path}")
    if sha256(path) != expected_sha:
        raise ValueError(f"SHA-256 mismatch: {path}")


def verify(stage: str, root: Path, reference_root: Path, tokenizer: Path) -> dict:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    relpath, expected_stage = STAGES[stage]
    manifest_path = root / relpath
    report = json.loads(manifest_path.read_text())
    if report.get("stage") != expected_stage or report.get("training_eligible", False):
        raise ValueError(f"unexpected stage or training approval: {manifest_path}")
    spec = root / "code/configs/finewiki_zh_2025_candidate_v1.json"
    clean = root / "clean_rule_v1/manifest.json"
    token_manifest = root / "clean_rule_v1/token_count_manifest.json"
    annotation = root / "audit/empty_language_annotations_v1.json"
    annotation_tokens = root / "audit/empty_language_token_count_v1.json"
    overlap = root / "audit/wave1_overlap_v1/manifest.json"
    index = root / "audit/unflagged_index_v1/manifest.json"
    residual = root / "audit/residual_extraction_v1.json"
    overlay = root / "audit/residual_flags_v1/manifest.json"
    ceval = root / "audit/heldout_ceval_val_v1/manifest.json"
    near = root / "audit/full_near_dedup_v1/manifest.json"
    triage = root / "audit/near_hit_title_triage_v1/manifest.json"
    links = {
        "spec_sha256": spec,
        "clean_manifest_sha256": clean,
        "token_manifest_sha256": token_manifest,
        "artifact_count_sha256": annotation_tokens,
        "prior_audit_sha256": annotation,
        "candidate_manifest_sha256": clean,
        "wave1_overlap_manifest_sha256": overlap,
        "index_manifest_sha256": index,
        "residual_report_sha256": residual,
        "overlay_manifest_sha256": overlay,
        "near_manifest_sha256": near,
        "triage_manifest_sha256": triage,
        "tokenizer_sha256": tokenizer,
    }
    if stage in ("wave1_overlap",):
        links["reference_manifest_sha256"] = reference_root / "manifest.json"
    elif stage in ("ceval_exact", "ceval_short"):
        links["reference_manifest_sha256"] = ceval
    if stage == "full_near_dedup":
        links["wave1_manifest_sha256"] = reference_root / "manifest.json"
    for key, path in links.items():
        if key in report:
            verify_file(path, report[key])
    if stage == "raw_download":
        cfg = json.loads(spec.read_text())
        by_name = {Path(x["path"]).name: x for x in cfg["files"]}
        if report["revision"] != cfg["revision"] or len(report["files"]) != len(by_name):
            raise ValueError("raw source spec mismatch")
        for item in report["files"]:
            if item["file"] not in by_name or item["sha256"] != by_name[item["file"]]["sha256"]:
                raise ValueError("raw LFS digest differs from pinned spec")
            verify_file(root / "raw" / item["file"], item["sha256"], item["bytes"])
    elif stage == "rule_clean":
        if report["counts"]["kept"] != sum(x["counts"]["kept"] for x in report["files"]):
            raise ValueError("clean kept count mismatch")
        for item in report["files"]:
            verify_file(root / "clean_rule_v1" / item["output_file"],
                        item["output_sha256"], item["output_bytes"])
    elif stage in ("unflagged_index", "residual_overlay", "ceval_fetch"):
        base = manifest_path.parent
        for item in report["files"]:
            verify_file(base / item["file"], item["sha256"], item["bytes"])
    elif stage == "quality_sample":
        verify_file(manifest_path.parent / "sample.input.jsonl", report["sample_sha256"])
    elif stage == "wave1_overlap":
        verify_file(manifest_path.parent / "hits.jsonl", report["hits_sha256"])
    elif stage in ("ceval_exact", "ceval_short"):
        verify_file(manifest_path.parent / "hits.jsonl", report["hits_sha256"])
        priority = json.loads(overlay.read_text())["counts"]
        if (report["screen_counts"]["documents_scanned"] != priority["priority_documents"] -
                priority["priority_wave1_overlap"]):
            raise ValueError("C-Eval screen did not cover all priority documents")
    elif stage == "full_near_dedup":
        verify_file(manifest_path.parent / "signature_index.parquet", report["signature_index_sha256"],
                    report["signature_index_bytes"])
        verify_file(manifest_path.parent / "near_hits.jsonl", report["near_hits_sha256"],
                    report["near_hits_bytes"])
        priority = json.loads(overlay.read_text())["counts"]
        if report["counts"]["finewiki_documents"] != priority["priority_documents"] - priority["priority_wave1_overlap"]:
            raise ValueError("near-dedup did not cover all priority documents")
    elif stage == "title_triage":
        verify_file(manifest_path.parent / "same_title_direct_cover_drops.jsonl", report["drop_overlay_sha256"])
        verify_file(manifest_path.parent / "different_title_review_pairs.jsonl", report["different_title_review_sha256"])
    elif stage == "strict_near_policy":
        verify_file(manifest_path.parent / "exclusion_overlay.jsonl", report["exclusion_overlay_sha256"])
        previous = json.loads(near.read_text())
        if (report["counts"]["remaining_documents"] + report["counts"]["excluded_documents"] !=
                previous["counts"]["finewiki_documents"] or
                report["counts"]["remaining_tokens"] + report["counts"]["excluded_tokens"] !=
                previous["selection_counts"]["selected_tokens"]):
            raise ValueError("strict near-policy conservation mismatch")
    elif stage == "token_count":
        if report["counts"]["documents"] != json.loads(clean.read_text())["counts"]["kept"]:
            raise ValueError("token count document mismatch")
    elif stage == "empty_language_tokens":
        totals = report["counts"]
        tokens = json.loads(token_manifest.read_text())["counts"]["tokens"]
        if totals["tokens:flagged"] + totals["tokens:unflagged"] != tokens:
            raise ValueError("artifact token partition mismatch")
    elif stage == "residual_scan":
        if report["counts"]["previously_unflagged"] != json.loads(annotation_tokens.read_text())["counts"]["documents:unflagged"]:
            raise ValueError("residual scan input mismatch")
    return {"stage": stage, "manifest_sha256": sha256(manifest_path), "verified": True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--reference-root", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(verify(args.stage, args.root, args.reference_root, args.tokenizer)))


if __name__ == "__main__":
    main()
