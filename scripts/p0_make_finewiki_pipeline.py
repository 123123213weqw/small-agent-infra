#!/usr/bin/env python3
"""Generate a resumable, fail-closed FineWiki CPU audit pipeline config.

All commands run on V100. No paid APIs, GPU jobs, source deletion, packing,
training-registry mutation, or training approval are included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="Existing/named V100 candidate root")
    ap.add_argument("--reference-root", type=Path, required=True, help="Frozen Wave1 V100 root")
    ap.add_argument("--tokenizer", type=Path, required=True, help="V100 tokenizer.json")
    ap.add_argument("--python", type=Path, required=True, help="V100 Python with PyArrow/tokenizers")
    ap.add_argument("--output", type=Path, required=True, help="Local pipeline JSON")
    args = ap.parse_args()
    for path in (args.root, args.reference_root, args.tokenizer, args.python):
        if not path.is_absolute():
            raise ValueError(f"remote path must be absolute: {path}")

    root, wave, tok, py = map(str, (args.root, args.reference_root, args.tokenizer, args.python))
    code = f"{root}/code/scripts"
    spec = f"{root}/code/configs/finewiki_zh_2025_candidate_v1.json"

    def run_script(name: str, *argv: str) -> str:
        return " ".join(shlex.quote(x) for x in (py, f"{code}/{name}", *argv))

    def ref(path: str) -> dict:
        return {"host": "CPU_NODE", "path": path}

    stages = []

    def add(name: str, script: str, args_: list[str], output: str,
            dependencies: list[str], helpers: list[str] | None = None) -> None:
        sources = [f"{code}/{script}", f"{code}/p0_verify_finewiki_stage.py",
                   *(f"{code}/{helper}" for helper in (helpers or [])), *dependencies]
        # Deduplicate while preserving order; a changed code snapshot or parent
        # manifest invalidates the recorded stage state.
        sources = list(dict.fromkeys(sources))
        check = run_script("p0_verify_finewiki_stage.py", "--stage", name, "--root", root,
                           "--reference-root", wave, "--tokenizer", tok)
        stages.append({"name": name, "kind": "command", "host": "CPU_NODE",
                       "command": run_script(script, *args_),
                       "inputs": [ref(x) for x in sources],
                       "outputs": [ref(output)],
                       "checks": [{"host": "CPU_NODE", "command": check}]})

    clean = f"{root}/clean_rule_v1/manifest.json"
    token_count = f"{root}/clean_rule_v1/token_count_manifest.json"
    annotation = f"{root}/audit/empty_language_annotations_v1.json"
    annotation_tokens = f"{root}/audit/empty_language_token_count_v1.json"
    overlap = f"{root}/audit/wave1_overlap_v1/manifest.json"
    index = f"{root}/audit/unflagged_index_v1/manifest.json"
    residual = f"{root}/audit/residual_extraction_v1.json"
    overlay = f"{root}/audit/residual_flags_v1/manifest.json"
    ceval = f"{root}/audit/heldout_ceval_val_v1/manifest.json"
    near = f"{root}/audit/full_near_dedup_v1/manifest.json"
    triage = f"{root}/audit/near_hit_title_triage_v1/manifest.json"

    add("raw_download", "p0_fetch_finewiki_zh.py",
        ["--spec", spec, "--root", root, "--endpoint", "https://hf-mirror.com", "--workers", "3"],
        f"{root}/raw_manifest.json", [spec])
    add("rule_clean", "p0_clean_finewiki_zh.py", ["--spec", spec, "--root", root, "--batch-size", "256"],
        clean, [spec, f"{root}/raw_manifest.json"], ["p0_data_pipeline.py"])
    add("token_count", "p0_count_finewiki_tokens.py", ["--root", root, "--tokenizer", tok],
        token_count, [clean, tok], ["p0_data_pipeline.py"])
    add("quality_sample", "p0_sample_finewiki_quality.py",
        ["--root", root, "--output", f"{root}/audit/stratified_quality_probe_v1", "--per-stratum", "50"],
        f"{root}/audit/stratified_quality_probe_v1/manifest.json", [clean], ["p0_data_pipeline.py"])
    add("empty_language_scan", "p0_audit_finewiki_extraction.py",
        ["--root", root, "--output", annotation], annotation, [clean],
        ["p0_data_pipeline.py", "p0_sample_finewiki_quality.py"])
    add("empty_language_tokens", "p0_count_finewiki_artifact_tokens.py",
        ["--root", root, "--tokenizer", tok, "--output", annotation_tokens],
        annotation_tokens, [clean, token_count, annotation, tok],
        ["p0_data_pipeline.py", "p0_audit_finewiki_extraction.py"])
    add("wave1_overlap", "p0_overlap_finewiki_wave1.py",
        ["--candidate-root", root, "--reference-root", wave,
         "--output", f"{root}/audit/wave1_overlap_v1"],
        overlap, [clean, f"{wave}/manifest.json"], ["p0_data_pipeline.py"])
    add("unflagged_index", "p0_index_finewiki_unflagged.py",
        ["--root", root, "--tokenizer", tok, "--output-dir", f"{root}/audit/unflagged_index_v1"],
        index, [clean, annotation_tokens, overlap, tok],
        ["p0_data_pipeline.py", "p0_audit_finewiki_extraction.py"])
    add("residual_scan", "p0_audit_finewiki_residual.py",
        ["--root", root, "--output", residual], residual, [clean, annotation_tokens],
        ["p0_data_pipeline.py", "p0_audit_finewiki_extraction.py", "p0_sample_finewiki_quality.py"])
    add("residual_overlay", "p0_index_finewiki_residual_flags.py",
        ["--root", root, "--output-dir", f"{root}/audit/residual_flags_v1"],
        overlay, [clean, index, residual],
        ["p0_data_pipeline.py", "p0_audit_finewiki_extraction.py", "p0_audit_finewiki_residual.py"])
    add("ceval_fetch", "p0_fetch_ceval_validation.py",
        ["--output-dir", f"{root}/audit/heldout_ceval_val_v1", "--endpoint", "https://hf-mirror.com"],
        ceval, [])
    add("ceval_exact", "p0_screen_finewiki_ceval.py",
        ["--candidate-root", root, "--reference-root", f"{root}/audit/heldout_ceval_val_v1",
         "--output-dir", f"{root}/audit/ceval_validation_exact_screen_v1"],
        f"{root}/audit/ceval_validation_exact_screen_v1/manifest.json",
        [clean, index, overlay, ceval], ["p0_data_pipeline.py"])
    add("ceval_short", "p0_screen_finewiki_ceval_short.py",
        ["--candidate-root", root, "--reference-root", f"{root}/audit/heldout_ceval_val_v1",
         "--output-dir", f"{root}/audit/ceval_validation_short_screen_v1"],
        f"{root}/audit/ceval_validation_short_screen_v1/manifest.json",
        [clean, index, overlay, ceval],
        ["p0_data_pipeline.py", "p0_screen_finewiki_ceval.py", "p0_probe_finewiki_near_overlap.py"])
    add("full_near_dedup", "p0_near_dedup_finewiki_priority.py",
        ["--candidate-root", root, "--reference-root", wave,
         "--output-dir", f"{root}/audit/full_near_dedup_v1"],
        near, [clean, index, overlay, f"{wave}/manifest.json"],
        ["p0_data_pipeline.py", "p0_screen_finewiki_ceval.py", "p0_near_dedup_ultradata_night8.py"])
    add("title_triage", "p0_triage_finewiki_near_hits.py",
        ["--candidate-root", root, "--output-dir", f"{root}/audit/near_hit_title_triage_v1"],
        triage, [clean, near], ["p0_data_pipeline.py"])
    add("strict_near_policy", "p0_finalize_finewiki_near_policy.py",
        ["--candidate-root", root, "--output-dir", f"{root}/audit/strict_near_policy_v1"],
        f"{root}/audit/strict_near_policy_v1/manifest.json", [near, triage],
        ["p0_data_pipeline.py", "p0_triage_finewiki_near_hits.py"])

    config = {"version": 1, "description": "FineWiki zh CPU-only audit; outputs remain training_eligible=false",
              "source_revision": "8bd13e72e6a002407649b3e898535f42ceb1aeb9",
              "paid_stages": 0, "gpu_stages": 0, "stages": stages}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "stages": len(stages)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
