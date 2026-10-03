#!/usr/bin/env python3
"""CPU-only curation control plane. Does not download, call teachers or train."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from small_agent.curation.core import canonical, fingerprint, normalize_document
from small_agent.curation.store import Batch, check_release, file_sha, prepare, read_json, read_jsonl, write_json


def partition(input_path: Path, output: Path, maximum_documents: int, maximum_bytes: int) -> dict:
    """Streaming work partitions, NOT train/val/test splits and NOT dedup."""
    if output.exists() or maximum_documents < 1 or maximum_bytes < 1:
        raise ValueError("new output directory and positive partition limits required")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = file_sha(input_path)
    tmp = Path(tempfile.mkdtemp(dir=output.parent, prefix=".curation-partition-"))
    dest, parts, count, body_bytes, total, oversize = None, [], 0, 0, 0, 0
    try:
        with (tmp / "oversize.quarantine.jsonl").open("w", encoding="utf-8") as held:
            for raw in read_jsonl(input_path):
                doc = normalize_document(raw)
                size = len(doc["document_text"].encode("utf-8"))
                total += 1
                if size > maximum_bytes:
                    held.write(canonical(raw) + "\n")
                    oversize += 1
                    continue
                if dest is None or count >= maximum_documents or body_bytes + size > maximum_bytes:
                    if dest:
                        dest.close()
                        parts[-1].update(documents=count, body_bytes=body_bytes)
                    name = f"part-{len(parts):06d}.jsonl"
                    parts.append({"path": name})
                    dest = (tmp / name).open("w", encoding="utf-8")
                    count, body_bytes = 0, 0
                dest.write(canonical(raw) + "\n")
                count += 1
                body_bytes += size
        if dest:
            dest.close()
            dest = None
            parts[-1].update(documents=count, body_bytes=body_bytes)
        if file_sha(input_path) != source_hash:
            raise ValueError("input changed while partitioning")
        for item in parts:
            item["sha256"] = file_sha(tmp / item["path"])
        report = {"version": 3, "stage": "work_partitions_not_training_splits", "source_sha256": source_hash,
                  "documents": total, "partitioned_documents": total - oversize, "parts": parts,
                  "oversize_quarantined_documents": oversize,
                  "oversize_file_sha256": file_sha(tmp / "oversize.quarantine.jsonl"),
                  "maximum_documents": maximum_documents, "maximum_body_bytes": maximum_bytes,
                  "training_approved": False}
        write_json(tmp / "PARTITIONS.json", report)
        os.rename(tmp, output)
        return report
    except BaseException:
        if dest:
            dest.close()
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def replay_legacy(input_path: Path, result_paths: list[Path], output: Path, design_path: Path | None) -> dict:
    """Diagnostic only. Never manufacture v3 evidence from legacy explanations."""
    if output.exists():
        raise FileExistsError(output)
    docs = [normalize_document(r) for r in read_jsonl(input_path)]
    if len({d["document_id"] for d in docs}) != len(docs):
        raise ValueError("duplicate legacy sample IDs")
    labels, conflicts, attempts = {}, set(), Counter()
    for path in result_paths:
        for r in read_jsonl(path):
            attempts[r.get("status", "unknown")] += 1
            if r.get("status") != "ok":
                continue
            a = r.get("annotation", {})
            doc_id = r.get("document_id")
            if not doc_id and r.get("source_id"):
                doc_id = json.loads(r["source_id"])[0]
            if a.get("document_id") != doc_id:
                raise ValueError("legacy label ID mismatch")
            if doc_id in labels and labels[doc_id] != a:
                conflicts.add(doc_id)
            labels[doc_id] = a
    counts, lints, strata, examples = Counter(), Counter(), {}, []
    for doc in docs:
        a = labels.get(doc["document_id"])
        if not a or doc["document_id"] in conflicts:
            category = "missing_or_conflicting"
        elif not {"decision", "confidence", "quality", "completeness", "educational_value", "format_integrity", "recommended_bucket", "reason_codes"} <= a.keys():
            category = "missing_or_conflicting"
        elif a["decision"] != "keep":
            category = "model_drop_not_verified_truth"
        elif (a["confidence"] < .85 or a["quality"] < 3 or a["completeness"] < 4
              or a["educational_value"] < 3 or a["format_integrity"] < 4 or "uncertain" in a["reason_codes"]):
            category = "score_or_confidence_failed"
        elif a["recommended_bucket"] != doc["previous_bucket"]:
            category = "bucket_only_mismatch_not_quality_verdict"
        else:
            category = "legacy_strict_agreement_not_accuracy"
        counts[category] += 1
        s = strata.setdefault(doc["sampling_stratum"], Counter())
        s[category] += 1
        s["total"] += 1
        for flag in doc["lint_flags"]:
            lints[flag["code"]] += 1
        if doc["lint_flags"]:
            examples.append({"document_id": doc["document_id"], "legacy_category": category,
                             "flags": [{k: v for k, v in flag.items() if k not in ("quote",)} for flag in doc["lint_flags"]]})
    report = {"stage": "legacy_diagnostic_only", "documents": len(docs), "categories": dict(counts),
              "strata": strata, "lint_occurrences": dict(lints), "flagged_documents": len(examples),
              "hold_flagged_documents": sum(any(f["severity"] == "hold" for f in d["lint_flags"]) for d in docs),
              "flagged_ids_and_spans_no_body": examples, "attempt_statuses": dict(attempts),
              "input_sha256": file_sha(input_path), "results_sha256": {str(p): file_sha(p) for p in result_paths},
              "new_v3_valid_reviews": 0, "new_training_approved_documents": 0, "training_started": False,
              "limits": ["old output lacks v3 grounded quotations and protocol; no automatic migration",
                         "matching quote and deterministic lint are risk checks, not general factual certification",
                         "legacy counts do not measure precision, false rejection or corpus truth",
                         "token counts are carried metadata, not a new tokenizer count"]}
    if design_path:
        design = read_json(design_path)["strata"]
        if set(design) != set(strata):
            raise ValueError("sample design strata differ")
        if abs(sum(s["document_population_weight"] for s in design.values()) - 1) > 1e-9:
            raise ValueError("invalid population weights")
        report["weighted_legacy_strict_agreement"] = sum(
            strata[k]["legacy_strict_agreement_not_accuracy"] / strata[k]["total"] * v["document_population_weight"] for k, v in design.items())
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    split = sub.add_parser("partition", help="stream into bounded immutable work batches")
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--max-documents", type=int, default=1000)
    split.add_argument("--max-body-bytes", type=int, default=32 << 20)
    prep = sub.add_parser("prepare", help="freeze a new full-body batch; never overwrite")
    for k in ("input", "batch", "policy", "schema", "prompt", "judges"):
        prep.add_argument("--" + k, type=Path, required=True)
    for action in ("status", "jobs", "import", "export"):
        s = sub.add_parser(action)
        s.add_argument("--batch", type=Path, required=True)
        if action in ("jobs", "import"):
            s.add_argument("--judge", required=True)
        if action in ("jobs", "export"):
            s.add_argument("--output", type=Path, required=True)
        if action == "import":
            s.add_argument("--results", type=Path, required=True)
    replay = sub.add_parser("replay-legacy")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--results", type=Path, nargs="+", required=True)
    replay.add_argument("--sample-design", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    release = sub.add_parser("check-release")
    release.add_argument("--batch", type=Path, required=True)
    release.add_argument("--snapshot", type=Path, required=True)
    release.add_argument("--checks", type=Path, required=True)
    args = p.parse_args()
    if args.action == "partition":
        result = partition(args.input, args.output, args.max_documents, args.max_body_bytes)
    elif args.action == "prepare":
        result = prepare(args.input, args.batch, args.policy, args.schema, args.prompt, args.judges)
    elif args.action == "replay-legacy":
        result = replay_legacy(args.input, args.results, args.output, args.sample_design)
    elif args.action == "check-release":
        batch = Batch(args.batch)
        try:
            result = check_release(batch, args.snapshot, args.checks)
        finally:
            batch.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["release_checks_bound"] else 2
    else:
        batch = Batch(args.batch)
        try:
            if args.action == "status":
                result = batch.report()
            elif args.action == "import":
                result = batch.import_results(args.judge, args.results)
            elif args.action == "export":
                result = batch.export(args.output)
            else:
                pending = batch.pending(args.judge)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as f:
                    for row in pending:
                        f.write(canonical(row) + "\n")
                result = {"pending_jobs": len(pending), "output_sha256": file_sha(args.output),
                          "api_called": False, "judge_protocol": batch.registry["judges"][args.judge]}
        finally:
            batch.close()
    # Do not print full bodies, schema output, or keys.
    if args.action == "replay-legacy":
        result = {k: v for k, v in result.items() if k != "flagged_ids_and_spans_no_body"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
