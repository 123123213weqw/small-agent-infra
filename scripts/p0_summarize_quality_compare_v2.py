#!/usr/bin/env python3
"""Summarize a resumable FineWiki/Wave1 chunk-quality audit without promotion."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def records(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def strict_keep(a: dict) -> bool:
    return (a["decision"] == "keep" and a["recommended_bucket"] == "general_zh"
            and a["confidence"] >= 0.85 and a["quality"] >= 3
            and a["completeness"] >= 4 and a["educational_value"] >= 3
            and a["format_integrity"] >= 4 and not a["reason_codes"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    manifest_path = args.input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    chunks_path = args.input_dir / "chunks.input.jsonl"
    coverage_path = args.input_dir / "coverage.jsonl"
    if (digest(chunks_path) != manifest["chunks_sha256"] or
            digest(coverage_path) != manifest["coverage_sha256"]):
        raise ValueError("input hash mismatch")
    chunks = {r["document_id"]: r for r in records(chunks_path)}
    coverage = {r["document_id"]: r for r in records(coverage_path)}
    if len(chunks) != sum(v for k, v in manifest["counts"].items() if k.startswith("chunks:")):
        raise ValueError("duplicate chunk IDs")
    if len(coverage) != sum(v for k, v in manifest["counts"].items() if k.startswith("documents:")):
        raise ValueError("duplicate parent IDs")
    latest = {}
    status = Counter()
    for r in records(args.results):
        source_ids = json.loads(r["source_id"])
        if len(source_ids) != 1 or source_ids[0] not in chunks:
            raise ValueError("unknown chunk ID")
        cid = source_ids[0]
        status[r["status"]] += 1
        if r["status"] == "ok":
            if r["annotation"]["document_id"] != cid:
                raise ValueError("model document ID mismatch")
            latest[cid] = r
    docs = defaultdict(list)
    for cid, row in chunks.items():
        docs[row["parent_document_id"]].append((row["chunk_index"], cid))
    if set(docs) != set(coverage):
        raise ValueError("coverage parent mismatch")
    by = defaultdict(Counter)
    reason_codes = defaultdict(Counter)
    score_rows = []
    for parent, info in coverage.items():
        links = sorted(docs[parent])
        if [i for i, _ in links] != list(range(info["chunks"])):
            raise ValueError("chunk sequence mismatch")
        labels = [latest.get(cid) for _, cid in links]
        key = info["source"] + "/" + info["stratum"]
        by[key]["sampled_documents"] += 1
        by[key]["sampled_chunks"] += len(labels)
        by[key]["scored_chunks"] += sum(x is not None for x in labels)
        if any(x is None for x in labels):
            by[key]["incomplete_documents"] += 1
            decision = "unscored"
        else:
            anns = [x["annotation"] for x in labels]
            raw_keep = all(a["decision"] == "keep" for a in anns)
            strict = all(strict_keep(a) for a in anns)
            decision = "strict_keep" if strict else ("raw_keep_non_strict" if raw_keep else "drop")
            by[key]["complete_documents"] += 1
            by[key][decision] += 1
            for ann in anns:
                reason_codes[info["source"]].update(ann["reason_codes"])
        score_rows.append({
            "document_id": parent, "source": info["source"],
            "stratum": info["stratum"], "sample_weight": info["sample_weight"],
            "chunks": info["chunks"], "scored_chunks": sum(x is not None for x in labels),
            "decision": decision,
        })
    complete = len(latest) == len(chunks)
    weighted = {}
    if complete:
        for source in ("finewiki_zh", "wave1_general_zh"):
            source_rows = [r for r in score_rows if r["source"] == source]
            denom = sum(r["sample_weight"] for r in source_rows)
            weighted[source] = {
                "strict_keep_fraction": sum(r["sample_weight"] for r in source_rows if r["decision"] == "strict_keep") / denom,
                "raw_keep_fraction": sum(r["sample_weight"] for r in source_rows if r["decision"] in ("strict_keep", "raw_keep_non_strict")) / denom,
                "population_documents": int(denom),
            }
    report = {
        "stage": "finewiki_vs_wave1_chunk_quality_summary_audit_only",
        "input_manifest_sha256": digest(manifest_path), "results_sha256": digest(args.results),
        "sampled_documents": len(coverage), "required_chunks": len(chunks),
        "scored_chunks": len(latest), "complete": complete,
        "result_row_statuses": dict(status),
        "by_source_stratum": {k: dict(v) for k, v in sorted(by.items())},
        "reason_codes": {k: dict(v) for k, v in reason_codes.items()},
        "population_weighted_only_if_complete": weighted,
        "training_eligible": False,
        "limitations": ["Qwen labels are not ground truth", "segment decisions do not prove global coherence", "different source populations and long-document length distributions limit direct comparison"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with args.output.with_name("document_scores.jsonl").open("w", encoding="utf-8") as f:
        for row in score_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("by_source_stratum", "reason_codes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
