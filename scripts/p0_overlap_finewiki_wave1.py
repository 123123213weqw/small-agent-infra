#!/usr/bin/env python3
"""Audit FineWiki exact-content and URL-family overlap against frozen Wave1."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile

import p0_data_pipeline as core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import pyarrow.parquet as pq

    ref_manifest_path = args.reference_root / "manifest.json"
    ref_manifest = json.loads(ref_manifest_path.read_text())
    if ref_manifest.get("stage") != "consolidate_candidates":
        raise ValueError("unexpected Wave1 reference stage")
    by_hash = defaultdict(list)
    by_family = defaultdict(list)
    reference_rows = 0
    for shard in ref_manifest["shards"]:
        path = args.reference_root / shard["domain"] / shard["file"]
        if path.stat().st_size != shard["bytes"] or core.sha256_file(path) != shard["sha256"]:
            raise ValueError(f"reference checksum mismatch: {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=[
            "document_id", "source_id", "split", "normalized_sha256", "family_id",
        ]):
            for row in batch.to_pylist():
                brief = {k: row[k] for k in ("document_id", "source_id", "split")}
                by_hash[row["normalized_sha256"]].append(brief)
                by_family[row["family_id"]].append(brief)
                reference_rows += 1
    if reference_rows != ref_manifest["counts"]["retained_documents"]:
        raise ValueError("reference row count mismatch")

    clean_root = args.candidate_root / "clean_rule_v1"
    manifest_path = clean_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["stage"] != "finewiki_zh_rule_clean_audit_only" or manifest["training_eligible"]:
        raise ValueError("unexpected FineWiki candidate stage")
    counts = Counter()
    hits = []
    for item in manifest["files"]:
        path = clean_root / item["output_file"]
        if core.sha256_file(path) != item["output_sha256"]:
            raise ValueError(f"candidate checksum mismatch: {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=[
            "document_id", "split", "normalized_sha256", "family_id", "url", "chars",
        ]):
            for row in batch.to_pylist():
                counts["candidate_documents"] += 1
                content_hits = by_hash.get(row["normalized_sha256"], [])
                family_hits = by_family.get(row["family_id"], [])
                if content_hits:
                    counts["exact_content_hits"] += 1
                if family_hits:
                    counts["same_url_family_hits"] += 1
                if content_hits or family_hits:
                    counts["either_hit"] += 1
                    hits.append({
                        "candidate_document_id": row["document_id"], "candidate_split": row["split"],
                        "candidate_url": row["url"], "candidate_chars": row["chars"],
                        "exact_content": bool(content_hits), "same_url_family": bool(family_hits),
                        "reference_content_hits": content_hits, "reference_family_hits": family_hits,
                    })
    if counts["candidate_documents"] != manifest["counts"]["kept"]:
        raise ValueError("candidate row count mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finewiki-overlap-", dir=args.output.parent) as tmp:
        staging = Path(tmp) / args.output.name
        staging.mkdir()
        hits_path = staging / "hits.jsonl"
        with hits_path.open("w", encoding="utf-8") as handle:
            for hit in hits:
                handle.write(core.canonical_json(hit) + "\n")
        report = {
            "stage": "finewiki_zh_vs_wave1_exact_and_family_overlap_audit_only",
            "reference_manifest_sha256": core.sha256_file(ref_manifest_path),
            "candidate_manifest_sha256": core.sha256_file(manifest_path),
            "reference_documents": reference_rows, "counts": dict(counts),
            "hits_sha256": core.sha256_file(hits_path),
            "training_eligible": False,
            "limitations": ["only this Wave1 reference pool", "no near-duplicate check", "no benchmark decontamination"],
        }
        (staging / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        staging.rename(args.output)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
