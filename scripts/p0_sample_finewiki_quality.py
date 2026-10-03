#!/usr/bin/env python3
"""Frozen, length × math stratified FineWiki audit sample (not training approval)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import tempfile

import p0_data_pipeline as core


SEED = "finewiki-zh-quality-probe-v1-20260926"


def stratum(chars: int, has_math: bool) -> str:
    if chars < 500:
        raise ValueError("sample input is not rule-cleaned")
    length = ("500-1999" if chars < 2000 else
              "2000-5999" if chars < 6000 else
              "6000-11999" if chars < 12000 else "12000+")
    return f"{length}/math={int(has_math)}"


def rank(document_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED}\0{document_id}".encode()).digest(), "big")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=50)
    args = parser.parse_args()
    if args.per_stratum < 1 or args.output.exists():
        raise ValueError("invalid sample size or output exists")
    import pyarrow.parquet as pq

    clean_root = args.root / "clean_rule_v1"
    manifest_path = clean_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["stage"] != "finewiki_zh_rule_clean_audit_only" or manifest["training_eligible"]:
        raise ValueError("unexpected clean manifest")
    heaps = defaultdict(list)
    population = Counter()
    total = 0
    for item in manifest["files"]:
        path = clean_root / item["output_file"]
        if core.sha256_file(path) != item["output_sha256"]:
            raise ValueError(f"clean file checksum mismatch: {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=[
            "document_id", "source_locator", "title", "url", "page_id", "version",
            "split", "has_math", "chars", "text", "normalized_sha256",
        ]):
            for row in batch.to_pylist():
                key = stratum(row["chars"], row["has_math"])
                population[key] += 1
                total += 1
                score = rank(row["document_id"])
                # Keep the smallest deterministic hash ranks in each stratum.
                heap = heaps[key]
                entry = (-score, row["document_id"], row)
                if len(heap) < args.per_stratum:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)
    if total != manifest["counts"]["kept"]:
        raise ValueError(f"sample population count {total} != clean manifest")
    if any(len(heap) != args.per_stratum for heap in heaps.values()) or len(heaps) != 8:
        raise ValueError(f"sample stratum shortfall: {[(k, len(v)) for k, v in heaps.items()]}")

    selected = []
    for key, heap in heaps.items():
        for _, _, row in heap:
            envelope = {
                "document_id": row["document_id"], "source_id": "finewiki_zh",
                "source_locator": row["source_locator"], "current_bucket": "general_zh",
                "language": "zh", "title": row["title"], "document_text": row["text"],
            }
            selected.append({
                "document_id": row["document_id"], "source_id": "finewiki_zh",
                "stratum": key, "population": population[key], "sample_size": args.per_stratum,
                "split": row["split"], "title": row["title"], "url": row["url"],
                "page_id": row["page_id"], "version": row["version"],
                "has_math": row["has_math"], "chars": row["chars"],
                "normalized_sha256": row["normalized_sha256"],
                "review_payload": core.canonical_json(envelope),
            })
    selected.sort(key=lambda x: (x["stratum"], rank(x["document_id"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finewiki-quality-", dir=args.output.parent) as tmp:
        staging = Path(tmp) / args.output.name
        staging.mkdir()
        sample_path = staging / "sample.input.jsonl"
        with sample_path.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(core.canonical_json(row) + "\n")
        report = {
            "stage": "finewiki_zh_stratified_quality_probe_audit_only",
            "seed": SEED, "clean_manifest_sha256": core.sha256_file(manifest_path),
            "sample_sha256": core.sha256_file(sample_path),
            "population": total, "strata_population": dict(sorted(population.items())),
            "per_stratum": args.per_stratum, "sampled": len(selected),
            "license_status": "pending", "training_eligible": False,
            "note": "Long documents may exceed quality-model context; no truncation or quality approval.",
        }
        (staging / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        staging.rename(args.output)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
