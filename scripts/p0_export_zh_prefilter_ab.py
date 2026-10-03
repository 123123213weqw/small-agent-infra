#!/usr/bin/env python3
"""Export a frozen, license-gated Chinese Wave1 teacher-label evaluation table."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def strict_keep(a: dict, domain: str) -> bool:
    return (a.get("decision") == "keep" and a.get("confidence", 0) >= .85
            and a.get("quality", 0) >= 3 and a.get("completeness", 0) >= 4
            and a.get("educational_value", 0) >= 3 and a.get("format_integrity", 0) >= 4
            and a.get("recommended_bucket") == domain)


def excerpt(text: str) -> str:
    if len(text) <= 4200:
        return text
    middle = len(text) // 2
    return text[:1800] + "\n[...CENTER SAMPLE...]\n" + text[middle - 300:middle + 300] + "\n[...END SAMPLE...]\n" + text[-1800:]


def hostname(locator: str, family_id: str) -> str:
    for part in locator.split("|"):
        if part.startswith("url="):
            host = urlsplit(part[4:]).hostname
            if host:
                return host.lower().removeprefix("www.")
    return "unknown:" + family_id


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--qwen-root", type=Path, required=True)
    p.add_argument("--accepted-index", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    import pyarrow.parquet as pq
    manifest = json.loads((args.pool / "manifest.json").read_text())
    if manifest.get("stage") != "consolidate_candidates" or not manifest["license_gate"]["training_eligible"]:
        raise ValueError("unapproved candidate pool")
    accepted = {json.loads(line)["document_id"] for line in args.accepted_index.open() if line.strip()}
    qwen = {}
    for path in sorted((args.qwen_root / "results6").glob("*/*.results.jsonl")):
        for line in path.open():
            result = json.loads(line)
            if result.get("status") != "ok" or result["source"]["current_bucket"] != "general_zh":
                continue
            annotation = result["annotation"]
            doc_id = annotation["document_id"]
            if doc_id in qwen:
                raise ValueError("duplicate successful Qwen document")
            qwen[doc_id] = annotation
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "general_zh_teacher_labels.jsonl"
    counts = Counter()
    with output.open("w", encoding="utf-8") as handle:
        for shard in manifest["shards"]:
            if shard["domain"] != "general_zh":
                continue
            path = args.pool / shard["domain"] / shard["file"]
            if digest(path) != shard["sha256"]:
                raise ValueError("source shard hash mismatch")
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=[
                "document_id", "family_id", "source_locator", "license_status", "split", "text", "token_count"
            ]):
                for row in batch.to_pylist():
                    if row["split"] != "train":
                        continue
                    if row["license_status"] != "approved":
                        raise ValueError("unapproved source record")
                    annotation = qwen.get(row["document_id"])
                    if annotation is None:
                        counts["no_valid_qwen"] += 1
                        continue
                    kept = strict_keep(annotation, "general_zh")
                    record = {
                        "document_id": row["document_id"], "family_id": row["family_id"],
                        "host": hostname(row["source_locator"], row["family_id"]),
                        "excerpt": excerpt(row["text"]), "reference_tokens": row["token_count"],
                        "qwen_strict_keep": kept, "final_approved": row["document_id"] in accepted,
                        "reason_codes": annotation.get("reason_codes", []),
                    }
                    if record["final_approved"] and not kept:
                        raise ValueError("final-approved record failed Qwen strict gate")
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    counts["teacher_valid"] += 1
                    counts["qwen_strict_keep" if kept else "qwen_reject"] += 1
                    counts["final_approved"] += record["final_approved"]
    report = {"stage": "zh_cpu_prefilter_ab_teacher_export_not_new_training_data",
              "pool_manifest_sha256": digest(args.pool / "manifest.json"),
              "accepted_index_sha256": digest(args.accepted_index),
              "input_sha256": digest(output), "counts": dict(counts)}
    (args.output_dir / "export_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
