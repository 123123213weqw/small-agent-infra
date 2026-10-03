#!/usr/bin/env python3
"""Fail-closed provisional Wave1 screen and stratified blind review sample.

This is an audit stage, not a training manifest. It reuses completed Qwen ratings.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import pyarrow.parquet as pq
from tokenizers import Tokenizer


DOMAINS = ("code_other", "code_python", "code_shell", "general_en", "general_zh", "math_en")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def strict_keep(a: dict, domain: str) -> bool:
    return (
        a.get("decision") == "keep"
        and a.get("confidence", 0) >= 0.85
        and a.get("quality", 0) >= 3
        and a.get("completeness", 0) >= 4
        and a.get("educational_value", 0) >= 3
        and a.get("format_integrity", 0) >= 4
        and a.get("recommended_bucket") == domain
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--qwen-root", type=Path, required=True)
    p.add_argument("--eval-hits", type=Path, required=True)
    p.add_argument("--tokenizer-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    pool_manifest = json.loads((args.pool / "manifest.json").read_text(encoding="utf-8"))
    if pool_manifest["stage"] != "consolidate_candidates" or not pool_manifest["license_gate"]["training_eligible"]:
        raise ValueError("Wave1 source/license gate failed")
    input_manifest = json.loads((args.qwen_root / "input_bands6_v1/manifest.json").read_text(encoding="utf-8"))
    finished = json.loads((args.qwen_root / "FINISHED6.json").read_text(encoding="utf-8"))
    first_input_manifest = json.loads((args.qwen_root / "input_shards/manifest.json").read_text(encoding="utf-8"))
    if first_input_manifest["source_manifest_sha256"] != sha256(args.pool / "manifest.json"):
        raise ValueError("Qwen first input source manifest mismatch")
    if input_manifest["source_manifest_sha256"] != sha256(args.qwen_root / "input_shards/manifest.json"):
        raise ValueError("Qwen banded input source manifest mismatch")
    if finished["input_manifest_sha256"] != sha256(args.qwen_root / "input_bands6_v1/manifest.json"):
        raise ValueError("Qwen finished/input manifest mismatch")
    if finished["expected"] != input_manifest["total_documents"]:
        raise ValueError("Qwen completion/input counts disagree")
    if not 0 <= finished["ok"] <= finished["expected"]:
        raise ValueError("invalid Qwen completion counts")

    input_by_id = {}
    for band in input_manifest["bands"]:
        for shard in band["shards"]:
            path = args.qwen_root / "input_bands6_v1" / shard["file"]
            if sha256(path) != shard["sha256"]:
                raise ValueError(f"Qwen input shard checksum mismatch: {path}")
            for row in read_jsonl(path):
                doc_id = row["document_id"]
                if doc_id in input_by_id:
                    raise ValueError(f"duplicate Qwen input: {doc_id}")
                input_by_id[doc_id] = row
    if len(input_by_id) != finished["expected"]:
        raise ValueError("Qwen input count mismatch")
    qwen = {}
    for band in ("heavy", "medium", "light"):
        for shard in range(6):
            path = args.qwen_root / "results6" / band / f"shard-{shard}.results.jsonl"
            for row in read_jsonl(path):
                if row.get("status") != "ok":
                    continue
                a = row["annotation"]
                doc_id = a["document_id"]
                if doc_id not in input_by_id or doc_id in qwen:
                    raise ValueError(f"unknown/conflicting Qwen result: {doc_id}")
                expected_hash = hashlib.sha256(input_by_id[doc_id]["review_payload"].encode()).hexdigest()
                if row["text_sha256"] != expected_hash:
                    raise ValueError(f"Qwen review payload hash mismatch: {doc_id}")
                qwen[doc_id] = row
    if len(qwen) != finished["ok"]:
        raise ValueError("Qwen valid result count mismatch")
    contaminated = {row["document_id"] for row in read_jsonl(args.eval_hits) if row["corpus_group"] == "wave1_train"}
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    reasons = Counter()
    docs_by_domain = Counter()
    tokens_by_domain = Counter()
    accepted = []
    with (args.output_dir / "provisional_index.jsonl").open("w", encoding="utf-8") as out:
        seen = 0
        for shard in pool_manifest["shards"]:
            path = args.pool / shard["domain"] / shard["file"]
            if path.stat().st_size != shard["bytes"] or sha256(path) != shard["sha256"]:
                raise ValueError(f"Wave1 source shard checksum mismatch: {path}")
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=256, columns=[
                "document_id", "source_id", "family_id", "domain", "text",
                "normalized_sha256", "license_status", "split", "source_metadata_json",
            ]):
                for row in batch.to_pylist():
                    seen += 1
                    if row["split"] != "train":
                        continue
                    doc_id = row["document_id"]
                    domain = row["domain"]
                    if row["license_status"] != "approved":
                        raise ValueError(f"non-approved source in Wave1: {doc_id}")
                    score = qwen.get(doc_id)
                    if doc_id in contaminated:
                        reason = "known_eval_phrase_hit"
                    elif score is None:
                        reason = "qwen_missing_or_overlong"
                    elif not strict_keep(score["annotation"], domain):
                        reason = "qwen_not_strict_same_domain_keep"
                    else:
                        meta = json.loads(row["source_metadata_json"])
                        upstream_score = meta.get("score")
                        if domain in ("general_en", "math_en") and (
                            not isinstance(upstream_score, (int, float)) or upstream_score < 3.5
                        ):
                            reason = "upstream_score_below_3_5"
                        else:
                            reason = "provisional_review_required"
                    reasons[(domain, reason)] += 1
                    if reason != "provisional_review_required":
                        continue
                    ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
                    if tokenizer.decode(ids, skip_special_tokens=False) != row["text"]:
                        raise ValueError(f"tokenizer roundtrip failed: {doc_id}")
                    item = {
                        "document_id": doc_id,
                        "source_id": row["source_id"],
                        "family_id": row["family_id"],
                        "domain": domain,
                        "original_split": "train",
                        "normalized_sha256": row["normalized_sha256"],
                        "own_text_tokens": len(ids),
                        "source_score": json.loads(row["source_metadata_json"]).get("score"),
                        "qwen_quality": score["annotation"]["quality"],
                        "qwen_confidence": score["annotation"]["confidence"],
                        "status": "provisional_review_required",
                    }
                    out.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                    accepted.append(item)
                    docs_by_domain[domain] += 1
                    tokens_by_domain[domain] += len(ids)
        if seen != pool_manifest["counts"]["retained_documents"]:
            raise ValueError("Wave1 source doc count mismatch")

    rng = random.Random(20260926)
    sampled = []
    for domain in DOMAINS:
        pool = sorted((x for x in accepted if x["domain"] == domain), key=lambda x: (x["own_text_tokens"], x["document_id"]))
        if len(pool) < 24:
            raise ValueError(f"too few provisional candidates: {domain}")
        groups = (pool[:len(pool)//3], pool[len(pool)//3:2*len(pool)//3], pool[2*len(pool)//3:])
        for bin_index, group in enumerate(groups):
            sampled.extend((domain, bin_index, row) for row in rng.sample(group, 8))
    sample_by_id = {row["document_id"]: (domain, bin_index, row) for domain, bin_index, row in sampled}
    if len(sample_by_id) != 144:
        raise ValueError("sample collision")
    sample_order = sorted(sample_by_id)
    rng.shuffle(sample_order)
    with (args.output_dir / "deepseek_blind144.input.jsonl").open("w", encoding="utf-8") as out:
        for doc_id in sample_order:
            source = sample_by_id[doc_id][2]
            row = dict(input_by_id[doc_id])
            row.update({
                "source_id": source["source_id"],
                "raw_row": row["row_in_shard"],
                "source_path": row["candidate_relpath"],
                "upstream_score": source["source_score"],
            })
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "deepseek_blind144.manifest.json").write_text(json.dumps({
        "stage": "wave1_provisional_keep_blind_deepseek_calibration_not_training_approval",
        "seed": 20260926,
        "selection": "8 per domain per own-token tertile, 6 domains, 144 total",
        "input_sha256": sha256(args.output_dir / "deepseek_blind144.input.jsonl"),
        "items": [
            {"document_id": doc_id, "domain": sample_by_id[doc_id][0],
             "length_bin": sample_by_id[doc_id][1], "own_text_tokens": sample_by_id[doc_id][2]["own_text_tokens"]}
            for doc_id in sample_order
        ],
    }, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "stage": "wave1_provisional_quality_screen_not_training_approved",
        "wave1_manifest_sha256": sha256(args.pool / "manifest.json"),
        "qwen_finished_sha256": sha256(args.qwen_root / "FINISHED6.json"),
        "tokenizer_json_sha256": sha256(args.tokenizer_json),
        "eval_hits_sha256": sha256(args.eval_hits),
        "scanned_train_documents": first_input_manifest["selected_documents"],
        "provisional_documents": len(accepted),
        "provisional_own_text_tokens": sum(tokens_by_domain.values()),
        "documents_by_domain": dict(sorted(docs_by_domain.items())),
        "tokens_by_domain": dict(sorted(tokens_by_domain.items())),
        "reasons": [
            {"domain": domain, "reason": reason, "count": n}
            for (domain, reason), n in sorted(reasons.items())
        ],
        "index_sha256": sha256(args.output_dir / "provisional_index.jsonl"),
        "training_eligible": False,
        "reason": "blind calibration and formal decontamination not yet passed",
    }
    (args.output_dir / "screen_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
