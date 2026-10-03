#!/usr/bin/env python3
"""Bounded, text-free-output near-overlap probe: FineWiki priority vs Wave1 zh.

This is a stratified audit probe, not a corpus-wide deduplication gate.
Source text is read from existing Parquet files and never copied to the output.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import tempfile

import p0_data_pipeline as core


def length_bin(chars: int) -> str:
    if chars < 2000:
        return "0500_1999"
    if chars < 6000:
        return "2000_5999"
    if chars < 12000:
        return "6000_11999"
    return "12000_plus"


def stratum(row: dict) -> str:
    return f"{length_bin(row['chars'])}:{'math' if row['has_math'] else 'no_math'}"


def stable_rank(document_id: str) -> int:
    return int.from_bytes(hashlib.blake2b(
        ("finewiki-near-probe-v1:" + document_id).encode(), digest_size=16
    ).digest(), "big")


def choose_sample(rows: list[dict], n_per_stratum: int) -> list[dict]:
    """Select deterministic hash-minimum records in every nonempty stratum."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(stratum(row), []).append(row)
    result = []
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda r: (stable_rank(r["document_id"]), r["document_id"]))
        result.extend(ordered[:n_per_stratum])
    return result


def load_priority_sample(root: Path, n_per_stratum: int) -> tuple[list[dict], dict]:
    import pyarrow.parquet as pq

    index_root = root / "audit/unflagged_index_v1"
    overlay_root = root / "audit/residual_flags_v1"
    index_path = index_root / "manifest.json"
    overlay_path = overlay_root / "manifest.json"
    index = json.loads(index_path.read_text())
    overlay = json.loads(overlay_path.read_text())
    if (index["stage"] != "finewiki_zh_unflagged_index_audit_only" or
            overlay["stage"] != "finewiki_zh_residual_flag_overlay_audit_only" or
            index["training_eligible"] or overlay["training_eligible"] or
            overlay["index_manifest_sha256"] != core.sha256_file(index_path)):
        raise ValueError("FineWiki index/overlay provenance mismatch")
    excluded = set()
    for item in overlay["files"]:
        path = overlay_root / item["file"]
        if core.sha256_file(path) != item["sha256"]:
            raise ValueError(f"residual overlay checksum mismatch: {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["document_id"]):
            excluded.update(batch.column(0).to_pylist())
    if len(excluded) != overlay["counts"]["residual_documents"]:
        raise ValueError("duplicate residual document IDs")
    counts = Counter()
    # A bounded heap per stratum avoids retaining the whole 230k-row index.
    heaps: dict[str, list[tuple[int, str, dict]]] = {}
    for item in index["files"]:
        path = index_root / item["file"]
        if core.sha256_file(path) != item["sha256"]:
            raise ValueError(f"unflagged index checksum mismatch: {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                counts["indexed"] += 1
                if row["document_id"] in excluded:
                    counts["residual_excluded"] += 1
                    continue
                if row["wave1_url_family_overlap"]:
                    counts["wave1_url_family_excluded"] += 1
                    continue
                counts["priority"] += 1
                key = stratum(row)
                counts[f"population:{key}"] += 1
                rank = stable_rank(row["document_id"])
                entry = (-rank, row["document_id"], row)
                heap = heaps.setdefault(key, [])
                if len(heap) < n_per_stratum:
                    heapq.heappush(heap, entry)
                elif rank < -heap[0][0]:
                    heapq.heapreplace(heap, entry)
    if (counts["indexed"] != index["counts"]["documents:unflagged"] or
            counts["residual_excluded"] != overlay["counts"]["residual_documents"] or
            counts["priority"] != overlay["counts"]["priority_documents"] -
            overlay["counts"]["priority_wave1_overlap"]):
        raise ValueError("priority selection count mismatch")
    sample = []
    for key in sorted(heaps):
        sample.extend(row for _, _, row in sorted(heaps[key], reverse=True))
    if len({row["document_id"] for row in sample}) != len(sample):
        raise ValueError("duplicate sample IDs")
    return sample, {
        "index_manifest_sha256": core.sha256_file(index_path),
        "overlay_manifest_sha256": core.sha256_file(overlay_path),
        "counts": dict(counts),
    }


def load_sample_text(root: Path, sample: list[dict]) -> dict[str, str]:
    import pyarrow.parquet as pq

    clean_root = root / "clean_rule_v1"
    clean = json.loads((clean_root / "manifest.json").read_text())
    if clean["stage"] != "finewiki_zh_rule_clean_audit_only" or clean["training_eligible"]:
        raise ValueError("unexpected FineWiki clean stage")
    by_file: dict[str, dict[int, dict]] = {}
    for row in sample:
        positions = by_file.setdefault(row["clean_file"], {})
        pos = row["row_in_clean_file"]
        if pos in positions:
            raise ValueError("duplicate sample source position")
        positions[pos] = row
    texts = {}
    for item in clean["files"]:
        name = item["output_file"]
        selected = by_file.pop(name, {})
        if not selected:
            continue
        path = clean_root / name
        if core.sha256_file(path) != item["output_sha256"]:
            raise ValueError(f"clean source checksum mismatch: {path}")
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["document_id", "text"]):
            for row in batch.to_pylist():
                pointer = selected.get(offset)
                if pointer:
                    if pointer["document_id"] != row["document_id"]:
                        raise ValueError("sample index pointer mismatch")
                    texts[row["document_id"]] = row["text"]
                offset += 1
        if offset != item["counts"]["kept"]:
            raise ValueError("clean source row count mismatch")
    if by_file or len(texts) != len(sample):
        raise ValueError("sample source text missing")
    return texts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-root", type=Path, required=True)
    ap.add_argument("--reference-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--per-stratum", type=int, default=25)
    args = ap.parse_args()
    if args.per_stratum < 1 or args.output_dir.exists():
        raise ValueError("per-stratum must be positive and output must not exist")
    import pyarrow.parquet as pq
    from p0_near_dedup_ultradata_night8 import NearIndex, shingles

    sample, provenance = load_priority_sample(args.candidate_root, args.per_stratum)
    texts = load_sample_text(args.candidate_root, sample)
    reference_manifest_path = args.reference_root / "manifest.json"
    reference = json.loads(reference_manifest_path.read_text())
    if reference["stage"] != "consolidate_candidates":
        raise ValueError("unexpected Wave1 reference stage")
    near = NearIndex()
    ref_counts = Counter()
    for item in reference["shards"]:
        if item["domain"] != "general_zh":
            continue
        path = args.reference_root / item["domain"] / item["file"]
        if path.stat().st_size != item["bytes"] or core.sha256_file(path) != item["sha256"]:
            raise ValueError(f"reference shard checksum mismatch: {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["document_id", "split", "text"]):
            for row in batch.to_pylist():
                near.match_or_add("wave1:" + row["document_id"], row["text"], "wave1_" + row["split"])
                ref_counts["documents"] += 1
                ref_counts[f"split:{row['split']}"] += 1
        print(f"reference scanned {ref_counts['documents']}", flush=True)
    expected_ref = sum(item["rows"] for item in reference["shards"] if item["domain"] == "general_zh")
    if ref_counts["documents"] != expected_ref:
        raise ValueError("reference document count mismatch")
    findings = []
    outcome = Counter()
    for row in sample:
        values = shingles(texts[row["document_id"]])
        signature = near.signature(values)
        possible = set()
        for key in near.keys(signature):
            possible.update(near.buckets.get(key, ()))
        best_id, best_group, best_score = None, None, 0.0
        for other_id in sorted(possible):
            other_values, other_group = near.representatives[other_id]
            union = len(values | other_values)
            score = len(values & other_values) / union if union else 1.0
            if score > best_score:
                best_id, best_group, best_score = other_id, other_group, score
        label = "near_overlap_082" if best_score >= 0.82 else "no_retrieved_near_overlap"
        outcome[f"{stratum(row)}:{label}"] += 1
        findings.append({
            "document_id": row["document_id"], "stratum": stratum(row),
            "token_count": row["token_count"], "split": row["split"],
            "disposition": label, "matched_wave1_id": best_id if label == "near_overlap_082" else None,
            "matched_wave1_split": best_group if label == "near_overlap_082" else None,
            "best_retrieved_jaccard": round(best_score, 6),
        })
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finewiki-near-probe-", dir=args.output_dir.parent) as tmp:
        stage = Path(tmp) / args.output_dir.name
        stage.mkdir()
        result_path = stage / "findings.jsonl"
        with result_path.open("w") as out:
            for row in findings:
                out.write(core.canonical_json(row) + "\n")
        report = {
            "stage": "finewiki_priority_vs_wave1_zh_near_overlap_probe_only",
            **provenance,
            "reference_manifest_sha256": core.sha256_file(reference_manifest_path),
            "reference_documents_scanned": ref_counts["documents"],
            "reference_split_counts": dict(ref_counts),
            "sample_documents": len(sample), "sample_per_stratum": args.per_stratum,
            "outcomes": dict(outcome),
            "findings_sha256": core.sha256_file(result_path),
            "algorithm": "Wave1 zh representatives: 5-token shingles, capped 8192, 32-permutation MinHash / 8-band LSH; exact shingle Jaccard on retrieved pairs; threshold 0.82",
            "limitations": ["stratified sample, not corpus-wide", "approximate retrieval may miss pairs",
                            "only frozen Wave1 general_zh, not all sources", "not benchmark decontamination"],
            "training_eligible": False,
        }
        (stage / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        stage.rename(args.output_dir)
    print(json.dumps({"complete": True, "sample_documents": len(sample), "outcomes": dict(outcome)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
