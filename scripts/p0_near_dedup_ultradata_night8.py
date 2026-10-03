#!/usr/bin/env python3
"""Approximate LSH retrieval + exact shingle Jaccard for UltraData candidates.

Checks Qwen-kept unflagged documents against Wave1 and earlier candidate records.
This is not external benchmark decontamination or training approval.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import random
import re

import numpy as np
import pyarrow.parquet as pq


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[\u3400-\u4dbf\u4e00-\u9fff]|[^\s]")
PRIME = (1 << 31) - 1
NUM_PERM = 32
BANDS = 8
THRESHOLD = 0.82
SHINGLE_SIZE = 5
MAX_SHINGLES = 8192


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shingles(text: str) -> set[int]:
    tokens = TOKEN_PATTERN.findall(text.casefold())
    if not tokens:
        return set()
    n = SHINGLE_SIZE
    if len(tokens) < n:
        phrases = {"\x1f".join(tokens)}
    else:
        phrases = {"\x1f".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
    hashes = (
        int.from_bytes(hashlib.blake2b(p.encode("utf-8"), digest_size=8).digest(), "big")
        % PRIME
        for p in phrases
    )
    return set(heapq.nsmallest(MAX_SHINGLES, hashes))


class NearIndex:
    def __init__(self) -> None:
        rng = random.Random(20260926)
        self.a = np.asarray([rng.randrange(1, PRIME) for _ in range(NUM_PERM)], dtype=np.uint64)
        self.b = np.asarray([rng.randrange(PRIME) for _ in range(NUM_PERM)], dtype=np.uint64)
        self.buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
        self.representatives: dict[str, tuple[set[int], str]] = {}

    def signature(self, values: set[int]) -> tuple[int, ...]:
        if not values:
            return (PRIME,) * NUM_PERM
        array = np.fromiter(values, dtype=np.uint64, count=len(values))
        transformed = (array[:, None] * self.a[None, :] + self.b[None, :]) % PRIME
        return tuple(map(int, transformed.min(axis=0)))

    @staticmethod
    def keys(signature: tuple[int, ...]):
        rows = NUM_PERM // BANDS
        for band in range(BANDS):
            yield band, signature[band * rows : (band + 1) * rows]

    def match_or_add(self, document_id: str, text: str, group: str) -> tuple[str | None, str | None, float]:
        values = shingles(text)
        signature = self.signature(values)
        keys = tuple(self.keys(signature))
        possible = set()
        for key in keys:
            possible.update(self.buckets.get(key, ()))
        best = (None, None, 0.0)
        for other_id in sorted(possible):
            other_values, other_group = self.representatives[other_id]
            union = len(values | other_values)
            similarity = len(values & other_values) / union if union else 1.0
            if similarity > best[2]:
                best = (other_id, other_group, similarity)
        if best[2] >= THRESHOLD:
            return best
        self.representatives[document_id] = (values, group)
        for key in keys:
            self.buckets[key].append(document_id)
        return None, None, best[2]


def existing_rows(pool: Path):
    manifest_path = pool / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "consolidate_candidates":
        raise ValueError("unexpected existing pool stage")
    seen = 0
    for shard in manifest["shards"]:
        path = pool / shard["domain"] / shard["file"]
        if path.stat().st_size != shard["bytes"] or sha256(path) != shard["sha256"]:
            raise ValueError(f"existing shard integrity failure: {path}")
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["document_id", "split", "text"]):
            for row in batch.to_pylist():
                seen += 1
                yield row
    if seen != manifest["counts"]["retained_documents"]:
        raise ValueError("existing document count mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = {}
    with args.index.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["postfilter_status"] == "quality_candidate_license_pending":
                selected[row["document_id"]] = row["source_id"]
    near_index = NearIndex()
    existing_counts = Counter()
    for row in existing_rows(args.pool):
        group = f"wave1_{row['split']}"
        near_index.match_or_add("wave1:" + row["document_id"], row["text"], group)
        existing_counts[group] += 1

    findings = Counter()
    seen = set()
    result = args.output_dir / "candidate_near_dedup_index.jsonl"
    with result.open("w", encoding="utf-8") as out:
        for band in ("heavy", "medium", "light"):
            for shard in range(8):
                path = args.inputs_dir / band / f"shard-{shard}.input.jsonl"
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        document_id = row["document_id"]
                        if document_id not in selected:
                            continue
                        if document_id in seen:
                            raise ValueError(f"candidate input repeated: {document_id}")
                        seen.add(document_id)
                        text = json.loads(row["review_payload"])["document_text"]
                        match_id, match_group, similarity = near_index.match_or_add(
                            "candidate:" + document_id, text, "candidate"
                        )
                        disposition = "near_duplicate" if match_id else "lsh_no_near_match"
                        findings[(selected[document_id], disposition, match_group or "none")] += 1
                        out.write(json.dumps({
                            "document_id": document_id,
                            "source_id": selected[document_id],
                            "disposition": disposition,
                            "matched_id": match_id,
                            "matched_group": match_group,
                            "jaccard": round(similarity, 6),
                        }, sort_keys=True) + "\n")
    if seen != set(selected):
        raise ValueError(f"missing candidates: {len(set(selected) - seen)}")
    summary = {
        "stage": "approximate_near_dedup_audit_not_training_approved",
        "existing_manifest_sha256": sha256(args.pool / "manifest.json"),
        "candidate_index_sha256": sha256(args.index),
        "result_sha256": sha256(result),
        "algorithm": {
            "token_shingle_size": SHINGLE_SIZE,
            "max_shingles": MAX_SHINGLES,
            "minhash_permutations": NUM_PERM,
            "lsh_bands": BANDS,
            "jaccard_threshold": THRESHOLD,
            "approximate_retrieval_not_exhaustive": True,
        },
        "existing_documents_scanned": sum(existing_counts.values()),
        "existing_documents_by_split": dict(sorted(existing_counts.items())),
        "candidates_scanned": len(seen),
        "findings": [
            {"source_id": s, "disposition": d, "matched_group": g, "count": n}
            for (s, d, g), n in sorted(findings.items())
        ],
    }
    (args.output_dir / "candidate_near_dedup_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
