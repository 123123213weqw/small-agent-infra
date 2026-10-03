#!/usr/bin/env python3
"""Count a consolidated candidate pool with the frozen student tokenizer."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")

    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    manifest_path = args.pool / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "consolidate_candidates":
        raise ValueError("pool must be consolidated")
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    by_domain: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    documents_by_split: Counter[str] = Counter()
    documents = 0
    roundtrip_failures = 0
    sampled_roundtrips = Counter()
    for shard in manifest["shards"]:
        path = args.pool / shard["domain"] / shard["file"]
        if path.stat().st_size != shard["bytes"] or sha256_file(path) != shard["sha256"]:
            raise ValueError(f"candidate shard checksum mismatch: {path}")
        table = pq.read_table(path, columns=["document_id", "domain", "split", "text"])
        if table.num_rows != shard["rows"]:
            raise ValueError(f"candidate shard row count mismatch: {path}")
        for row in table.to_pylist():
            encoding = tokenizer.encode(row["text"], add_special_tokens=False)
            count = len(encoding.ids)
            by_domain[row["domain"]] += count
            by_split[row["split"]] += count
            documents_by_split[row["split"]] += 1
            documents += 1
            if sampled_roundtrips[row["domain"]] < 10:
                sampled_roundtrips[row["domain"]] += 1
                if tokenizer.decode(encoding.ids, skip_special_tokens=False) != row["text"]:
                    roundtrip_failures += 1
    if documents != manifest["counts"]["retained_documents"]:
        raise ValueError("candidate document count mismatch")
    report = {
        "stage": "candidate_own_tokenizer_count_not_quality_approved",
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "tokenizer_json_sha256": sha256_file(args.tokenizer_json),
        "documents": documents,
        "text_tokens_excluding_eos": sum(by_domain.values()),
        "tokens_by_domain": dict(sorted(by_domain.items())),
        "tokens_by_split": dict(sorted(by_split.items())),
        "documents_by_split": dict(sorted(documents_by_split.items())),
        "sampled_roundtrips": sum(sampled_roundtrips.values()),
        "sampled_roundtrip_failures": roundtrip_failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
