#!/usr/bin/env python3
"""Count actual tokenizer_v1 tokens in rule-cleaned FineWiki, audit only."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import p0_data_pipeline as core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    clean_root = args.root / "clean_rule_v1"
    manifest_path = clean_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["stage"] != "finewiki_zh_rule_clean_audit_only" or manifest["training_eligible"]:
        raise ValueError("unexpected clean manifest")
    tokenizer_sha = core.sha256_file(args.tokenizer)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    aggregate = Counter()
    reports = []
    for file_report in manifest["files"]:
        path = clean_root / file_report["output_file"]
        if core.sha256_file(path) != file_report["output_sha256"]:
            raise ValueError(f"clean file checksum mismatch: {path}")
        counts = Counter()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["text", "split"]):
            texts = batch.column(0).to_pylist()
            splits = batch.column(1).to_pylist()
            encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
            for split, encoding in zip(splits, encodings):
                length = len(encoding.ids)
                counts["documents"] += 1
                counts["tokens"] += length
                counts[f"tokens:{split}"] += length
                counts[f"documents:{split}"] += 1
        if counts["documents"] != file_report["counts"].get("kept", 0):
            raise ValueError(f"document count mismatch: {path}")
        reports.append({"file": path.name, "counts": dict(counts)})
        aggregate.update(counts)
        print(json.dumps(reports[-1], ensure_ascii=False), flush=True)

    report = {
        "stage": "finewiki_zh_token_count_audit_only",
        "clean_manifest_sha256": core.sha256_file(manifest_path),
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_sha256": tokenizer_sha,
        "counts": dict(aggregate),
        "files": reports,
        "license_status": "pending",
        "training_eligible": False,
    }
    core.atomic_write_text(clean_root / "token_count_manifest.json",
                           json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"complete": True, "counts": dict(aggregate)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
