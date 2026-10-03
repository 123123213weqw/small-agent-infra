#!/usr/bin/env python3
"""Count own-tokenizer tokens with/without visible FineWiki extraction defects."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import p0_data_pipeline as core
from p0_audit_finewiki_extraction import EMPTY_LANGUAGE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    clean_root = args.root / "clean_rule_v1"
    clean_manifest_path = clean_root / "manifest.json"
    clean_manifest = json.loads(clean_manifest_path.read_text())
    token_manifest_path = clean_root / "token_count_manifest.json"
    token_manifest = json.loads(token_manifest_path.read_text())
    if (clean_manifest["stage"] != "finewiki_zh_rule_clean_audit_only" or
            token_manifest["clean_manifest_sha256"] != core.sha256_file(clean_manifest_path) or
            token_manifest["tokenizer_sha256"] != core.sha256_file(args.tokenizer)):
        raise ValueError("source or tokenizer manifest mismatch")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    counts = Counter()
    for file_report in clean_manifest["files"]:
        path = clean_root / file_report["output_file"]
        if core.sha256_file(path) != file_report["output_sha256"]:
            raise ValueError(f"clean shard checksum mismatch: {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["text", "split"]):
            texts = batch.column(0).to_pylist()
            splits = batch.column(1).to_pylist()
            encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
            for text, split, encoded in zip(texts, splits, encodings):
                flag = "flagged" if EMPTY_LANGUAGE.search(text) else "unflagged"
                n = len(encoded.ids)
                counts[f"documents:{flag}"] += 1
                counts[f"tokens:{flag}"] += n
                counts[f"tokens:{flag}:{split}"] += n
        print(f"counted {path.name}", flush=True)
    if (counts["tokens:flagged"] + counts["tokens:unflagged"] !=
            token_manifest["counts"]["tokens"]):
        raise ValueError("token total mismatch")
    if (counts["documents:flagged"] + counts["documents:unflagged"] !=
            clean_manifest["counts"]["kept"]):
        raise ValueError("document total mismatch")
    report = {
        "stage": "finewiki_zh_extraction_artifact_token_audit_only",
        "clean_manifest_sha256": core.sha256_file(clean_manifest_path),
        "token_manifest_sha256": core.sha256_file(token_manifest_path),
        "tokenizer_sha256": core.sha256_file(args.tokenizer),
        "artifact_regex": EMPTY_LANGUAGE.pattern,
        "counts": dict(counts),
        "training_eligible": False,
        "note": "Unflagged means only this specific regex did not match; it is not quality approval.",
    }
    core.atomic_write_text(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
