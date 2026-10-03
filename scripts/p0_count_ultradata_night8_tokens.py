#!/usr/bin/env python3
"""Count Qwen-kept UltraData candidates with the frozen student tokenizer.

This creates metadata, not an approved training list or token stream.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    with args.index.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            document_id = row["document_id"]
            if document_id in index:
                raise ValueError(f"duplicate document ID: {document_id}")
            index[document_id] = row

    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    by_source = Counter()
    by_status = Counter()
    docs_by_source = Counter()
    docs_by_status = Counter()
    seen = set()
    output = args.output_dir / "candidate_token_index.jsonl"
    with output.open("w", encoding="utf-8") as target:
        for band in ("heavy", "medium", "light"):
            for shard in range(8):
                path = args.inputs_dir / band / f"shard-{shard}.input.jsonl"
                with path.open(encoding="utf-8") as source:
                    for line in source:
                        payload = json.loads(line)
                        document_id = payload["document_id"]
                        if document_id not in index:
                            continue
                        if document_id in seen:
                            raise ValueError(f"input repeated: {document_id}")
                        seen.add(document_id)
                        row = index[document_id]
                        review = json.loads(payload["review_payload"])
                        text = review["document_text"]
                        ids = tokenizer.encode(text, add_special_tokens=False).ids
                        source_id = row["source_id"]
                        status = row["postfilter_status"]
                        by_source[source_id] += len(ids)
                        by_status[status] += len(ids)
                        docs_by_source[source_id] += 1
                        docs_by_status[status] += 1
                        target.write(json.dumps({
                            "document_id": document_id,
                            "source_id": source_id,
                            "postfilter_status": status,
                            "text_tokens": len(ids),
                        }, sort_keys=True) + "\n")
    if seen != set(index):
        raise ValueError(f"missing input records: {len(set(index) - seen)}")
    report = {
        "stage": "candidate_token_count_not_training_approved",
        "postfilter_index_sha256": sha256(args.index),
        "tokenizer_json_sha256": sha256(args.tokenizer_json),
        "candidate_token_index_sha256": sha256(output),
        "documents": len(seen),
        "text_tokens_excluding_eos": sum(by_source.values()),
        "tokens_by_source": dict(sorted(by_source.items())),
        "documents_by_source": dict(sorted(docs_by_source.items())),
        "tokens_by_postfilter_status": dict(sorted(by_status.items())),
        "documents_by_postfilter_status": dict(sorted(docs_by_status.items())),
        "training_eligible": False,
        "reason": "upstream source rights and quality/near-duplicate/decontamination gates pending",
    }
    (args.output_dir / "candidate_token_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
