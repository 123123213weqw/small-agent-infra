#!/usr/bin/env python3
"""Extract completed Qwen annotations for a frozen Wave1 blind sample."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-root", type=Path, required=True)
    p.add_argument("--sample-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sample = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
    ids = {row["document_id"] for row in sample["items"]}
    if len(ids) != len(sample["items"]):
        raise ValueError("sample IDs repeated")
    matches = {}
    for band in ("heavy", "medium", "light"):
        for shard in range(6):
            path = args.qwen_root / "results6" / band / f"shard-{shard}.results.jsonl"
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("status") != "ok":
                        continue
                    doc_id = row["annotation"]["document_id"]
                    if doc_id not in ids:
                        continue
                    if doc_id in matches:
                        raise ValueError(f"Qwen sample result repeated: {doc_id}")
                    matches[doc_id] = {
                        "document_id": doc_id,
                        "annotation": row["annotation"],
                        "prompt_sha256": row["prompt_sha256"],
                        "schema_sha256": row["schema_sha256"],
                        "text_sha256": row["text_sha256"],
                    }
    if set(matches) != ids:
        raise ValueError(f"missing Qwen sample results: {len(ids - set(matches))}")
    with args.output.open("x", encoding="utf-8") as out:
        for doc_id in sorted(ids):
            out.write(json.dumps(matches[doc_id], ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"documents": len(ids), "sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
