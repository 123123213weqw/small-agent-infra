#!/usr/bin/env python3
"""Convert existing review-payload JSONL into an immutable idle-wash queue."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace existing queue: {args.output}")
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    seen = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.source.open(encoding="utf-8") as src, args.output.open("x", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            item = json.loads(line)
            doc_id = item["document_id"]
            if doc_id in seen or not isinstance(item.get("review_payload"), str):
                raise ValueError("duplicate document_id or missing review_payload")
            seen.add(doc_id)
            job = {
                "document_id": doc_id,
                "prompt_sha256": prompt_hash,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": item["review_payload"]},
                ],
                "max_tokens": 256,
            }
            dst.write(json.dumps(job, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break
    print(json.dumps({"queue": str(args.output), "documents": count,
                      "prompt_sha256": prompt_hash}, ensure_ascii=False))


if __name__ == "__main__":
    main()
