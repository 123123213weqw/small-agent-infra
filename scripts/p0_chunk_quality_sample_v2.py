#!/usr/bin/env python3
"""Exhaustively chunk a frozen bilingual-quality comparison sample for local Qwen.

Every source character is covered; no long document is silently omitted.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--sample-manifest", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--chunk-tokens", type=int, default=2400)
    ap.add_argument("--model-context", type=int, default=4096)
    ap.add_argument("--max-output", type=int, default=384)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.sample_manifest.read_text())
    if manifest["stage"] != "finewiki_strict_vs_wave1_zh_quality_sample_audit_only" or digest(args.sample) != manifest["sample_sha256"]:
        raise ValueError("sample/manifest mismatch")
    if args.chunk_tokens < 512 or args.chunk_tokens > 3000:
        raise ValueError("chunk-tokens outside safe range")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("fast tokenizer needed for exact character coverage")
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    rows = []
    coverage = []
    counts = Counter()
    with args.sample.open(encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            raw = doc["text"]
            if len(raw) != doc["chars"]:
                raise ValueError("sample char length mismatch")
            offsets = tokenizer(raw, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            if not offsets:
                raise ValueError("empty tokenized document")
            spans = []
            start = 0
            while start < len(offsets):
                end = min(start + args.chunk_tokens, len(offsets))
                a = 0 if start == 0 else offsets[start][0]
                b = len(raw) if end == len(offsets) else offsets[end][0]
                if b <= a:
                    raise ValueError("non-progressing tokenizer offsets")
                spans.append((a, b))
                start = end
            if spans[0][0] != 0 or spans[-1][1] != len(raw) or any(x[1] != y[0] for x, y in zip(spans, spans[1:])):
                raise ValueError("document has uncovered character span")
            counts[f"documents:{doc['source']}"] += 1
            for j, (a, b) in enumerate(spans):
                chunk_id = hashlib.sha256(f"{doc['document_id']}:{j}".encode()).hexdigest()
                payload = json.dumps({
                    "document_id": chunk_id, "current_bucket": "general_zh",
                    "document_text": raw[a:b],
                }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                encoded = tokenizer.apply_chat_template([
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": payload},
                ], tokenize=True, add_generation_prompt=True, enable_thinking=False)
                n = len(encoded if isinstance(encoded, list) else encoded["input_ids"])
                if n + args.max_output > args.model_context:
                    raise ValueError(f"chunk does not fit: {doc['document_id']}:{j} prompt={n}")
                rows.append({
                    "document_id": chunk_id, "parent_document_id": doc["document_id"],
                    "source": doc["source"], "stratum": doc["stratum"],
                    "chunk_index": j, "chunk_count": len(spans),
                    "char_start": a, "char_end": b, "prompt_tokens_est": n,
                    "review_payload": payload,
                })
                counts[f"chunks:{doc['source']}"] += 1
            coverage.append({
                "document_id": doc["document_id"], "source": doc["source"],
                "stratum": doc["stratum"], "chars": len(raw), "chunks": len(spans),
                "sample_weight": doc["population_in_stratum"] / doc["sample_size_in_stratum"],
            })
    if counts["documents:finewiki_zh"] != manifest["finewiki_sample"] or counts["documents:wave1_general_zh"] != manifest["wave1_sample"]:
        raise ValueError("document sample mismatch")
    args.output.mkdir(parents=True)
    path = args.output / "chunks.input.jsonl"
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    cover = args.output / "coverage.jsonl"
    with cover.open("x", encoding="utf-8") as f:
        for row in coverage:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    result = {
        "stage": "full_coverage_quality_chunk_input_audit_only",
        "sample_manifest_sha256": digest(args.sample_manifest),
        "sample_sha256": digest(args.sample), "prompt_sha256": digest(args.prompt),
        "chunks_sha256": digest(path), "coverage_sha256": digest(cover),
        "model": str(args.model), "chunk_tokens": args.chunk_tokens,
        "model_context": args.model_context, "max_output": args.max_output,
        "counts": dict(counts), "training_eligible": False,
        "limitations": "Full character coverage; segment-level judgments do not by themselves establish global document coherence.",
    }
    (args.output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
