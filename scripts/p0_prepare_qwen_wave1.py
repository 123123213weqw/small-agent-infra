#!/usr/bin/env python3
"""Build auditable, context-safe input shards for eight local Qwen scorers.

No label from this stage is approved as a training decision. The original
Parquet pool is immutable; long documents are recorded for later handling.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-output-tokens", type=int, default=384)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    if args.output_dir.exists() or args.shards < 1:
        raise ValueError("output already exists or invalid shard count")

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    manifest_path = args.pool / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_meta = source_manifest["unified_index"]
    index_path = args.pool / index_meta["file"]
    if sha256_file(index_path) != index_meta["sha256"]:
        raise ValueError("unified index checksum mismatch")
    index = pq.read_table(index_path, columns=["document_id", "split", "license_status", "candidate_relpath", "row_in_shard"]).to_pylist()
    unapproved = [r["document_id"] for r in index if r["split"] == args.split and r["license_status"] != "approved"]
    if unapproved:
        raise ValueError(f"{len(unapproved)} source documents lack approved license status")
    selected = {r["document_id"]: r for r in index if r["split"] == args.split}
    if not selected:
        raise ValueError(f"no documents for split {args.split!r}")

    prompt = args.prompt.read_text(encoding="utf-8").strip()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    eligible: list[dict] = []
    overlong: list[dict] = []
    seen: set[str] = set()
    domain_counts: Counter[str] = Counter()
    for shard in source_manifest["shards"]:
        path = args.pool / shard["domain"] / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"shard checksum mismatch: {path}")
        table = pq.read_table(path, columns=[
            "document_id", "source_id", "source_locator", "domain", "language", "text", "token_count"
        ])
        for offset, row in enumerate(table.to_pylist()):
            doc_id = row["document_id"]
            indexed = selected.get(doc_id)
            if indexed is None:
                continue
            relpath = f"{shard['domain']}/{shard['file']}"
            if indexed["candidate_relpath"] != relpath or indexed["row_in_shard"] != offset:
                raise ValueError(f"index/shard mismatch: {doc_id}")
            if doc_id in seen:
                raise ValueError(f"duplicate document_id: {doc_id}")
            seen.add(doc_id)
            envelope = {
                "document_id": doc_id,
                "source_id": row["source_id"],
                "source_locator": row["source_locator"],
                "current_bucket": row["domain"],
                "language": row["language"],
                "token_count": row["token_count"],
                "document_text": row["text"],
            }
            payload = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            encoded = tokenizer.apply_chat_template([
                {"role": "system", "content": prompt},
                {"role": "user", "content": payload},
            ], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            prompt_tokens = len(encoded["input_ids"])
            minimal = {
                "document_id": doc_id,
                "current_bucket": row["domain"],
                "token_count": row["token_count"],
                "prompt_tokens_est": prompt_tokens,
                "candidate_relpath": relpath,
                "row_in_shard": offset,
            }
            if prompt_tokens + args.max_output_tokens > args.max_model_len:
                overlong.append(minimal)
            else:
                eligible.append({**minimal, "text_chars": len(row["text"]), "review_payload": payload})
                domain_counts[row["domain"]] += 1
    if seen != selected.keys():
        raise ValueError(f"missing {len(selected.keys() - seen)} indexed documents")

    # Longest-first bin packing balances estimated prefill cost across replicas.
    shards: list[list[dict]] = [[] for _ in range(args.shards)]
    costs = [0] * args.shards
    for row in sorted(eligible, key=lambda x: (-x["prompt_tokens_est"], x["document_id"])):
        target = min(range(args.shards), key=lambda i: (costs[i], len(shards[i]), i))
        shards[target].append(row)
        costs[target] += row["prompt_tokens_est"] + args.max_output_tokens

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qwen-wave1-", dir=args.output_dir.parent) as temp:
        directory = Path(temp)
        shard_meta = []
        for i, rows in enumerate(shards):
            path = directory / f"shard-{i}.input.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            shard_meta.append({"file": path.name, "documents": len(rows),
                               "estimated_tokens_with_output_reservation": costs[i],
                               "sha256": sha256_file(path)})
        long_path = directory / "overlong.jsonl"
        with long_path.open("w", encoding="utf-8") as handle:
            for row in overlong:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        report = {
            "stage": "qwen_official_fp8_rating_inputs_not_training_approved",
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_index_sha256": index_meta["sha256"],
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "model_path": str(args.model_path),
            "split": args.split,
            "max_model_len": args.max_model_len,
            "max_output_tokens": args.max_output_tokens,
            "selected_documents": len(selected),
            "scoreable_documents": len(eligible),
            "overlong_documents": len(overlong),
            "overlong_sha256": sha256_file(long_path),
            "scoreable_by_domain": dict(sorted(domain_counts.items())),
            "shards": shard_meta,
        }
        (directory / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        directory.rename(args.output_dir)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
