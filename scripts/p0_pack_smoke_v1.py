#!/usr/bin/env python3
"""Encode frozen P0 smoke or approved-real documents into split-safe streams."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack(source: Path, tokenizer_dir: Path, output: Path, sequence_length: int) -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    if output.exists():
        raise FileExistsError(output)
    source_manifest = json.loads((source / "manifest.json").read_text())
    tokenizer_manifest = json.loads((tokenizer_dir / "manifest.json").read_text())
    source_stage = source_manifest.get("stage")
    if source_stage not in {"p0_1m_smoke_frozen", "accepted_real"}:
        raise ValueError("unexpected source stage")
    if (
        source_stage == "accepted_real"
        and not source_manifest.get("license_gate", {}).get("training_eligible")
    ):
        raise ValueError("accepted_real source is not training eligible")
    if tokenizer_manifest.get("stage") != "frozen_tokenizer":
        raise ValueError("tokenizer_v1 is not frozen")
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if sha256(tokenizer_path) != tokenizer_manifest["artifacts"]["tokenizer.json"]["sha256"]:
        raise ValueError("tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer_manifest["special_token_ids"]["<eos>"]
    vocab_size = tokenizer_manifest["vocab_size"]
    if not 0 <= eos < vocab_size <= 65536:
        raise ValueError("uint16 stream cannot represent tokenizer vocabulary")
    rows = []
    for shard in source_manifest["shards"]:
        path = source / shard["domain"] / shard["file"]
        if path.stat().st_size != shard["bytes"] or sha256(path) != shard["sha256"]:
            raise ValueError(f"source shard integrity failure: {path}")
        rows.extend(pq.read_table(path, columns=["document_id", "split", "sample_rank", "text"]).to_pylist())
    expected_documents = source_manifest["counts"][
        "accepted_documents" if source_stage == "accepted_real" else "documents"
    ]
    if len(rows) != expected_documents:
        raise ValueError("source document count mismatch")
    rows.sort(key=lambda row: (row["sample_rank"], row["document_id"]))
    streams = {split: [] for split in ("train", "validation", "test")}
    spans = {split: [] for split in streams}
    for row in rows:
        split = row["split"]
        if split not in streams:
            raise ValueError(f"unknown split: {split}")
        ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
        if tokenizer.decode(ids, skip_special_tokens=False) != row["text"]:
            raise ValueError(f"roundtrip failed: {row['document_id']}")
        if any(i == eos or i >= vocab_size for i in ids):
            raise ValueError(f"EOS collision or token out of range: {row['document_id']}")
        start = len(streams[split])
        streams[split].extend(ids)
        streams[split].append(eos)
        spans[split].append({"document_id": row["document_id"], "start": start, "end": len(streams[split])})
    if (
        source_stage == "p0_1m_smoke_frozen"
        and {split: len(items) for split, items in spans.items()} != source_manifest["splits"]
    ):
        raise ValueError("split document counts changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.tmp-", dir=output.parent) as tmp_name:
        tmp = Path(tmp_name)
        split_info = {}
        for split, tokens in streams.items():
            path = tmp / f"{split}.npy"
            np.save(path, np.asarray(tokens, dtype=np.uint16), allow_pickle=False)
            split_info[split] = {
                "documents": len(spans[split]), "tokens_including_eos": len(tokens),
                "eos_tokens": len(spans[split]), "full_2048_blocks": len(tokens) // sequence_length,
                "tail_tokens": len(tokens) % sequence_length,
                "file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
            }
        manifest = {
            "version": "p0_pack_smoke_v1",
            "stage": (
                "approved_real_token_stream"
                if source_stage == "accepted_real"
                else "internal_smoke_token_stream"
            ),
            "sequence_length": sequence_length, "dtype": "uint16",
            "source_manifest_sha256": sha256(source / "manifest.json"),
            "tokenizer_manifest_sha256": sha256(tokenizer_dir / "manifest.json"),
            "tokenizer_json_sha256": sha256(tokenizer_path),
            "vocab_size": vocab_size, "eos_token_id": eos,
            "license_gate": source_manifest["license_gate"], "splits": split_info,
            "document_spans": spans,
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        tmp.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    args = parser.parse_args()
    if args.sequence_length < 2:
        parser.error("sequence length must be at least 2")
    result = pack(args.source, args.tokenizer, args.output, args.sequence_length)
    print(json.dumps({"stage": result["stage"], "splits": result["splits"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
