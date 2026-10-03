"""Streaming, atomic construction of the sharded token format."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any, Iterable

import numpy as np

from .sharded_format import FLAG_PADDED_TAIL, VERSION, sha256_file


@dataclass(frozen=True)
class TokenizedDocument:
    document_id: str
    split: str
    domain: str
    token_ids: list[int]


class _Writer:
    def __init__(self, directory: Path, split: str, domain: str, shard_id: int, sequence_length: int):
        self.directory = directory
        self.split = split
        self.domain = domain
        self.shard_id = shard_id
        self.sequence_length = sequence_length
        self.bin_name = f"{split}-{shard_id:05d}.bin"
        self.idx_name = f"{split}-{shard_id:05d}.idx"
        self.handle = (directory / self.bin_name).open("wb")
        self.tokens = 0
        self.documents = 0

    def append(self, ids: list[int], eos_token_id: int) -> None:
        values = np.asarray([*ids, eos_token_id], dtype="<u2")
        self.handle.write(values.tobytes())
        self.tokens += len(values)
        self.documents += 1

    def finish(self) -> dict[str, Any] | None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        stride = self.sequence_length - 1
        if self.split == "train":
            samples = max(0, 1 + (self.tokens - self.sequence_length) // stride)
        else:
            samples = max(0, (self.tokens - 1 + stride - 1) // stride)
        if samples < 1:
            (self.directory / self.bin_name).unlink()
            return None
        idx_path = self.directory / self.idx_name
        with idx_path.open("wb") as index:
            for sample_id in range(samples):
                offset = sample_id * stride
                valid = min(self.sequence_length, self.tokens - offset)
                index.write(struct.pack("<QII", offset, valid, FLAG_PADDED_TAIL if valid < self.sequence_length else 0))
            index.flush()
            os.fsync(index.fileno())
        bin_path = self.directory / self.bin_name
        return {
            "shard_id": self.shard_id,
            "split": self.split,
            "domain": self.domain,
            "bin": self.bin_name,
            "idx": self.idx_name,
            "tokens": self.tokens,
            "samples": samples,
            "documents": self.documents,
            "bin_bytes": bin_path.stat().st_size,
            "bin_sha256": sha256_file(bin_path),
            "idx_bytes": idx_path.stat().st_size,
            "idx_sha256": sha256_file(idx_path),
        }


def pack_sharded(
    documents: Iterable[TokenizedDocument],
    output: str | Path,
    *,
    sequence_length: int,
    target_shard_tokens: int,
    vocab_size: int,
    eos_token_id: int,
    tokenizer_sha256: str,
    mixture: dict[str, float],
    license_gate: dict[str, Any],
    source_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Write documents without accumulating a shard-sized Python list in RAM."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if sequence_length < 2 or target_shard_tokens < sequence_length:
        raise ValueError("invalid sequence or target shard length")
    if not 0 <= eos_token_id < vocab_size <= 65536:
        raise ValueError("uint16 vocabulary metadata is invalid")
    if not tokenizer_sha256 or not mixture or any(weight <= 0 for weight in mixture.values()):
        raise ValueError("tokenizer hash and positive mixture are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    writers: dict[tuple[str, str], _Writer] = {}
    shards: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    next_shard_id = 0
    try:
        for document in documents:
            if not document.document_id or document.document_id in seen_ids:
                raise ValueError(f"missing or duplicate document id: {document.document_id!r}")
            seen_ids.add(document.document_id)
            if document.split not in {"train", "validation", "test"} or not document.domain:
                raise ValueError("invalid split/domain in tokenized document")
            if document.split == "train" and document.domain not in mixture:
                raise ValueError(f"train domain missing from mixture: {document.domain}")
            if any(not 0 <= token < vocab_size or token == eos_token_id for token in document.token_ids):
                raise ValueError(f"invalid token or EOS collision: {document.document_id}")
            key = (document.split, document.domain)
            writer = writers.get(key)
            length = len(document.token_ids) + 1
            if writer is not None and writer.tokens and writer.tokens + length > target_shard_tokens:
                entry = writer.finish()
                if entry is not None:
                    shards.append(entry)
                writer = None
            if writer is None:
                writer = _Writer(temporary, *key, next_shard_id, sequence_length)
                next_shard_id += 1
                writers[key] = writer
            writer.append(document.token_ids, eos_token_id)
        for writer in writers.values():
            entry = writer.finish()
            if entry is not None:
                shards.append(entry)
        # Writers can finish out of creation order because domains interleave.
        # Rename to stable contiguous global IDs after all streams are closed.
        shards.sort(key=lambda entry: entry["shard_id"])
        for entry in shards:
            for name in (entry["bin"], entry["idx"]):
                (temporary / name).rename(temporary / (name + ".renaming"))
        for actual_id, entry in enumerate(shards):
            old_bin, old_idx = entry["bin"], entry["idx"]
            new_bin = f"{entry['split']}-{actual_id:05d}.bin"
            new_idx = f"{entry['split']}-{actual_id:05d}.idx"
            (temporary / (old_bin + ".renaming")).rename(temporary / new_bin)
            (temporary / (old_idx + ".renaming")).rename(temporary / new_idx)
            entry["bin"], entry["idx"] = new_bin, new_idx
            entry["shard_id"] = actual_id
        train_domains = {entry["domain"] for entry in shards if entry["split"] == "train"}
        if train_domains != set(mixture):
            raise ValueError("mixture domains differ from retained train shard domains")
        if not any(entry["split"] == "validation" for entry in shards):
            raise ValueError("validation split needs at least one sample")
        manifest = {
            "version": VERSION,
            "stage": (
                "approved_sharded_token_stream"
                if license_gate.get("training_eligible")
                else "internal_smoke_sharded_token_stream"
            ),
            "dtype": "uint16-le",
            "index_dtype": "<QII",
            "sequence_length": sequence_length,
            "stride": sequence_length - 1,
            "vocab_size": vocab_size,
            "eos_token_id": eos_token_id,
            "tokenizer_sha256": tokenizer_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "license_gate": license_gate,
            "mixture": mixture,
            "shards": shards,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
        return manifest
    except BaseException:
        for writer in writers.values():
            if not writer.handle.closed:
                writer.handle.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
