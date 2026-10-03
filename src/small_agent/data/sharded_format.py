"""Stable on-disk format for mmap-backed token shards."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "sharded_token_stream_v1"
INDEX_DTYPE = np.dtype([("token_offset", "<u8"), ("valid_tokens", "<u4"), ("flags", "<u4")])
FLAG_PADDED_TAIL = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_shard_path(root: Path, name: str) -> Path:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"unsafe shard filename: {name!r}")
    return root / name


@dataclass(frozen=True)
class ShardInfo:
    shard_id: int
    split: str
    domain: str
    bin_path: Path
    idx_path: Path
    tokens: int
    samples: int


def load_sharded_manifest(root: str | Path, *, verify_hashes: bool = True) -> tuple[dict[str, Any], list[ShardInfo]]:
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError(f"unsupported sharded manifest: {manifest.get('version')!r}")
    if manifest.get("dtype") != "uint16-le" or manifest.get("index_dtype") != "<QII":
        raise ValueError("unsupported sharded token/index dtype")
    sequence_length = int(manifest["sequence_length"])
    if sequence_length < 2 or int(manifest["stride"]) != sequence_length - 1:
        raise ValueError("invalid sharded sequence length or stride")
    vocab_size = int(manifest["vocab_size"])
    if not 0 <= int(manifest["eos_token_id"]) < vocab_size <= 65536:
        raise ValueError("invalid sharded vocabulary metadata")
    if not manifest.get("tokenizer_sha256"):
        raise ValueError("sharded manifest is missing tokenizer SHA-256")
    shards: list[ShardInfo] = []
    seen_names: set[str] = set()
    for expected_id, entry in enumerate(manifest["shards"]):
        if int(entry["shard_id"]) != expected_id:
            raise ValueError("shard ids must be contiguous and ordered")
        split, domain = str(entry["split"]), str(entry["domain"])
        if split not in {"train", "validation", "test"} or not domain:
            raise ValueError("invalid split or domain")
        paths: dict[str, Path] = {}
        for kind, suffix, bytes_per_item in (("bin", ".bin", 2), ("idx", ".idx", 16)):
            name = str(entry[kind])
            if not name.endswith(suffix) or name in seen_names:
                raise ValueError(f"duplicate or invalid {kind} filename: {name}")
            seen_names.add(name)
            path = safe_shard_path(root, name)
            expected_items = int(entry["tokens" if kind == "bin" else "samples"])
            expected_bytes = expected_items * bytes_per_item
            if int(entry[f"{kind}_bytes"]) != expected_bytes or path.stat().st_size != expected_bytes:
                raise ValueError(f"shard {kind} byte count mismatch: {path}")
            if verify_hashes and sha256_file(path) != entry[f"{kind}_sha256"]:
                raise ValueError(f"shard {kind} SHA-256 mismatch: {path}")
            paths[kind] = path
        tokens = int(entry["tokens"])
        samples = int(entry["samples"])
        if tokens < 2 or samples < 1:
            raise ValueError("empty shard is not allowed")
        expected_samples = (
            1 + (tokens - sequence_length) // (sequence_length - 1)
            if split == "train" and tokens >= sequence_length
            else 0 if split == "train"
            else (tokens - 1 + sequence_length - 2) // (sequence_length - 1)
        )
        if samples != expected_samples:
            raise ValueError(f"shard sample count mismatch: {expected_id}")
        index = np.memmap(paths["idx"], dtype=INDEX_DTYPE, mode="r", shape=(samples,))
        expected_offsets = np.arange(samples, dtype=np.uint64) * np.uint64(sequence_length - 1)
        expected_valid = np.minimum(sequence_length, tokens - expected_offsets).astype(np.uint32)
        expected_flags = np.where(expected_valid < sequence_length, FLAG_PADDED_TAIL, 0)
        if (
            not np.array_equal(index["token_offset"], expected_offsets)
            or not np.array_equal(index["valid_tokens"], expected_valid)
            or not np.array_equal(index["flags"], expected_flags)
        ):
            raise ValueError(f"shard index records are inconsistent: {expected_id}")
        shards.append(ShardInfo(expected_id, split, domain, paths["bin"], paths["idx"], tokens, samples))
    if not shards:
        raise ValueError("manifest has no shards")
    for split in ("train", "validation"):
        if not any(shard.split == split for shard in shards):
            raise ValueError(f"manifest has no {split} shards")
    mixture = manifest.get("mixture", {})
    train_domains = {shard.domain for shard in shards if shard.split == "train"}
    if set(mixture) != train_domains or any(float(weight) <= 0 for weight in mixture.values()):
        raise ValueError("mixture domains/weights do not match train shards")
    return manifest, shards
