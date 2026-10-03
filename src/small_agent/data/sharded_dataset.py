"""Indexed mmap reader for immutable token shards."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .sharded_format import FLAG_PADDED_TAIL, INDEX_DTYPE, ShardInfo, load_sharded_manifest


class ShardedTokenDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        sequence_length: int,
        *,
        pad_token_id: int = 0,
        verify_hashes: bool = True,
        max_open_shards: int = 4,
    ) -> None:
        if max_open_shards < 1:
            raise ValueError("max_open_shards must be positive")
        self.data_dir = Path(data_dir).resolve()
        self.manifest, all_shards = load_sharded_manifest(self.data_dir, verify_hashes=verify_hashes)
        if sequence_length != int(self.manifest["sequence_length"]):
            raise ValueError("sequence length differs from frozen shard format")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split: {split}")
        if not 0 <= pad_token_id < int(self.manifest["vocab_size"]):
            raise ValueError("pad token outside vocabulary")
        self.split = split
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
        self.shards: list[ShardInfo] = [shard for shard in all_shards if shard.split == split]
        self._prefix = [0]
        for shard in self.shards:
            self._prefix.append(self._prefix[-1] + shard.samples)
        self._open: OrderedDict[int, tuple[np.memmap, np.memmap]] = OrderedDict()
        self.max_open_shards = max_open_shards

    def __len__(self) -> int:
        return self._prefix[-1]

    def _locate(self, sample_ref: int) -> tuple[int, int, ShardInfo]:
        if sample_ref < 0:
            sample_ref += len(self)
        if not 0 <= sample_ref < len(self):
            raise IndexError(sample_ref)
        local_shard = bisect_right(self._prefix, sample_ref) - 1
        return local_shard, sample_ref - self._prefix[local_shard], self.shards[local_shard]

    def reference(self, sample_ref: int) -> dict[str, int | str]:
        local_shard, sample_id, shard = self._locate(sample_ref)
        _, index = self._open_shard(local_shard)
        row = index[sample_id]
        return {
            "sample_ref": sample_ref,
            "shard_id": shard.shard_id,
            "sample_id": sample_id,
            "token_offset": int(row["token_offset"]),
            "domain": shard.domain,
        }

    def _open_shard(self, local_shard: int) -> tuple[np.memmap, np.memmap]:
        cached = self._open.get(local_shard)
        if cached is not None:
            self._open.move_to_end(local_shard)
            return cached
        shard = self.shards[local_shard]
        tokens = np.memmap(shard.bin_path, dtype="<u2", mode="r", shape=(shard.tokens,))
        index = np.memmap(shard.idx_path, dtype=INDEX_DTYPE, mode="r", shape=(shard.samples,))
        cached = (tokens, index)
        self._open[local_shard] = cached
        if len(self._open) > self.max_open_shards:
            self._open.popitem(last=False)
        return cached

    def __getitem__(self, sample_ref: int) -> dict[str, torch.Tensor]:
        local_shard, sample_id, shard = self._locate(int(sample_ref))
        stream, index = self._open_shard(local_shard)
        row = index[sample_id]
        start, valid, flags = (int(row[name]) for name in ("token_offset", "valid_tokens", "flags"))
        if (
            start != sample_id * (self.sequence_length - 1)
            or not 2 <= valid <= self.sequence_length
            or start + valid > shard.tokens
            or flags != (FLAG_PADDED_TAIL if valid < self.sequence_length else 0)
        ):
            raise ValueError(f"invalid index record: shard={shard.shard_id} sample={sample_id}")
        ids = np.full(self.sequence_length, self.pad_token_id, dtype=np.int64)
        ids[:valid] = stream[start : start + valid]
        if int(ids[:valid].max()) >= int(self.manifest["vocab_size"]):
            raise ValueError("token outside vocabulary")
        labels = ids.copy()
        labels[valid:] = -100
        mask = np.zeros(self.sequence_length, dtype=np.int64)
        mask[:valid] = 1
        return {
            "input_ids": torch.from_numpy(ids),
            "labels": torch.from_numpy(labels),
            "attention_mask": torch.from_numpy(mask),
            "sample_ref": torch.tensor(sample_ref, dtype=torch.int64),
            "shard_id": torch.tensor(shard.shard_id, dtype=torch.int64),
            "sample_id": torch.tensor(sample_id, dtype=torch.int64),
            "token_offset": torch.tensor(start, dtype=torch.int64),
            "valid_tokens": torch.tensor(valid, dtype=torch.int64),
        }

    def coverage_report(self) -> dict[str, Any]:
        tokens = sum(shard.tokens for shard in self.shards)
        predicted = sum(
            min(shard.samples * (self.sequence_length - 1), shard.tokens - 1)
            for shard in self.shards
        )
        return {
            "format": "sharded_v1",
            "split": self.split,
            "shards": len(self.shards),
            "stream_tokens": tokens,
            "samples": len(self),
            "sequence_length": self.sequence_length,
            "predicted_tokens": predicted,
            "prediction_coverage": predicted / max(tokens - len(self.shards), 1),
        }
