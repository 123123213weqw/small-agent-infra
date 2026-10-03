"""Checkpointable deterministic sampler for single-node DDP training."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import Sampler


class StatefulDistributedSampler(Sampler[int]):
    """Shard one deterministic epoch order across DDP ranks.

    ``samples_per_rank_multiple`` trims the epoch so every rank consumes a
    whole number of optimizer-step sample groups.  This prevents DataLoader's
    ``drop_last`` from silently discarding a different tail when microbatch
    size changes.

    The cursor is advanced before yielding an index, so ``state_dict`` points
    to the next index to consume.  Exact cursor recovery requires a DataLoader
    without worker prefetch (``num_workers=0``); worker-safe commit tracking is
    deliberately deferred until the basic trainer is validated.
    """

    VERSION = 2

    def __init__(
        self,
        dataset_size: int,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        samples_per_rank_multiple: int = 1,
    ) -> None:
        if dataset_size < 1:
            raise ValueError("dataset_size must be positive")
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if samples_per_rank_multiple < 1:
            raise ValueError("samples_per_rank_multiple must be positive")
        if not drop_last and samples_per_rank_multiple != 1:
            raise ValueError("padding mode only supports samples_per_rank_multiple=1")
        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.samples_per_rank_multiple = samples_per_rank_multiple
        self.epoch = 0
        self.cursor = 0

        if drop_last:
            global_multiple = num_replicas * samples_per_rank_multiple
            self.total_size = dataset_size // global_multiple * global_multiple
            self.num_samples = self.total_size // num_replicas
        else:
            self.num_samples = (dataset_size + num_replicas - 1) // num_replicas
            self.total_size = self.num_samples * num_replicas
        if self.num_samples == 0:
            raise ValueError("dataset is too small for drop_last DDP sharding")

    def _global_indices(self) -> list[int]:
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        else:
            indices = list(range(self.dataset_size))
        if self.drop_last:
            return indices[: self.total_size]
        padding = self.total_size - len(indices)
        if padding:
            repeats = (padding + len(indices) - 1) // len(indices)
            indices.extend((indices * repeats)[:padding])
        return indices

    def rank_indices(self) -> list[int]:
        indices = self._global_indices()
        result = indices[self.rank : self.total_size : self.num_replicas]
        if len(result) != self.num_samples:
            raise AssertionError("distributed sampler produced an uneven shard")
        return result

    def __iter__(self) -> Iterator[int]:
        indices = self.rank_indices()
        while self.cursor < self.num_samples:
            position = self.cursor
            self.cursor += 1
            yield indices[position]

    def __len__(self) -> int:
        return self.num_samples - self.cursor

    @property
    def remaining(self) -> int:
        return self.num_samples - self.cursor

    def set_epoch(self, epoch: int, *, reset_cursor: bool = True) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch
        if reset_cursor:
            self.cursor = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "dataset_size": self.dataset_size,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "samples_per_rank_multiple": self.samples_per_rank_multiple,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "num_samples": self.num_samples,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "version": self.VERSION,
            "dataset_size": self.dataset_size,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "samples_per_rank_multiple": self.samples_per_rank_multiple,
            "num_samples": self.num_samples,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"sampler state mismatch for {key}: {state.get(key)!r} != {value!r}"
                )
        epoch, cursor = int(state["epoch"]), int(state["cursor"])
        if epoch < 0 or not 0 <= cursor <= self.num_samples:
            raise ValueError("invalid sampler epoch or cursor")
        self.epoch = epoch
        self.cursor = cursor
