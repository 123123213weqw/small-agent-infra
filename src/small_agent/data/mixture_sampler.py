"""Stateless-position domain mixing and commit-based DDP sample recovery."""

from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import random
from typing import Any, Iterator

from torch.utils.data import Sampler

from .sharded_dataset import ShardedTokenDataset
from .sharded_format import sha256_file


def _seed(*parts: object) -> int:
    encoded = json.dumps(parts, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def mixture_counts(weights: dict[str, float], block_size: int) -> dict[str, int]:
    if block_size < 1 or not weights:
        raise ValueError("positive block size and nonempty mixture required")
    domains = sorted(weights)
    rational = {domain: Fraction(str(weights[domain])) for domain in domains}
    if any(weight <= 0 for weight in rational.values()):
        raise ValueError("mixture weights must be positive")
    denominator = sum(rational.values())
    targets = {domain: rational[domain] * block_size / denominator for domain in domains}
    counts = {domain: int(targets[domain]) for domain in domains}
    remaining = block_size - sum(counts.values())
    remainder_order = sorted(domains, key=lambda domain: (-(targets[domain] - counts[domain]), domain))
    for domain in remainder_order[:remaining]:
        counts[domain] += 1
    if any(count < 1 for count in counts.values()):
        raise ValueError("mixture block is too small to include every domain")
    return counts


class DeterministicMixtureSchedule:
    """Map a global position to a stable sample reference, without mutable RNG state."""

    VERSION = 1

    def __init__(self, dataset: ShardedTokenDataset, *, seed: int, block_size: int = 1000):
        if dataset.split != "train":
            raise ValueError("mixture schedule only applies to the train split")
        self.dataset = dataset
        self.seed = int(seed)
        self.block_size = int(block_size)
        self.manifest_sha256 = sha256_file(dataset.data_dir / "manifest.json")
        self.counts = mixture_counts(dataset.manifest["mixture"], self.block_size)
        self.domains = tuple(sorted(self.counts))
        self.shard_indices = {
            domain: [i for i, shard in enumerate(dataset.shards) if shard.domain == domain]
            for domain in self.domains
        }
        if any(not shards for shards in self.shard_indices.values()):
            raise ValueError("mixture domain has no train shards")
        self.domain_samples = {
            domain: sum(dataset.shards[i].samples for i in indices)
            for domain, indices in self.shard_indices.items()
        }

    @lru_cache(maxsize=16)
    def _block(self, block: int) -> tuple[tuple[str, int], ...]:
        tickets = [domain for domain in self.domains for _ in range(self.counts[domain])]
        random.Random(_seed(self.seed, self.manifest_sha256, "mixture", block)).shuffle(tickets)
        prior = dict.fromkeys(self.domains, 0)
        results: list[tuple[str, int]] = []
        for domain in tickets:
            results.append((domain, prior[domain]))
            prior[domain] += 1
        return tuple(results)

    @lru_cache(maxsize=16)
    def _cycle_layout(self, domain: str, cycle: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        order = self.shard_indices[domain].copy()
        random.Random(_seed(self.seed, self.manifest_sha256, "shards", domain, cycle)).shuffle(order)
        prefix = [0]
        for local_shard in order:
            prefix.append(prefix[-1] + self.dataset.shards[local_shard].samples)
        return tuple(order), tuple(prefix)

    @lru_cache(maxsize=32)
    def _sample_order(self, local_shard: int, cycle: int) -> tuple[int, ...]:
        sample_ids = list(range(self.dataset.shards[local_shard].samples))
        random.Random(
            _seed(self.seed, self.manifest_sha256, "samples", self.dataset.shards[local_shard].shard_id, cycle)
        ).shuffle(sample_ids)
        return tuple(sample_ids)

    def sample_ref_at(self, global_position: int) -> int:
        if global_position < 0:
            raise ValueError("global position must be nonnegative")
        block, position = divmod(global_position, self.block_size)
        domain, prior_in_block = self._block(block)[position]
        ordinal = block * self.counts[domain] + prior_in_block
        cycle, within_cycle = divmod(ordinal, self.domain_samples[domain])
        order, prefix = self._cycle_layout(domain, cycle)
        order_index = bisect_right(prefix, within_cycle) - 1
        local_shard = order[order_index]
        sample_position = within_cycle - prefix[order_index]
        sample_id = self._sample_order(local_shard, cycle)[sample_position]
        return self.dataset._prefix[local_shard] + sample_id

    def reference_at(self, global_position: int) -> dict[str, int | str]:
        return {
            "global_position": global_position,
            **self.dataset.reference(self.sample_ref_at(global_position)),
        }


class MixtureDistributedSampler(Sampler[int]):
    """Dispatch ahead, but checkpoint only optimizer-step-committed positions."""

    VERSION = 1

    def __init__(
        self,
        schedule: DeterministicMixtureSchedule,
        *,
        num_replicas: int,
        rank: int,
        micro_batch_size: int,
        gradient_accumulation_steps: int,
    ) -> None:
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("invalid DDP rank/world size")
        if micro_batch_size < 1 or gradient_accumulation_steps < 1:
            raise ValueError("invalid batch or accumulation size")
        self.schedule = schedule
        self.num_replicas = num_replicas
        self.rank = rank
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_step_samples = num_replicas * micro_batch_size * gradient_accumulation_steps
        self.committed_global_position = 0
        self.produced_local_samples = 0
        self.dispatch_base = 0

    def global_position_for_local(self, local_sample: int, *, base: int | None = None) -> int:
        if local_sample < 0:
            raise ValueError("local sample position must be nonnegative")
        microstep, within_batch = divmod(local_sample, self.micro_batch_size)
        start = self.dispatch_base if base is None else base
        return start + microstep * self.num_replicas * self.micro_batch_size + self.rank * self.micro_batch_size + within_batch

    def __iter__(self) -> Iterator[int]:
        self.dispatch_base = self.committed_global_position
        self.produced_local_samples = 0
        while True:
            position = self.global_position_for_local(self.produced_local_samples)
            self.produced_local_samples += 1
            yield self.schedule.sample_ref_at(position)

    def commit_step(self) -> None:
        next_position = self.committed_global_position + self.global_step_samples
        local_committed = (next_position - self.dispatch_base) // self.num_replicas
        if self.produced_local_samples < local_committed:
            raise ValueError("cannot commit samples not yet dispatched")
        self.committed_global_position = next_position

    @property
    def epoch(self) -> int:
        return self.committed_global_position // max(len(self.schedule.dataset), 1)

    @property
    def cursor(self) -> int:
        return self.committed_global_position // self.num_replicas

    def state_dict(self) -> dict[str, Any]:
        next_position = self.committed_global_position + self.rank * self.micro_batch_size
        return {
            "version": self.VERSION,
            "manifest_sha256": self.schedule.manifest_sha256,
            "seed": self.schedule.seed,
            "mixture_block_size": self.schedule.block_size,
            "mixture_counts": self.schedule.counts,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "committed_global_position": self.committed_global_position,
            "next_reference": self.schedule.reference_at(next_position),
        }

    def global_commit_fingerprint(self) -> dict[str, Any]:
        """Fields that must agree across all DDP ranks at a checkpoint barrier."""
        return {
            "manifest_sha256": self.schedule.manifest_sha256,
            "seed": self.schedule.seed,
            "mixture_counts": self.schedule.counts,
            "num_replicas": self.num_replicas,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "committed_global_position": self.committed_global_position,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "version": self.VERSION,
            "manifest_sha256": self.schedule.manifest_sha256,
            "seed": self.schedule.seed,
            "mixture_block_size": self.schedule.block_size,
            "mixture_counts": self.schedule.counts,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"mixture sampler state mismatch for {key}")
        position = int(state["committed_global_position"])
        if position < 0 or position % self.global_step_samples:
            raise ValueError("committed position is not an optimizer-step boundary")
        next_position = position + self.rank * self.micro_batch_size
        if state.get("next_reference") != self.schedule.reference_at(next_position):
            raise ValueError("next shard/sample/token offset changed across resume")
        self.committed_global_position = position
        self.dispatch_base = position
        self.produced_local_samples = 0
