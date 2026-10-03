from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_agent.data import (  # noqa: E402
    DeterministicMixtureSchedule,
    MixtureDistributedSampler,
    ShardedTokenDataset,
)
from small_agent.data.sharded_pack import TokenizedDocument, pack_sharded  # noqa: E402
from small_agent.data.sharded_format import sha256_file  # noqa: E402
from small_agent.training.checkpoint import load_full_checkpoint, save_full_checkpoint  # noqa: E402
from small_agent.training.scheduler import TokenLRScheduler, WarmupCosineConfig  # noqa: E402


def make_shards(root: Path) -> Path:
    documents = []
    for domain, start in (("general", 10), ("code", 40)):
        for i in range(8):
            documents.append(TokenizedDocument(f"{domain}-{i}", "train", domain, [start + i, start + i + 1, start + i + 2, start + i + 3]))
    documents.extend(
        [
            TokenizedDocument("val-0", "validation", "general", [70, 71, 72]),
            TokenizedDocument("val-1", "validation", "general", [73, 74]),
        ]
    )
    pack_sharded(
        documents,
        root,
        sequence_length=5,
        target_shard_tokens=12,
        vocab_size=128,
        eos_token_id=2,
        tokenizer_sha256="toy-tokenizer-hash",
        mixture={"general": 0.5, "code": 0.5},
        license_gate={"training_eligible": False},
    )
    return root


class ShardedDataTests(unittest.TestCase):
    def test_pack_mmap_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_shards(Path(temporary) / "shards")
            train = ShardedTokenDataset(root, "train", 5)
            validation = ShardedTokenDataset(root, "validation", 5)
            self.assertEqual(len(train), 16)
            self.assertEqual(len(validation), 2)
            self.assertGreater(len(train.shards), 2)
            first = train[0]
            self.assertEqual(first["input_ids"].tolist(), [10, 11, 12, 13, 2])
            self.assertEqual(first["token_offset"].item(), 0)
            self.assertEqual(first["sample_id"].item(), 0)
            tail = validation[1]
            self.assertEqual(tail["valid_tokens"].item(), 3)
            self.assertEqual(tail["labels"].tolist()[-2:], [-100, -100])
            self.assertAlmostEqual(train.coverage_report()["prediction_coverage"], 8 / 9)

    def test_shard_corruption_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_shards(Path(temporary) / "shards")
            manifest = json.loads((root / "manifest.json").read_text())
            shard = root / manifest["shards"][0]["idx"]
            contents = bytearray(shard.read_bytes())
            contents[1] ^= 1
            shard.write_bytes(contents)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                ShardedTokenDataset(root, "train", 5)

    def test_index_semantics_rejected_even_with_updated_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_shards(Path(temporary) / "shards")
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            shard = root / manifest["shards"][0]["idx"]
            contents = bytearray(shard.read_bytes())
            contents[0] = 1  # First sample must start at token zero.
            shard.write_bytes(contents)
            manifest["shards"][0]["idx_sha256"] = sha256_file(shard)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "index records are inconsistent"):
                ShardedTokenDataset(root, "train", 5)

    def test_schedule_is_deterministic_and_domain_balanced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = ShardedTokenDataset(make_shards(Path(temporary) / "shards"), "train", 5)
            one = DeterministicMixtureSchedule(dataset, seed=19, block_size=4)
            two = DeterministicMixtureSchedule(dataset, seed=19, block_size=4)
            refs = [one.sample_ref_at(i) for i in range(40)]
            self.assertEqual(refs, [two.sample_ref_at(i) for i in range(40)])
            self.assertEqual(len(set(refs[:16])), 16)
            for start in range(0, 40, 4):
                domains = [one.reference_at(i)["domain"] for i in range(start, start + 4)]
                self.assertEqual(domains.count("general"), 2)
                self.assertEqual(domains.count("code"), 2)

    def test_ddp_positions_are_disjoint_and_match_global_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = ShardedTokenDataset(make_shards(Path(temporary) / "shards"), "train", 5)
            schedule = DeterministicMixtureSchedule(dataset, seed=23, block_size=4)
            samplers = [
                MixtureDistributedSampler(schedule, num_replicas=2, rank=rank, micro_batch_size=2, gradient_accumulation_steps=2)
                for rank in range(2)
            ]
            iterators = [iter(sampler) for sampler in samplers]
            emitted = [[next(iterator) for _ in range(4)] for iterator in iterators]
            merged = emitted[0][:2] + emitted[1][:2] + emitted[0][2:] + emitted[1][2:]
            self.assertEqual(merged, [schedule.sample_ref_at(i) for i in range(8)])
            self.assertEqual(len(set(merged)), 8)

    def test_commit_only_and_resume_ignores_dispatched_ahead(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = ShardedTokenDataset(make_shards(Path(temporary) / "shards"), "train", 5)
            schedule = DeterministicMixtureSchedule(dataset, seed=17, block_size=4)
            sampler = MixtureDistributedSampler(
                schedule, num_replicas=2, rank=1, micro_batch_size=1, gradient_accumulation_steps=2
            )
            iterator = iter(sampler)
            first_step = [next(iterator) for _ in range(2)]
            _prefetched_but_untrained = [next(iterator) for _ in range(4)]
            sampler.commit_step()
            state = sampler.state_dict()
            self.assertEqual(state["committed_global_position"], 4)
            self.assertEqual(first_step, [schedule.sample_ref_at(1), schedule.sample_ref_at(3)])
            resumed = MixtureDistributedSampler(
                schedule, num_replicas=2, rank=1, micro_batch_size=1, gradient_accumulation_steps=2
            )
            resumed.load_state_dict(state)
            loader = DataLoader(dataset, batch_size=1, sampler=resumed, num_workers=0)
            batches = iter(loader)
            trace = [int(next(batches)["sample_ref"][0]) for _ in range(6)]
            self.assertEqual(trace, [schedule.sample_ref_at(position) for position in (5, 7, 9, 11, 13, 15)])
            mutated = dict(state)
            mutated["next_reference"] = dict(state["next_reference"])
            mutated["next_reference"]["token_offset"] += 1
            with self.assertRaisesRegex(ValueError, "next shard/sample/token offset"):
                resumed.load_state_dict(mutated)

    def test_optimizer_checkpoint_restores_exact_next_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = ShardedTokenDataset(make_shards(root / "shards"), "train", 5)
            schedule = DeterministicMixtureSchedule(dataset, seed=41, block_size=4)
            sampler = MixtureDistributedSampler(
                schedule, num_replicas=1, rank=0, micro_batch_size=1, gradient_accumulation_steps=2
            )
            batches = iter(DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0))
            first = [int(next(batches)["sample_ref"][0]) for _ in range(2)]
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = TokenLRScheduler(optimizer, WarmupCosineConfig(1e-3, 1e-4, 1, 100))
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            scheduler.step(8)
            sampler.commit_step()
            compatibility = {"data_reader": "sharded_v1", "manifest": schedule.manifest_sha256}
            checkpoint, _ = save_full_checkpoint(
                root / "checkpoints", step=1, model=model, optimizer=optimizer,
                scheduler=scheduler, sampler=sampler, compatibility=compatibility,
            )
            expected = [int(next(batches)["sample_ref"][0]) for _ in range(10)]
            restored_sampler = MixtureDistributedSampler(
                schedule, num_replicas=1, rank=0, micro_batch_size=1, gradient_accumulation_steps=2
            )
            restored_model = torch.nn.Linear(2, 2)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
            restored_scheduler = TokenLRScheduler(
                restored_optimizer, WarmupCosineConfig(1e-3, 1e-4, 1, 100)
            )
            load_full_checkpoint(
                checkpoint, model=restored_model, optimizer=restored_optimizer,
                scheduler=restored_scheduler, sampler=restored_sampler,
                compatibility=compatibility, device=torch.device("cpu"),
            )
            resumed = iter(DataLoader(dataset, batch_size=1, sampler=restored_sampler, num_workers=0))
            actual = [int(next(resumed)["sample_ref"][0]) for _ in range(10)]
            self.assertEqual(actual, expected)
            self.assertEqual(first + actual[:2], [schedule.sample_ref_at(i) for i in range(4)])

    def test_worker_prefetch_preserves_order_and_commit_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = ShardedTokenDataset(make_shards(Path(temporary) / "shards"), "train", 5)
            schedule = DeterministicMixtureSchedule(dataset, seed=43, block_size=4)
            baseline = MixtureDistributedSampler(
                schedule, num_replicas=1, rank=0, micro_batch_size=1,
                gradient_accumulation_steps=2,
            )
            baseline_loader = iter(DataLoader(dataset, batch_size=1, sampler=baseline, num_workers=0))
            expected = [int(next(baseline_loader)["sample_ref"][0]) for _ in range(18)]

            prefetched = MixtureDistributedSampler(
                schedule, num_replicas=1, rank=0, micro_batch_size=1,
                gradient_accumulation_steps=2,
            )
            loader = DataLoader(
                dataset, batch_size=1, sampler=prefetched, num_workers=2,
                prefetch_factor=2, pin_memory=True,
            )
            batches = iter(loader)
            consumed = [int(next(batches)["sample_ref"][0]) for _ in range(2)]
            self.assertEqual(consumed, expected[:2])
            prefetched.commit_step()
            state = prefetched.state_dict()
            self.assertEqual(state["committed_global_position"], 2)
            # Workers are permitted to have dispatched later positions already.
            self.assertGreaterEqual(prefetched.produced_local_samples, 2)
            rest = [int(next(batches)["sample_ref"][0]) for _ in range(16)]
            self.assertEqual(consumed + rest, expected)
            restored = MixtureDistributedSampler(
                schedule, num_replicas=1, rank=0, micro_batch_size=1,
                gradient_accumulation_steps=2,
            )
            restored.load_state_dict(state)
            resumed = iter(DataLoader(dataset, batch_size=1, sampler=restored, num_workers=0))
            self.assertEqual(
                [int(next(resumed)["sample_ref"][0]) for _ in range(16)],
                expected[2:],
            )
            del batches, loader

    def test_two_rank_worker_prefetch_resume_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = ShardedTokenDataset(make_shards(Path(temporary) / "shards"), "train", 5)
            schedule = DeterministicMixtureSchedule(dataset, seed=47, block_size=4)
            for rank in (0, 1):
                sampler = MixtureDistributedSampler(
                    schedule, num_replicas=2, rank=rank,
                    micro_batch_size=1, gradient_accumulation_steps=2,
                )
                loader = DataLoader(
                    dataset, batch_size=1, sampler=sampler, num_workers=2,
                    prefetch_factor=2, pin_memory=True,
                )
                iterator = iter(loader)
                first_step = [int(next(iterator)["sample_ref"][0]) for _ in range(2)]
                sampler.commit_step()
                state = sampler.state_dict()
                expected = [int(next(iterator)["sample_ref"][0]) for _ in range(8)]
                restored = MixtureDistributedSampler(
                    schedule, num_replicas=2, rank=rank,
                    micro_batch_size=1, gradient_accumulation_steps=2,
                )
                restored.load_state_dict(state)
                resumed = iter(DataLoader(dataset, batch_size=1, sampler=restored, num_workers=0))
                actual = [int(next(resumed)["sample_ref"][0]) for _ in range(8)]
                self.assertEqual(actual, expected)
                self.assertEqual(first_step + actual, [
                    schedule.sample_ref_at(position)
                    for position in range(rank, 20, 2)
                ])
                del iterator, loader


if __name__ == "__main__":
    unittest.main()
