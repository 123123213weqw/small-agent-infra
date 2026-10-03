from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_agent.data import StatefulDistributedSampler  # noqa: E402
from small_agent.training.checkpoint import (  # noqa: E402
    load_distributed_full_checkpoint,
    load_full_checkpoint,
    resolve_checkpoint,
    save_distributed_full_checkpoint,
    save_full_checkpoint,
)
from small_agent.training.scheduler import (  # noqa: E402
    TokenLRScheduler,
    WarmupCosineConfig,
)


def _distributed_roundtrip_worker(
    rank: int, world_size: int, rendezvous: str, checkpoint_root: str, result_root: str
) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    try:
        device = torch.device("cpu")
        torch.manual_seed(123)
        model = torch.nn.Linear(3, 2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = TokenLRScheduler(
            optimizer, WarmupCosineConfig(1e-3, 1e-4, 2, 20)
        )
        sampler = StatefulDistributedSampler(
            16, num_replicas=world_size, rank=rank, seed=17
        )
        sampler_iterator = iter(sampler)
        next(sampler_iterator)
        next(sampler_iterator)
        loader_generator = torch.Generator().manual_seed(2000 + rank)
        torch.rand((), generator=loader_generator)

        loss = model(torch.ones((2, 3), device=device)).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step(8)
        torch.manual_seed(1000 + rank)
        torch.rand(())
        compatibility = {"test": "distributed-roundtrip", "world_size": world_size}
        checkpoint, _ = save_distributed_full_checkpoint(
            Path(checkpoint_root),
            step=1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            loader_generator=loader_generator,
            compatibility=compatibility,
            rank=rank,
            world_size=world_size,
            keep=1,
        )
        expected_torch_random = float(torch.rand(()))
        expected_loader_random = float(torch.rand((), generator=loader_generator))

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        sampler.set_epoch(3)
        scheduler.step(5)
        torch.manual_seed(9999)
        loader_generator.manual_seed(9999)

        manifest = load_distributed_full_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            loader_generator=loader_generator,
            compatibility=compatibility,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        result = {
            "step": manifest["step"],
            "tokens_seen": scheduler.tokens_seen,
            "sampler_cursor": sampler.cursor,
            "torch_random": float(torch.rand(())),
            "expected_torch_random": expected_torch_random,
            "loader_random": float(torch.rand((), generator=loader_generator)),
            "expected_loader_random": expected_loader_random,
        }
        (Path(result_root) / f"rank_{rank}.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
    finally:
        dist.destroy_process_group()


class CheckpointTests(unittest.TestCase):
    def test_distributed_roundtrip_restores_rank_local_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rendezvous = root / "rendezvous"
            checkpoints = root / "checkpoints"
            results = root / "results"
            results.mkdir()
            mp.spawn(
                _distributed_roundtrip_worker,
                args=(2, str(rendezvous), str(checkpoints), str(results)),
                nprocs=2,
                join=True,
            )
            checkpoint = resolve_checkpoint(checkpoints)
            self.assertTrue((checkpoint / "runtime_rank_00000.pt").is_file())
            self.assertTrue((checkpoint / "runtime_rank_00001.pt").is_file())
            self.assertTrue((checkpoint / "sampler_rank_00000.json").is_file())
            self.assertTrue((checkpoint / "sampler_rank_00001.json").is_file())
            for rank in range(2):
                result = json.loads((results / f"rank_{rank}.json").read_text())
                self.assertEqual(result["step"], 1)
                self.assertEqual(result["tokens_seen"], 8)
                self.assertEqual(result["sampler_cursor"], 2)
                self.assertEqual(
                    result["torch_random"], result["expected_torch_random"]
                )
                self.assertEqual(
                    result["loader_random"], result["expected_loader_random"]
                )

    def test_roundtrip_restores_all_training_state(self) -> None:
        device = torch.device("cpu")
        model = torch.nn.Linear(4, 3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        schedule_config = WarmupCosineConfig(1e-3, 1e-4, 10, 100)
        scheduler = TokenLRScheduler(optimizer, schedule_config)
        sampler = StatefulDistributedSampler(20, seed=7)

        loss = model(torch.ones((2, 4), device=device)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step(8)
        consumed = [next(iter(sampler)) for _ in range(3)]
        self.assertEqual(len(consumed), 3)
        expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
        compatibility = {"model": "abc", "data": "def"}

        with tempfile.TemporaryDirectory() as temporary:
            checkpoints = Path(temporary)
            path, _ = save_full_checkpoint(
                checkpoints,
                step=1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                compatibility=compatibility,
                keep=1,
            )
            self.assertEqual(resolve_checkpoint(checkpoints), path)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            scheduler.step(3)
            sampler.set_epoch(2)
            manifest = load_full_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                compatibility=compatibility,
                device=device,
            )
            self.assertEqual(manifest["step"], 1)
            self.assertEqual(scheduler.tokens_seen, 8)
            self.assertEqual(sampler.cursor, 3)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_compatibility_drift_is_rejected(self) -> None:
        device = torch.device("cpu")
        model = torch.nn.Linear(2, 2).to(device)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = TokenLRScheduler(
            optimizer, WarmupCosineConfig(1e-3, 1e-4, 1, 10)
        )
        sampler = StatefulDistributedSampler(4)
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = save_full_checkpoint(
                Path(temporary),
                step=0,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                compatibility={"data": "one"},
            )
            with self.assertRaisesRegex(ValueError, "compatibility"):
                load_full_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    compatibility={"data": "two"},
                    device=device,
                )

    def test_continuation_matches_uninterrupted_training_exactly(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(123)
        model = torch.nn.Linear(3, 2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = WarmupCosineConfig(1e-3, 1e-4, 2, 20)
        scheduler = TokenLRScheduler(optimizer, config)
        sampler = StatefulDistributedSampler(8, seed=17)
        samples = [torch.tensor([[i, i + 1, i + 2]], dtype=torch.float32) for i in range(8)]

        def take_step(current_model, current_optimizer, current_scheduler, iterator):
            sample = samples[next(iterator)]
            current_optimizer.zero_grad(set_to_none=True)
            loss = current_model(sample).square().mean()
            loss.backward()
            current_scheduler.step(1)
            current_optimizer.step()

        iterator = iter(sampler)
        take_step(model, optimizer, scheduler, iterator)
        take_step(model, optimizer, scheduler, iterator)
        compatibility = {"test": "exact-continuation"}

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, _ = save_full_checkpoint(
                Path(temporary),
                step=2,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                compatibility=compatibility,
            )
            take_step(model, optimizer, scheduler, iterator)
            take_step(model, optimizer, scheduler, iterator)
            expected = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }

            torch.manual_seed(999)
            resumed_model = torch.nn.Linear(3, 2).to(device)
            resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
            resumed_scheduler = TokenLRScheduler(resumed_optimizer, config)
            resumed_sampler = StatefulDistributedSampler(8, seed=17)
            load_full_checkpoint(
                checkpoint,
                model=resumed_model,
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                sampler=resumed_sampler,
                compatibility=compatibility,
                device=device,
            )
            resumed_iterator = iter(resumed_sampler)
            take_step(
                resumed_model, resumed_optimizer, resumed_scheduler, resumed_iterator
            )
            take_step(
                resumed_model, resumed_optimizer, resumed_scheduler, resumed_iterator
            )
            for name, value in resumed_model.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
