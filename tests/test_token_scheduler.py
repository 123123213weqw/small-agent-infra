from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_agent.training.scheduler import (  # noqa: E402
    TokenLRScheduler,
    WarmupCosineConfig,
    warmup_cosine_lr,
)


class TokenSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = WarmupCosineConfig(
            maximum_learning_rate=2e-4,
            minimum_learning_rate=2e-5,
            warmup_tokens=100,
            total_tokens=1000,
        )

    def test_warmup_and_cosine_boundaries(self) -> None:
        self.assertEqual(warmup_cosine_lr(0, self.config), 0.0)
        self.assertAlmostEqual(warmup_cosine_lr(50, self.config), 1e-4)
        self.assertAlmostEqual(warmup_cosine_lr(100, self.config), 2e-4)
        self.assertAlmostEqual(warmup_cosine_lr(1000, self.config), 2e-5)
        self.assertAlmostEqual(warmup_cosine_lr(2000, self.config), 2e-5)

    def test_state_restore_reproduces_lr(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([parameter], lr=9.0)
        scheduler = TokenLRScheduler(optimizer, self.config)
        scheduler.step(250)
        state = scheduler.state_dict()

        other_parameter = torch.nn.Parameter(torch.ones(()))
        other_optimizer = torch.optim.SGD([other_parameter], lr=3.0)
        restored = TokenLRScheduler(other_optimizer, self.config)
        restored.load_state_dict(state)
        self.assertEqual(restored.tokens_seen, 250)
        self.assertEqual(restored.learning_rate, scheduler.learning_rate)

    def test_config_drift_is_rejected(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(()))
        scheduler = TokenLRScheduler(torch.optim.SGD([parameter], lr=1.0), self.config)
        changed = WarmupCosineConfig(2e-4, 2e-5, 101, 1000)
        restored = TokenLRScheduler(torch.optim.SGD([parameter], lr=1.0), changed)
        with self.assertRaisesRegex(ValueError, "config"):
            restored.load_state_dict(scheduler.state_dict())


if __name__ == "__main__":
    unittest.main()
