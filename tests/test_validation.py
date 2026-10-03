from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_agent.evaluation import evaluate_causal_lm  # noqa: E402


class DummyModel(torch.nn.Module):
    def forward(self, input_ids, labels, attention_mask, use_cache):
        del labels, attention_mask, use_cache
        # Make batch losses different so token-weighting is observable.
        return SimpleNamespace(loss=input_ids[0, 0].float())


class ValidationTests(unittest.TestCase):
    def test_loss_is_weighted_by_valid_causal_targets(self) -> None:
        model = DummyModel()
        model.train()
        batches = [
            {
                "input_ids": torch.tensor([[2, 3, 4, 5]]),
                "labels": torch.tensor([[2, 3, 4, 5]]),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([[4, 5, 0, 0]]),
                "labels": torch.tensor([[4, 5, -100, -100]]),
                "attention_mask": torch.tensor([[1, 1, 0, 0]]),
            },
        ]
        result = evaluate_causal_lm(
            model, batches, torch.device("cpu"), autocast_dtype=None
        )
        self.assertEqual(result.predicted_tokens, 4)
        self.assertEqual(result.batches, 2)
        self.assertAlmostEqual(result.loss, (2.0 * 3 + 4.0 * 1) / 4)
        self.assertTrue(model.training)

    def test_zero_targets_is_rejected(self) -> None:
        batch = {
            "input_ids": torch.tensor([[2]]),
            "labels": torch.tensor([[2]]),
            "attention_mask": torch.tensor([[1]]),
        }
        with self.assertRaisesRegex(ValueError, "zero"):
            evaluate_causal_lm(
                DummyModel(), [batch], torch.device("cpu"), autocast_dtype=None
            )


if __name__ == "__main__":
    unittest.main()
