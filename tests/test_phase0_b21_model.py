import json
import unittest
from pathlib import Path

import torch

from experiments.phase0_b21_model import GroupCollator, UtilitySetModel, parameter_count, utility_losses
from experiments.phase0_b21_train import resolve_seed_config


ROOT = Path(__file__).resolve().parents[1]


def config(name: str) -> dict:
    return json.loads((ROOT / "configs" / "b2_1_stage1" / f"{name}.json").read_text())


def group(count: int = 5) -> dict:
    utilities = [0.0] * count
    utilities[1] = 1.0
    return {
        "split": "test",
        "episode_id": "e1",
        "decision_id": 1,
        "task": "delayed_query",
        "query_visibility": "announced_query",
        "capacity": count - 1,
        "hard_negative_count": 0,
        "goal": "回答 test_lookup_1",
        "recent_context": [{"text": "近期记录", "event_index": 3}],
        "records": [
            {
                "uid": f"e1:r{i}",
                "text": f"记录 test_k{i}=test_v{i}",
                "event_index": i,
                "relative_age": count - 1 - i,
                "is_candidate": i == count - 1,
                "source": "tool",
            }
            for i in range(count)
        ],
        "candidate_index": count - 1,
        "utilities": utilities,
        "oracle_eviction_indices": [i for i, value in enumerate(utilities) if value == 0],
    }


class Phase0B21ModelTests(unittest.TestCase):
    def test_seed_roles_are_separated_with_legacy_fallback(self):
        legacy = resolve_seed_config({"seed": 7})
        self.assertEqual(legacy["model_seed"], 7)
        self.assertEqual(legacy["data_seed"], 7)
        self.assertEqual(legacy["evaluation_seed"], 7)

        paired = resolve_seed_config(
            {"seed": 2, "model_seed": 2, "data_seed": 0, "evaluation_seed": 0}
        )
        self.assertEqual(paired["model_seed"], 2)
        self.assertEqual(paired["data_seed"], 0)
        self.assertEqual(paired["evaluation_seed"], 0)

    def test_collator_pads_variable_sets(self):
        batch = GroupCollator("joint", record_bytes=32, context_bytes=48)([group(5), group(9)])
        self.assertEqual(tuple(batch["record_ids"].shape), (2, 9, 32))
        self.assertEqual(batch["set_mask"].sum(dim=1).tolist(), [5, 9])
        self.assertEqual(batch["oracle_mask"].sum(dim=1).tolist(), [4, 8])

    def test_all_models_have_matched_parameter_counts(self):
        names = (
            "A_pointwise",
            "B_joint_mse",
            "C_joint_rank",
            "D_joint_evict",
            "E_joint_full",
            "F_joint_partial",
        )
        counts = [parameter_count(UtilitySetModel(config(name))) for name in names]
        self.assertLessEqual(max(counts) / min(counts), 1.01)

    def test_loss_components_are_finite(self):
        cfg = config("E_joint_full")
        batch = GroupCollator("joint", record_bytes=32, context_bytes=48)([group(5), group(9)])
        predictions = torch.full_like(batch["utilities"], 0.2)
        losses = utility_losses(predictions, batch, cfg)
        self.assertEqual(set(losses), {"total", "regression", "ranking", "eviction"})
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertGreater(float(losses["ranking"]), 0.0)

    def test_partial_attention_limits_each_valid_query(self):
        model = UtilitySetModel(config("F_joint_partial"))
        set_mask = torch.tensor([[True] * 9 + [False] * 4])
        mask = model._partial_attention_mask(set_mask)
        one_head = mask[0]
        self.assertTrue(all(int((~one_head[i]).sum()) <= 4 for i in range(9)))
        self.assertTrue(all(int((~one_head[i]).sum()) == 1 for i in range(9, 13)))


if __name__ == "__main__":
    unittest.main()
