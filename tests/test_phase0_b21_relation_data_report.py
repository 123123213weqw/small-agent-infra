import unittest

from experiments.phase0_b21_relation_data_report import SPLITS, TASKS, build_report


def fake_run(augmented: bool) -> dict:
    success = 0.85 if augmented else 0.80
    regret = 0.01 if augmented else 0.02
    rollout = {
        f"{split}/model": {
            "success": success,
            "required_recall": success,
            "eviction_regret": regret,
            "stale_value_rate": 0.0,
        }
        for split in SPLITS
    }
    by_task = {}
    for split in SPLITS:
        for task in TASKS:
            task_value = 0.60 if augmented and task == "relation_chain" else 0.80
            if not augmented and task == "relation_chain":
                task_value = 0.20
            by_task[f"{split}/{task}"] = {
                "success": task_value,
                "required_recall": task_value,
                "eviction_regret": regret,
                "stale_value_rate": 0.0,
            }
    return {"parameters": 100, "rollout": rollout, "rollout_by_task": by_task}


class Phase0B21RelationDataReportTests(unittest.TestCase):
    def test_clear_data_only_improvement_passes(self):
        runs = {
            seed: {"baseline": fake_run(False), "augmented": fake_run(True)}
            for seed in range(3)
        }
        summary, rows, markdown = build_report(runs)
        self.assertTrue(summary["pass"])
        self.assertEqual(summary["decision"], "relation_data_fix_passed")
        self.assertEqual(len(rows), 3 * 2 * len(SPLITS))
        self.assertIn("数据增强结论：通过", markdown)


if __name__ == "__main__":
    unittest.main()
