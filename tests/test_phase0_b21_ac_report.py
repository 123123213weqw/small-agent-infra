import unittest

from experiments.phase0_b21_ac_report import SPLITS, TASKS, build_report


def fake_run(model: str) -> dict:
    success = 0.5 if model == "A" else 0.6
    regret = 0.02 if model == "A" else 0.01
    rollout = {}
    by_task = {}
    efficiency = {}
    for split in SPLITS:
        rollout[f"{split}/model"] = {
            "success": success,
            "required_recall": success,
            "eviction_regret": regret,
            "stale_value_rate": 0.0,
        }
        efficiency[split] = {
            "groups_per_second": 100.0,
            "peak_memory_mib": 1000.0,
        }
        for task in TASKS:
            by_task[f"{split}/{task}"] = {
                "success": success,
                "required_recall": success,
                "eviction_regret": regret,
                "stale_value_rate": 0.0,
            }
    return {
        "parameters": 100,
        "rollout": rollout,
        "rollout_by_task": by_task,
        "offline_efficiency": efficiency,
    }


class Phase0B21ACReportTests(unittest.TestCase):
    def test_clear_paired_improvement_passes(self):
        runs = {seed: {model: fake_run(model) for model in ("A", "C")} for seed in range(3)}
        summary, rows, markdown = build_report(runs)
        self.assertTrue(summary["pass"])
        self.assertEqual(summary["decision"], "pass_b2_1_seed_confirmation")
        self.assertEqual(len(rows), 3 * 2 * len(SPLITS))
        self.assertIn("多种子确认结论：通过", markdown)


if __name__ == "__main__":
    unittest.main()
