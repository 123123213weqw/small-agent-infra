import unittest

from experiments.phase0_b21_report import NAMES, SPLITS, TASKS, build_report


def fake_run(name: str) -> dict:
    success = 0.5
    regret = 0.02
    if name == "E_joint_full":
        success, regret = 0.7, 0.01
    elif name == "F_joint_partial":
        success, regret = 0.6, 0.015
    rollout = {}
    offline = {}
    efficiency = {}
    by_task = {}
    for split in SPLITS:
        rollout[f"{split}/model"] = {
            "success": success,
            "required_recall": success,
            "eviction_regret": regret,
            "stale_value_rate": 0.0,
        }
        rollout[f"{split}/fifo"] = {
            "success": 0.1,
            "required_recall": 0.1,
            "eviction_regret": 0.04,
            "stale_value_rate": 0.0,
        }
        rollout[f"{split}/oracle"] = {
            "success": 1.0,
            "required_recall": 1.0,
            "eviction_regret": 0.0,
            "stale_value_rate": 0.0,
        }
        offline[split] = {
            "mse": 0.01,
            "pairwise_accuracy": 0.8,
            "argmin_correct": 0.8,
            "regret": 0.01,
        }
        efficiency[split] = {"groups_per_second": 100.0, "peak_memory_mib": 1000.0}
        for task in TASKS:
            by_task[f"{split}/{task}"] = {
                "success": success,
                "required_recall": success,
                "eviction_regret": regret,
                "stale_value_rate": 0.0,
            }
    return {
        "architecture": "joint",
        "parameters": 100,
        "rollout": rollout,
        "offline": offline,
        "offline_efficiency": efficiency,
        "rollout_by_task": by_task,
    }


class Phase0B21ReportTests(unittest.TestCase):
    def test_gate_accepts_clear_full_set_improvement(self):
        summary, rows, markdown = build_report({name: fake_run(name) for name in NAMES})
        self.assertTrue(summary["pass"])
        self.assertEqual(summary["decision"], "continue_to_stage2")
        self.assertEqual(summary["post_hoc_winner"], "E_joint_full")
        self.assertEqual(len(summary["post_hoc_diagnostics"]), len(NAMES) - 1)
        self.assertEqual(len(rows), len(NAMES) * len(SPLITS))
        self.assertIn("进入第二阶段", markdown)
        self.assertIn("诊断性消融", markdown)


if __name__ == "__main__":
    unittest.main()
