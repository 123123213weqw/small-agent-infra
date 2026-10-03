import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("full8", ROOT / "scripts/p0_wave2_full8_bundle_v1.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def budget():
    return M.transition_budget(508262, 200, 4800, 9825600, 8, 1, 4, 2048)


def configs():
    return [json.loads((ROOT / "configs/runs" / f).read_text()) for f in
            ("base_1b_prior_wave2_internal_1040m_6l40_v1.json", "base_1b_prior_wave2_internal_full_8l40_v1.json")]


class Full8Tests(unittest.TestCase):
    def test_keeps_actual_update_step_tokens_and_one_pass(self):
        b = budget()
        self.assertEqual(b["source_step"], 200)
        self.assertEqual(b["end_step"], 15934)
        self.assertEqual(b["loss_tokens_per_step"], 65504)
        self.assertEqual(b["total_loss_tokens"], 1040465536)
        self.assertEqual(b["tail_repeated_samples"], 26)

    def test_no_unaligned_or_wrong_token_cursor(self):
        for position, tokens in ((4801, 9825600), (4800, 0), (508262, 1040412314)):
            with self.assertRaises(ValueError):
                M.transition_budget(508262, 200, position, tokens, 8, 1, 4, 2048)

    def test_config_changes_only_explicit_allowed_fields(self):
        a, b = configs()
        M.validate_config(a, b, budget())
        self.assertEqual(a["scheduler"], b["scheduler"])
        self.assertEqual(b["checkpoint"]["keep_full"], 3)

    def test_scheduler_or_data_change_rejected(self):
        a, b = configs()
        for key, value in (("data_dir", "other"), ("seed", 0), ("scheduler", {})):
            c = copy.deepcopy(b)
            c[key] = value
            with self.assertRaises(ValueError):
                M.validate_config(a, c, budget())

    def test_optimizer_step_not_renumbered_to_eight_equivalent_150(self):
        a, b = configs()
        b["max_steps"] = 15884
        with self.assertRaises(ValueError):
            M.validate_config(a, b, budget())

    def test_report_checks_restored_token_offset(self):
        b = budget()
        rows = [{"event": "resume", "step": 200, "tokens_seen": 9825600},
                {"event": "train", "step": 15934, "tokens_seen": 1040465536,
                 "loss": 4., "grad_norm": 1., "learning_rate": .00002, "tokens_per_second": 23000.},
                {"event": "validation", "step": 15934, "loss": 4.1}]
        self.assertEqual(M.full_summary(rows, b)["final_validation"]["loss"], 4.1)
        rows[1]["tokens_seen"] = 15934 * 65504
        with self.assertRaises(ValueError):
            M.full_summary(rows, b)

    def test_report_rejects_missing_resume_and_final_validation(self):
        b = budget()
        rows = [{"event": "train", "step": 15934, "tokens_seen": 1040465536,
                 "loss": 4., "grad_norm": 1., "learning_rate": .00002, "tokens_per_second": 23000.}]
        with self.assertRaises(ValueError):
            M.full_summary(rows, b)


if __name__ == "__main__":
    unittest.main()
