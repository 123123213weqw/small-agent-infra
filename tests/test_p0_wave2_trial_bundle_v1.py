import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("trial_bundle", ROOT / "scripts/p0_wave2_trial_bundle_v1.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def events():
    rows = [{"event": "validation", "step": 0, "loss": 10.5}]
    for step in range(10, 201, 10):
        if step == 110:
            rows.append({"event": "resume", "step": 100, "tokens_seen": 4912800})
        rows.append({"event": "train", "step": step, "tokens_seen": step * 49128,
                     "loss": 8.5, "grad_norm": 1., "learning_rate": .0001,
                     "tokens_per_second": 20000., "peak_allocated_gib": 30., "peak_reserved_gib": 35.})
    rows.append({"event": "validation", "step": 200, "loss": 8.})
    return rows


class BundleTests(unittest.TestCase):
    def test_six_card_full_budget(self):
        b = M.budget(508262, 6, 1, 4, 2048)
        self.assertEqual(b["steps"], 21178)
        self.assertEqual(b["loss_tokens_per_step"], 49128)
        self.assertEqual(b["loss_tokens"], 1040432784)

    def test_config_keeps_full_schedule(self):
        c = json.loads((ROOT / "configs/runs/base_1b_prior_wave2_internal_1040m_6l40_v1.json").read_text())
        b = M.budget(c["mixture_block_size"], 6, c["micro_batch_size"], c["gradient_accumulation_steps"], c["context_length"])
        self.assertEqual(c["max_steps"], b["steps"])
        self.assertEqual(c["scheduler"]["total_tokens"], b["loss_tokens"])
        self.assertGreater(c["scheduler"]["warmup_tokens"], 200*b["loss_tokens_per_step"])
        self.assertFalse(c["distributable"])

    def test_report_requires_real_resume_and_validation(self):
        s = M.trial_summary(events(), 49128)
        self.assertEqual(s["loss_tokens"], 9825600)
        self.assertEqual(len(s["resume_events"]), 1)
        self.assertEqual(s["median_last_windows_loss_tokens_per_second"], 20000.)

    def test_missing_resume_rejected(self):
        with self.assertRaises(ValueError):
            M.trial_summary([e for e in events() if e["event"] != "resume"], 49128)

    def test_bad_accounting_rejected(self):
        v = events()
        v[1]["tokens_seen"] += 1
        with self.assertRaises(ValueError):
            M.trial_summary(v, 49128)

    def test_nonfinite_rejected(self):
        v = copy.deepcopy(events())
        v[1]["grad_norm"] = float("nan")
        with self.assertRaises(ValueError):
            M.trial_summary(v, 49128)

    def test_missing_final_validation_rejected(self):
        with self.assertRaises(ValueError):
            M.trial_summary(events()[:-1], 49128)


if __name__ == "__main__":
    unittest.main()
