import hashlib
import importlib.util
from pathlib import Path
import unittest
import sys

PATH = Path(__file__).resolve().parents[1] / 'scripts/training_eval_evalplus_compat_v6.py'
spec = importlib.util.spec_from_file_location('evalplus_compat_test_v6', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.path.insert(0, str(PATH.parent))
from finalize_training_quality_recovery_v6 import capability_pair


class CompatibilityTests(unittest.TestCase):
    def test_accounting_only(self):
        raw = ('prefix\n' + m.OLD + '\nsuffix').encode()
        changed = m.patched_source(raw, hashlib.sha256(raw).hexdigest())
        self.assertEqual(changed, 'prefix\n' + m.NEW + '\nsuffix')
        self.assertIn('assert abs(_poly(*inp, out)) <= atol', changed)
        self.assertEqual(changed.replace(m.NEW, m.OLD), raw.decode())

    def test_crlf_locked_before_normalization(self):
        raw = m.OLD.replace('\n', '\r\n').encode()
        self.assertEqual(m.patched_source(raw, hashlib.sha256(raw).hexdigest()), m.NEW)

    def test_unknown_dependency_fails_closed(self):
        with self.assertRaises(ValueError):
            m.patched_source(m.OLD.encode())

    def test_missing_or_duplicate_branch_fails_closed(self):
        for text in ['none', m.OLD + m.OLD]:
            raw = text.encode()
            with self.assertRaises(ValueError):
                m.patched_source(raw, hashlib.sha256(raw).hexdigest())

    def test_floor_is_inconclusive_not_a_certificate(self):
        scores = {'gsm8k/exact_match,strict-match': 0., 'humaneval/base_pass@1': 0.,
                  'arc_easy/acc,none': .5, 'generation/repeated_4gram_fraction': .2}
        row = capability_pair(1, scores, scores)
        self.assertTrue(row['diagnostic_gate_passed'])
        self.assertEqual(len(row['floor_metrics_inconclusive']), 2)
        self.assertFalse(row['floor_metrics_certified'])

    def test_real_score_regression_not_masked(self):
        left = {'arc_easy/acc,none': .5, 'generation/repeated_4gram_fraction': .2}
        self.assertFalse(capability_pair(1, left, dict(left, **{'arc_easy/acc,none': .46}))['diagnostic_gate_passed'])
        self.assertFalse(capability_pair(1, left, dict(left, **{'generation/repeated_4gram_fraction': .251}))['diagnostic_gate_passed'])

    def test_invalid_metrics_fail_closed(self):
        scores = {'arc_easy/acc,none': .5, 'generation/repeated_4gram_fraction': .2}
        for value in [float('nan'), float('inf'), -.1, 1.1]:
            with self.assertRaises(ValueError):
                capability_pair(1, scores, dict(scores, **{'arc_easy/acc,none': value}))
        with self.assertRaises(ValueError):
            capability_pair(1, scores, {})


if __name__ == '__main__':
    unittest.main()
