import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / "scripts" / "p0_zh_prefilter_ab.py"
spec = importlib.util.spec_from_file_location("p0_zh_prefilter_ab", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PrefilterTests(unittest.TestCase):
    def test_host_split_is_stable(self):
        self.assertEqual(module.split_for_host("example.com"), module.split_for_host("example.com"))

    def test_calibration_never_exceeds_false_reject_budget(self):
        scores = [.91, .8, .7, .2, .1, .99]
        clean = [True, True, True, True, True, False]
        threshold = module.threshold_at_clean_budget(scores, clean, .2)
        self.assertLessEqual(sum(s > threshold for s, c in zip(scores, clean) if c), 1)

    def test_spam_rule_requires_repeated_evidence(self):
        self.assertFalse(module.rule_reject("一篇关于博彩行业监管的文章"))
        self.assertTrue(module.rule_reject("六合彩 现金网 真人娱乐城"))


if __name__ == "__main__":
    unittest.main()
