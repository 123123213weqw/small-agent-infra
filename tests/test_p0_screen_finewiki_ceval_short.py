import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p0_screen_finewiki_ceval_short import eligible_question, matches_two_options  # noqa: E402


class CevShortScreenTest(unittest.TestCase):
    def test_requires_two_distinct_options_after_exact_stem(self):
        stem = "关于战略管理表述错误的是"
        options = ["选项甲描述完整", "选项乙描述完整", "丙", "丁"]
        self.assertTrue(eligible_question(stem, options))
        self.assertTrue(matches_two_options("前言" + stem + "a选项甲描述完整b选项乙描述完整", stem, options))
        self.assertFalse(matches_two_options(stem + "a选项甲描述完整", stem, options))
        self.assertFalse(matches_two_options("选项甲描述完整选项乙描述完整" + stem, stem, options))
        self.assertFalse(matches_two_options(stem + "选项甲描述完整选项甲描述完整", stem,
                                            [options[0], options[0]]))

    def test_rejects_too_short_stem(self):
        self.assertFalse(eligible_question("莎士比亚的李尔王", ["一种著名戏剧", "一种自然规律"]))


if __name__ == "__main__":
    unittest.main()
