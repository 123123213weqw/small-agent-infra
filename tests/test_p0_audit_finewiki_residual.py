import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p0_audit_finewiki_residual import EMPTY_PARENS, OTHER_EMPTY_LANGUAGE  # noqa: E402


class ResidualPatternsTests(unittest.TestCase):
    def test_obvious_missing_extracts(self):
        self.assertIsNotNone(EMPTY_PARENS.search("恆星（）是一種天體"))
        self.assertIsNotNone(OTHER_EMPTY_LANGUAGE.search("作品（英文：）"))
        self.assertIsNotNone(OTHER_EMPTY_LANGUAGE.search("土耳其語：，縮寫"))

    def test_not_normal_content(self):
        self.assertIsNone(EMPTY_PARENS.search("函數()"))
        self.assertIsNone(OTHER_EMPTY_LANGUAGE.search("英文：example"))


if __name__ == "__main__":
    unittest.main()
