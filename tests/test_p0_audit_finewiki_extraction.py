import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p0_audit_finewiki_extraction import EMPTY_LANGUAGE  # noqa: E402


class ExtractionPatternTests(unittest.TestCase):
    def test_empty_annotation(self):
        self.assertIsNotNone(EMPTY_LANGUAGE.search("瑞士（德語：；法語：）"))
        self.assertIsNotNone(EMPTY_LANGUAGE.search("电影（英語：）"))

    def test_nonempty_annotation(self):
        self.assertIsNone(EMPTY_LANGUAGE.search("瑞士（德語：Schweiz；法語：Suisse）"))


if __name__ == "__main__":
    unittest.main()
