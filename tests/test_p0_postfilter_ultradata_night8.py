"""Offline regression checks for the conservative postfilter audit."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p0_postfilter_ultradata_night8.py"
spec = importlib.util.spec_from_file_location("p0_postfilter_ultradata_night8", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PostfilterTests(unittest.TestCase):
    def test_normalization_matches_p0_outer_trim_and_internal_indentation(self):
        raw = "\x00  def f():  \r\n      pass  \r\n\r\n\r\n\r\n"
        self.assertEqual(module.normalized_text(raw), "def f():\n      pass")

    def test_vendor_path_is_review_flag(self):
        flags = module.quality_flags(module.SOURCE_CODE, "def f():\n    return 1\n",
                                     "owner/repo/src/thirdparty/pkg/file.py")
        self.assertIn("code_vendored_path", flags)
        flags = module.quality_flags(module.SOURCE_CODE, "import external\ndef f():\n    return 1\n",
                                     "owner/repo/src/my_file.py")
        self.assertNotIn("code_vendored_path", flags)

    def test_python2_parse_failure_not_hard_reject(self):
        flags = module.quality_flags(module.SOURCE_CODE, "print 'hello'\n")
        self.assertIn("code_python3_parse_failure", flags)


if __name__ == "__main__":
    unittest.main()
