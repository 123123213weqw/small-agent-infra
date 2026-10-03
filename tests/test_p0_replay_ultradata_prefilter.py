"""Conservative source-path filter regression tests."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p0_replay_ultradata_prefilter.py"
spec = importlib.util.spec_from_file_location("p0_replay_ultradata_prefilter", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PrefilterTests(unittest.TestCase):
    def code(self, path):
        return {"source_id": module.CODE_SOURCE, "source_locator": "owner/repo/" + path}

    def test_environment_copy_is_hard_rejected(self):
        for path in ("venv/Lib/site-packages/pkg/a.py", ".eggs/pkg/a.py",
                     "src/third_party/pkg/a.py", "src/_vendor/pkg/a.py"):
            self.assertIn("vendored_environment_path", module.classify_hard(self.code(path), "x = 1"))

    def test_original_code_is_not_rejected_for_import_or_name(self):
        self.assertEqual([], module.classify_hard(self.code("src/site_packages_utils.py"),
                                                  "import external_package\nx = 1"))
        self.assertEqual([], module.classify_hard(self.code("src/vendor_tools.py"), "x = 1"))

    def test_repository_name_does_not_trigger_path_rule(self):
        row = {"source_id": module.CODE_SOURCE,
               "source_locator": "site-packages/vendor/src/my_project.py"}
        self.assertEqual([], module.classify_hard(row, "x = 1"))

    def test_python3_parse_failure_is_review_only(self):
        hard, flags = module.classify(self.code("src/example.py"), "def f(:\n")
        self.assertEqual([], hard)
        self.assertIn("python3_parse_failure", flags)


if __name__ == "__main__":
    unittest.main()
