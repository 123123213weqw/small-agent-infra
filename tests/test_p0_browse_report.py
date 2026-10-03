import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p0_browse_report", ROOT / "scripts" / "p0_browse_report.py"
)
REPORT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REPORT)


class P0BrowseReportTests(unittest.TestCase):
    def test_audit_rank_is_stable_and_seeded(self):
        self.assertEqual(REPORT.audit_rank("a", "doc"), REPORT.audit_rank("a", "doc"))
        self.assertNotEqual(REPORT.audit_rank("a", "doc"), REPORT.audit_rank("b", "doc"))

    def test_choose_samples_is_deterministic_and_stratified(self):
        rows = [
            {"domain": domain, "document_id": f"{domain}-{number}"}
            for domain in ("zh", "code")
            for number in range(20)
        ]
        first = REPORT.choose_samples(rows, 3, "seed")
        second = REPORT.choose_samples(reversed(rows), 3, "seed")
        self.assertEqual(first, second)
        self.assertEqual(sum(row["domain"] == "zh" for row in first), 3)
        self.assertEqual(sum(row["domain"] == "code" for row in first), 3)

    def test_preview_marks_omitted_middle(self):
        preview = REPORT.preview_text("x" * 5000, head_chars=10, tail_chars=10)
        self.assertIn("中间省略 4,980 个字符", preview)
        self.assertEqual(preview.count("x"), 20)


if __name__ == "__main__":
    unittest.main()
