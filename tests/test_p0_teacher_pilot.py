import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p0_teacher_pilot", ROOT / "scripts" / "p0_teacher_pilot.py"
)
PILOT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PILOT)


class P0TeacherPilotTests(unittest.TestCase):
    def test_cheap_code_path_rejection(self):
        record = {
            "domain": "code_other",
            "source_locator": "repo=x|path=node_modules/a/index.js",
            "text": "function useful() { return 1; }",
        }
        self.assertEqual(PILOT.cheap_reject(record), ["vendored_dependency_path"])

    def test_paywall_and_feedback_rejection(self):
        paywall = {
            "domain": "math_en", "source_locator": "url=x",
            "text": "Want to see the full answer? See Solution\n*Response times may vary",
        }
        self.assertIn("paywall_or_answer_preview", PILOT.cheap_reject(paywall))
        feedback = {
            "domain": "general_zh", "source_locator": "url=x",
            "text": "不感兴趣\n广告软文\n标题夸张\n感谢您的反馈",
        }
        self.assertEqual(PILOT.cheap_reject(feedback), ["embedded_recommendation_feedback_widget"])

    def test_long_text_selection_is_deterministic(self):
        text = "0123456789" * 11_000
        first, meta = PILOT.review_text(text, "a" * 64)
        second, second_meta = PILOT.review_text(text, "a" * 64)
        self.assertEqual(first, second)
        self.assertEqual(meta, second_meta)
        self.assertEqual(meta["mode"], "head_middle_tail")
        self.assertTrue(first.startswith(text[:8000]))
        self.assertTrue(first.endswith(text[-8000:]))

    def test_select_all_eligible_is_deterministic_and_prefiltered(self):
        records = [
            {
                "document_id": "2" * 64,
                "domain": "general_en",
                "source_locator": "url=two",
                "text": "A useful document.",
            },
            {
                "document_id": "1" * 64,
                "domain": "code_other",
                "source_locator": "repo=x|path=node_modules/a.js",
                "text": "function x() {}",
            },
            {
                "document_id": "3" * 64,
                "domain": "math_en",
                "source_locator": "url=three",
                "text": "A complete proof.",
            },
        ]
        first = PILOT.select_all_eligible(records, "seed")
        second = PILOT.select_all_eligible(reversed(records), "seed")
        self.assertEqual(first, second)
        self.assertEqual({row["document_id"] for row in first}, {"2" * 64, "3" * 64})

    def test_gate_is_strict(self):
        good = {
            "decision": "keep", "confidence": 0.9, "quality": 4,
            "completeness": 4, "educational_value": 3, "format_integrity": 4,
            "recommended_bucket": "general_en",
        }
        self.assertTrue(PILOT.passes_gate(good))
        self.assertFalse(PILOT.passes_gate({**good, "confidence": 0.84}))
        self.assertFalse(PILOT.passes_gate({**good, "completeness": 3}))


if __name__ == "__main__":
    unittest.main()
