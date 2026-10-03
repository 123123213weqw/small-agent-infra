"""Offline tests for Label Studio prediction conversion."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "import_labelstudio_predictions.py"
spec = importlib.util.spec_from_file_location("import_labelstudio_predictions", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PredictionConversionTests(unittest.TestCase):
    def test_choice_issue_and_reason(self):
        result = module.to_prediction({
            "decision": "剔除",
            "quality": "低",
            "issues": ["模板化或重复"],
            "reason": "主体是网页导航。",
        })
        self.assertEqual([r["from_name"] for r in result], ["decision", "quality", "issues", "notes"])
        self.assertEqual(result[0]["value"], {"choices": ["剔除"]})
        self.assertEqual(result[3]["value"], {"text": ["主体是网页导航。"]})

    def test_empty_optional_fields_omitted(self):
        result = module.to_prediction({"decision": "保留", "quality": "高", "issues": [], "reason": ""})
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
