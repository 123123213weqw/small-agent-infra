"""Offline integrity tests for the fixed 400-record audit split."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data_audit_v2_eval.py"
spec = importlib.util.spec_from_file_location("data_audit_v2_eval", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DataAuditV2EvalTests(unittest.TestCase):
    def setUp(self):
        self.tasks = {
            (source, idx): {"data": {"source_file": source, "source_row_idx": idx, "text": f"{source} {idx}"}}
            for source in ("a.jsonl", "b.jsonl") for idx in range(4)
        }

    def test_prepare_holdout_excludes_dev(self):
        cases = {("a.jsonl", 0): "保留", ("b.jsonl", 0): "剔除"}
        dev = set(cases)
        regression, holdout = module.prepare(self.tasks, cases, dev, per_source=2, seed=4)
        self.assertEqual(len(regression), 2)
        self.assertEqual(len(holdout["ids"]), 4)
        self.assertTrue(all((x["source_file"], x["source_row_idx"]) not in dev for x in holdout["ids"]))

    def test_regression_must_be_development_data(self):
        with self.assertRaisesRegex(ValueError, "development sample"):
            module.prepare(self.tasks, {("a.jsonl", 0): "保留"}, set(), per_source=1, seed=4)

    def test_prediction_text_hash_must_match(self):
        task = self.tasks[("a.jsonl", 0)]
        prediction = {
            "source": {"data.source_file": "a.jsonl", "data.source_row_idx": 0},
            "source_id": '["a.jsonl",0]',
            "text_sha256": "incorrect",
            "status": "ok",
            "prompt_sha256": "p", "schema_sha256": "s", "model": "m",
            "annotation": {"decision": "保留"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.jsonl"
            path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "text hash mismatch"):
                module.load_predictions(path, {module.task_key(task): task})


if __name__ == "__main__":
    unittest.main()
