"""Offline tests for the reusable annotation CLI."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "annotate_jsonl.py"
spec = importlib.util.spec_from_file_location("annotate_jsonl", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AnnotateJsonlTests(unittest.TestCase):
    def test_json_array_and_nested_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps([{"data": {"text": "abc", "source": "web", "row": 3}}]))
            records = module.read_records(path)
            job = module.make_job(records[0], 0, "data.text", ["data.source", "data.row"], ["data.row"], "p", "s", "m")
            self.assertEqual(job["source_id"], '["web",3]')
            self.assertEqual(job["source"], {"data.row": 3})
            self.assertEqual(job["text_sha256"], module.digest("abc"))

    def test_jsonl_and_resume_key(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jsonl"
            source.write_text('{"excerpt":"one","id":1}\n{"excerpt":"two","id":2}\n')
            records = module.read_records(source)
            self.assertEqual(len(records), 2)
            job = module.make_job(records[0], 0, "excerpt", ["id"], [], "p", "s", "m")
            output = Path(directory) / "output.jsonl"
            output.write_text(module.canonical_json({**job, "status": "ok"}) + "\n")
            self.assertIn(module.job_key(job), module.completed_keys(output))
            changed_prompt = {**job, "prompt_sha256": "new"}
            self.assertNotIn(module.job_key(changed_prompt), module.completed_keys(output))

    def test_missing_text_is_error(self):
        with self.assertRaises(KeyError):
            module.make_job({"other": "x"}, 0, "excerpt", [], [], "p", "s", "m")

    def test_temperature_changes_resume_key(self):
        record = {"excerpt": "sample", "id": 1}
        default = module.make_job(record, 0, "excerpt", ["id"], [], "p", "s", "m")
        zero = module.make_job(record, 0, "excerpt", ["id"], [], "p", "s", "m", 0.0)
        self.assertNotEqual(module.job_key(default), module.job_key(zero))

    def test_local_inference_settings_change_resume_key(self):
        record = {"excerpt": "sample", "id": 1}
        a = module.make_job(record, 0, "excerpt", ["id"], [], "p", "s", "m", 0.0, "settings-a")
        b = module.make_job(record, 0, "excerpt", ["id"], [], "p", "s", "m", 0.0, "settings-b")
        self.assertNotEqual(module.job_key(a), module.job_key(b))

    def test_json_schema_generation_copy_keeps_full_validator_schema(self):
        schema = {"$schema": "draft", "type": "object", "properties": {"codes": {"type": "array", "uniqueItems": True}}}
        result = module.make_response_format(schema, "json_schema")
        self.assertEqual(result["type"], "json_schema")
        self.assertNotIn("uniqueItems", result["json_schema"]["schema"]["properties"]["codes"])
        self.assertTrue(schema["properties"]["codes"]["uniqueItems"])

    def test_document_id_match(self):
        self.assertTrue(module.document_id_matches_source({"document_id": "abc"}, '["abc"]'))
        self.assertFalse(module.document_id_matches_source({"document_id": "other"}, '["abc"]'))
        self.assertFalse(module.document_id_matches_source({"document_id": "abc"}, 'not-json'))


if __name__ == "__main__":
    unittest.main()
