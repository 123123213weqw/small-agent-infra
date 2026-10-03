import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p0_data_pipeline", ROOT / "scripts" / "p0_data_pipeline.py"
)
PIPELINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PIPELINE)


class P0DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipeline_config = json.loads(
            (ROOT / "configs" / "data" / "p0_1m_smoke_v1.json").read_text()
        )
        self.source = {
            "adapter_version": "p0_source_adapter_v1",
            "dataset": "fixture/dataset",
            "source_id": "fixture",
            "revision": "a" * 40,
            "subset": None,
            "domain": "general_zh",
            "language": "zh",
            "license_status": "pending",
            "text_field": "text",
            "locator_fields": ["id", "url"],
            "family_fields": ["url"],
            "metadata_fields": ["url"],
        }

    def test_normalization_preserves_code_indentation_and_math(self):
        text = "Ａ = 1\r\n    print(Ａ)  \r\n\r\n\r\n\r\n$x^2$\x00"
        normalized = PIPELINE.normalize_text(text, max_blank_line_run=2)
        self.assertIn("    print(Ａ)", normalized)
        self.assertIn("$x^2$", normalized)
        self.assertNotIn("\r", normalized)
        self.assertNotIn("\x00", normalized)
        self.assertNotIn("\n\n\n\n", normalized)

    def test_ids_and_split_are_stable(self):
        row = {
            "id": "1",
            "url": "https://Example.com/a?q=tracking#section",
            "text": "这是一个完整的中文测试段落。" * 30,
        }
        first = PIPELINE.make_record(row, 0, self.source, self.pipeline_config)
        second = PIPELINE.make_record(row, 0, self.source, self.pipeline_config)
        self.assertEqual(first["document_id"], second["document_id"])
        self.assertEqual(first["family_id"], second["family_id"])
        self.assertEqual(first["split"], second["split"])

    def test_url_family_joins_across_sources(self):
        row = {
            "id": "1",
            "url": "https://Example.com/a?q=tracking#section",
            "text": "这是一个完整的中文测试段落。" * 30,
        }
        first = PIPELINE.make_record(row, 0, self.source, self.pipeline_config)
        other = dict(self.source)
        other["source_id"] = "another_source"
        second = PIPELINE.make_record(row, 0, other, self.pipeline_config)
        self.assertEqual(first["family_id"], second["family_id"])
        self.assertEqual(first["split"], second["split"])

    def test_row_fallback_locator_includes_source_file(self):
        source = dict(self.source)
        source["locator_fields"] = ["missing_id"]
        source["family_fields"] = []
        row = {"text": "完整测试内容。" * 50}
        record = PIPELINE.make_record(
            row, 7, source, self.pipeline_config, source_file="shard-00001.parquet"
        )
        self.assertEqual(record["source_locator"], "file=shard-00001.parquet|row=7")

    def test_source_schema_validation_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "text_field"):
            PIPELINE.validate_source_fields(self.source, ["id", "url"])
        with self.assertRaisesRegex(ValueError, "locator_fields"):
            PIPELINE.validate_source_fields(self.source, ["text"])
        report = PIPELINE.validate_source_fields(self.source, ["text", "id"])
        self.assertEqual(report["locator_fields_present"], ["id"])
        self.assertEqual(report["metadata_fields_missing"], ["url"])

    def test_declarative_row_filters(self):
        source = dict(self.source)
        source["row_filters"] = [
            {"field": "language", "allowed_values": ["Python"]},
            {"field": "license", "allowed_values": ["mit", "apache-2.0"]},
        ]
        self.assertTrue(
            PIPELINE.row_matches_source(
                {"language": "Python", "license": "mit"}, source
            )
        )
        self.assertFalse(
            PIPELINE.row_matches_source(
                {"language": "Shell", "license": "mit"}, source
            )
        )
        with self.assertRaisesRegex(ValueError, "row filter fields"):
            PIPELINE.validate_source_fields(source, ["text", "id", "language"])

    def test_pipeline_split_is_used(self):
        pipeline = dict(self.pipeline_config)
        pipeline["split"] = {"train": 0.0, "validation": 0.0, "test": 1.0}
        row = {"id": "x", "text": "完整测试内容。" * 50}
        record = PIPELINE.make_record(row, 0, self.source, pipeline)
        self.assertEqual(record["split"], "test")

    def test_short_and_repeated_documents_are_rejected(self):
        keep, reasons = PIPELINE.rule_filter("太短", "general_zh")
        self.assertFalse(keep)
        self.assertIn("too_short", reasons)
        repeated = "\n".join(["same repeated line"] * 20)
        keep, reasons = PIPELINE.rule_filter(repeated, "general_en", min_chars=10)
        self.assertFalse(keep)
        self.assertIn("repeated_lines", reasons)

    def test_teacher_request_is_content_addressed_and_bounded(self):
        row = {
            "id": "1",
            "url": "https://example.com/long",
            "text": "开头" + ("中间内容" * 4000) + "结尾",
        }
        record = PIPELINE.make_record(row, 0, self.source, self.pipeline_config)
        request = PIPELINE.make_teacher_request(record, max_chars=1000)
        self.assertLessEqual(len(request["text"]), 1020)
        self.assertIn("中间省略", request["text"])
        self.assertEqual(request["request_id"], PIPELINE.make_teacher_request(record, 1000)["request_id"])

    def test_atomic_write_replaces_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.jsonl"
            PIPELINE.atomic_write_text(path, "one\n")
            PIPELINE.atomic_write_text(path, "two\n")
            self.assertEqual(path.read_text(), "two\n")


if __name__ == "__main__":
    unittest.main()
