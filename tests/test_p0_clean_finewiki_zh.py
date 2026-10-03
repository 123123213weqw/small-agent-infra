import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p0_clean_finewiki_zh import clean_record  # noqa: E402


REVISION = "8bd13e72e6a002407649b3e898535f42ceb1aeb9"


class FinewikiCleanTests(unittest.TestCase):
    def row(self):
        return {
            "text": "这是一篇百科文章。" + "自然语言内容和技术说明。" * 65,
            "id": "zhwiki/123/456",
            "wikiname": "zhwiki",
            "page_id": 123,
            "title": "测试条目",
            "url": "https://zh.wikipedia.org/wiki/测试条目",
            "date_modified": "2025-08-01T12:34:56Z",
            "in_language": "zh",
            "version": "456",
            "has_math": False,
        }

    def test_valid_record_retains_provenance_but_not_training_approval(self):
        record, reasons = clean_record(self.row(), REVISION, "000_00000.parquet")
        self.assertEqual(reasons, [])
        self.assertEqual(record["license_status"], "pending")
        self.assertEqual(record["source_revision"], REVISION)
        self.assertEqual(record["raw_file"], "000_00000.parquet")
        self.assertIn(record["split"], {"train", "validation", "test"})
        self.assertEqual(record["date_modified"], "2025-08-01T12:34:56Z")

    def test_short_and_wrong_language_rejected(self):
        row = self.row()
        row["text"] = "短"
        row["in_language"] = "en"
        record, reasons = clean_record(row, REVISION, "000_00000.parquet")
        self.assertIsNone(record)
        self.assertEqual(reasons, ["too_short", "wrong_wiki_or_language"])

    def test_missing_url_rejected(self):
        row = self.row()
        row["url"] = None
        record, reasons = clean_record(row, REVISION, "000_00000.parquet")
        self.assertIsNone(record)
        self.assertIn("bad_url", reasons)


if __name__ == "__main__":
    unittest.main()
