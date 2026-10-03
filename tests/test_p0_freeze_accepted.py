import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p0_freeze_accepted", ROOT / "scripts" / "p0_freeze_accepted.py")
FREEZE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FREEZE)


class P0FreezeAcceptedTests(unittest.TestCase):
    def test_freeze_partitions_every_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool = root / "pool"
            shard_dir = pool / "general_en"
            shard_dir.mkdir(parents=True)
            rows = []
            for number in range(3):
                rows.append(
                    {
                        "document_id": str(number) * 64,
                        "source_id": "source",
                        "source_locator": f"row={number}",
                        "domain": "general_en",
                        "text": f"text {number}",
                        "token_count": 10 + number,
                        "license_status": "pending",
                    }
                )
            shard = shard_dir / "part.parquet"
            pq.write_table(pa.Table.from_pylist(rows), shard)
            manifest = {
                "stage": "consolidate_candidates",
                "tokenizer": {"model_id": "test"},
                "shards": [{"domain": "general_en", "file": shard.name, "rows": 3, "sha256": FREEZE.sha256_file(shard)}],
            }
            (pool / "manifest.json").write_text(json.dumps(manifest))
            adjudicated = root / "adjudicated.jsonl"
            decisions = [
                {
                    "document_id": "0" * 64, "route": "keep", "teacher_status": "ok",
                    "decision": "keep", "confidence": 0.9, "quality": 4, "completeness": 5,
                    "educational_value": 4, "format_integrity": 5, "recommended_bucket": "general_en",
                    "reason_codes": [], "evidence": "complete",
                },
                {
                    "document_id": "1" * 64, "route": "drop", "teacher_status": "ok",
                    "decision": "drop", "confidence": 0.9, "quality": 2, "completeness": 2,
                    "educational_value": 2, "format_integrity": 4, "recommended_bucket": "drop",
                    "reason_codes": ["low_information"], "evidence": "short",
                },
            ]
            adjudicated.write_text("".join(json.dumps(row) + "\n" for row in decisions))
            prefilter = root / "prefilter.jsonl"
            prefilter.write_text(json.dumps({"document_id": "2" * 64, "reason_codes": ["seo"]}) + "\n")
            output = root / "accepted"
            result = FREEZE.freeze(pool, adjudicated, prefilter, output)
            self.assertEqual(result["counts"]["accepted_documents"], 1)
            self.assertEqual(result["counts"]["accepted_tokens"], 10)
            self.assertEqual(result["counts"]["rejected_documents"], 2)
            self.assertFalse(result["license_gate"]["training_eligible"])
            accepted = pq.read_table(output / "general_en" / "part-00000.parquet").to_pylist()
            self.assertEqual(accepted[0]["teacher_quality"], 4)

    def test_empty_accepted_set_is_not_training_eligible(self):
        """空接受集不得报告 training_eligible=true。

        旧实现用 `set(licenses) <= {"approved"}`，空集的子集判断恒为真，
        于是「一篇都没接受」会写出 training_eligible=true 的 manifest。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool = root / "pool"
            shard_dir = pool / "general_en"
            shard_dir.mkdir(parents=True)
            rows = [
                {
                    "document_id": "a" * 64,
                    "source_id": "source",
                    "source_locator": "row=0",
                    "domain": "general_en",
                    "text": "text",
                    "token_count": 10,
                    "license_status": "pending",
                }
            ]
            shard = shard_dir / "part.parquet"
            pq.write_table(pa.Table.from_pylist(rows), shard)
            manifest = {
                "stage": "consolidate_candidates",
                "tokenizer": {"model_id": "test"},
                "shards": [
                    {
                        "domain": "general_en",
                        "file": shard.name,
                        "rows": 1,
                        "sha256": FREEZE.sha256_file(shard),
                    }
                ],
            }
            (pool / "manifest.json").write_text(json.dumps(manifest))
            adjudicated = root / "adjudicated.jsonl"
            adjudicated.write_text(
                json.dumps(
                    {
                        "document_id": "a" * 64, "route": "drop", "teacher_status": "ok",
                        "decision": "drop", "confidence": 0.9, "quality": 1,
                        "completeness": 1, "educational_value": 1, "format_integrity": 1,
                        "recommended_bucket": "drop", "reason_codes": ["low_information"],
                        "evidence": "junk",
                    }
                )
                + "\n"
            )
            prefilter = root / "prefilter.jsonl"
            prefilter.write_text("")
            result = FREEZE.freeze(pool, adjudicated, prefilter, root / "accepted")
            self.assertEqual(result["counts"]["accepted_documents"], 0)
            self.assertEqual(result["license_gate"]["statuses"], {})
            self.assertFalse(result["license_gate"]["training_eligible"])


if __name__ == "__main__":
    unittest.main()
