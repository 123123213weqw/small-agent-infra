"""Regression tests for two-judge agreement on both quality and target bucket."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p0_two_judge_gate.py"


def annotation(document_id: str, bucket: str) -> dict:
    return {
        "document_id": document_id,
        "decision": "keep",
        "confidence": 0.95,
        "quality": 4,
        "completeness": 4,
        "educational_value": 4,
        "format_integrity": 4,
        "recommended_bucket": bucket,
    }


class TwoJudgeGateTest(unittest.TestCase):
    def test_bucket_disagreement_drops_and_agreed_rebucket_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = [
                {"document_id": "a" * 64, "current_bucket": "math_en", "review_payload": "first"},
                {"document_id": "b" * 64, "current_bucket": "math_en", "review_payload": "second"},
            ]
            (root / "input.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
            for model, targets in (("deepseek", ("technical", "technical")),
                                   ("qwen", ("math_en", "technical"))):
                rows = []
                for doc, bucket in zip(docs, targets):
                    rows.append({
                        "source_id": json.dumps([doc["document_id"]]),
                        "status": "ok",
                        "prompt_sha256": "same-prompt",
                        "schema_sha256": "same-schema",
                        "text_sha256": hashlib.sha256(doc["review_payload"].encode()).hexdigest(),
                        "annotation": annotation(doc["document_id"], bucket),
                    })
                (root / f"{model}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            subprocess.run([
                sys.executable, str(SCRIPT), "--input", str(root / "input.jsonl"),
                "--deepseek", str(root / "deepseek.jsonl"),
                "--qwen", str(root / "qwen.jsonl"),
                "--output", str(root / "gate.json"),
            ], check=True, capture_output=True, text=True)
            report = json.loads((root / "gate.json").read_text())
            first, second = report["records"]
            self.assertEqual((first["route"], first["reason"], first["target_bucket"]),
                             ("exclude_from_training", "bucket_disagreement", None))
            self.assertEqual((second["route"], second["reason"], second["target_bucket"]),
                             ("candidate_keep", "both_strict_keep", "technical"))


if __name__ == "__main__":
    unittest.main()
