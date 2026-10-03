import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p0_index_finewiki_unflagged import load_wave1_overlap  # noqa: E402
import p0_data_pipeline as core  # noqa: E402


class UnflaggedIndexTests(unittest.TestCase):
    def test_loads_verified_family_hits(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            hits = root / "hits.jsonl"
            hits.write_text(json.dumps({
                "candidate_document_id": "abc", "exact_content": False,
                "same_url_family": True,
            }) + "\n")
            (root / "manifest.json").write_text(json.dumps({
                "stage": "finewiki_zh_vs_wave1_exact_and_family_overlap_audit_only",
                "hits_sha256": core.sha256_file(hits), "counts": {"either_hit": 1},
            }))
            self.assertEqual(load_wave1_overlap(root), {"abc"})
            hits.write_text("tampered\n")
            with self.assertRaises(ValueError):
                load_wave1_overlap(root)


if __name__ == "__main__":
    unittest.main()
