import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p0_finalize_finewiki_near_policy import build_policy  # noqa: E402


class FineWikiNearPolicyTest(unittest.TestCase):
    def test_quarantine_ambiguous_endpoints_then_direct_cover(self):
        def pair(a, b, score=0.9):
            return {"group": "finewiki_internal", "left_document_id": a,
                    "right_document_id": b, "jaccard": score}

        hits = [pair("a", "b"), pair("b", "c"), pair("d", "e")]
        different = [{"left_document_id": "b", "right_document_id": "c"}]
        metadata = {x: {"split": "train", "token_count": 100} for x in "abcde"}
        metadata["e"]["token_count"] = 200
        excluded, counts = build_policy(hits, different, metadata)
        self.assertEqual(set(excluded), {"b", "c", "d"})
        self.assertEqual(excluded["d"]["keeper_document_id"], "e")
        self.assertEqual(counts["quarantine_documents"], 2)
        self.assertEqual(counts["excluded_tokens"], 300)


if __name__ == "__main__":
    unittest.main()
