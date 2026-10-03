import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p0_probe_finewiki_near_overlap import choose_sample, length_bin, stable_rank, stratum  # noqa: E402


class FineWikiNearProbeSelectionTest(unittest.TestCase):
    def test_length_bin_edges(self):
        self.assertEqual(length_bin(1999), "0500_1999")
        self.assertEqual(length_bin(2000), "2000_5999")
        self.assertEqual(length_bin(5999), "2000_5999")
        self.assertEqual(length_bin(6000), "6000_11999")
        self.assertEqual(length_bin(11999), "6000_11999")
        self.assertEqual(length_bin(12000), "12000_plus")

    def test_fixed_hash_selection_is_order_independent(self):
        rows = [
            {"document_id": f"doc-{i}", "chars": 500 + (i % 4) * 4000,
             "has_math": bool(i % 2)}
            for i in range(80)
        ]
        chosen = choose_sample(rows, 3)
        self.assertEqual(chosen, choose_sample(list(reversed(rows)), 3))
        per_stratum = {}
        for row in chosen:
            per_stratum[stratum(row)] = per_stratum.get(stratum(row), 0) + 1
        self.assertTrue(all(count <= 3 for count in per_stratum.values()))
        for key in per_stratum:
            expected = sorted(
                (row for row in rows if stratum(row) == key),
                key=lambda row: (stable_rank(row["document_id"]), row["document_id"]),
            )[:3]
            self.assertEqual([r for r in chosen if stratum(r) == key], expected)


if __name__ == "__main__":
    unittest.main()
