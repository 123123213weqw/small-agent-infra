import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p0_sample_finewiki_quality import rank, stratum  # noqa: E402


class FinewikiSampleTests(unittest.TestCase):
    def test_strata_boundaries(self):
        self.assertEqual(stratum(500, False), "500-1999/math=0")
        self.assertEqual(stratum(1999, True), "500-1999/math=1")
        self.assertEqual(stratum(2000, False), "2000-5999/math=0")
        self.assertEqual(stratum(6000, True), "6000-11999/math=1")
        self.assertEqual(stratum(12000, False), "12000+/math=0")
        with self.assertRaises(ValueError):
            stratum(499, False)

    def test_rank_is_deterministic(self):
        self.assertEqual(rank("x"), rank("x"))
        self.assertNotEqual(rank("x"), rank("y"))


if __name__ == "__main__":
    unittest.main()
