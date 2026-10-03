import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p0_train_tokenizers", ROOT / "scripts" / "p0_train_tokenizers.py")
TRAIN = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TRAIN)


class P0TrainTokenizerTests(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(TRAIN.percentile([], 0.5), 0)
        self.assertEqual(TRAIN.percentile([1, 2, 3, 4], 0.5), 2)
        self.assertEqual(TRAIN.percentile([1, 2, 3, 4], 0.95), 4)

    def test_first_mismatch(self):
        self.assertEqual(TRAIN.first_mismatch("abc", "abx"), 2)
        self.assertEqual(TRAIN.first_mismatch("abc", "abcx"), 3)


if __name__ == "__main__":
    unittest.main()
