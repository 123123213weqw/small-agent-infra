import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p0_triage_finewiki_near_hits import greedy_direct_cover, normalize_title  # noqa: E402


class FineWikiTitleTriageTest(unittest.TestCase):
    def test_normalize_title(self):
        self.assertEqual(normalize_title("大雅區 （臺灣）"), "大雅區臺灣")
        self.assertEqual(normalize_title("Ａ．B"), "ab")

    def test_direct_cover_does_not_collapse_chain(self):
        # A--B--C but A and C are not directly near duplicates.
        graph = {"a": {"b": 0.9}, "b": {"a": 0.9, "c": 0.9}, "c": {"b": 0.9}}
        metadata = {
            "a": {"split": "test", "token_count": 100},
            "b": {"split": "train", "token_count": 200},
            "c": {"split": "train", "token_count": 300},
        }
        removed = greedy_direct_cover(graph, metadata)
        self.assertEqual(removed, {"b": ("a", 0.9)})

    def test_longer_kept_within_split(self):
        graph = {"a": {"b": 0.99}, "b": {"a": 0.99}}
        metadata = {"a": {"split": "train", "token_count": 100},
                    "b": {"split": "train", "token_count": 200}}
        self.assertEqual(greedy_direct_cover(graph, metadata), {"a": ("b", 0.99)})


if __name__ == "__main__":
    unittest.main()
