import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("transfer_stats", ROOT / "scripts/p0_wave2_transfer_stats_v1.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def fixture(domain="first"):
    return {"stride": 2047, "shards": [
        {"split": s, "domain": domain, "documents": 3, "tokens": 4100, "samples": 2}
        for s in M.SPLITS]}


class TransferStatsTests(unittest.TestCase):
    def test_count_excludes_eos_from_body(self):
        s = M.summarize(fixture())
        self.assertEqual(s["total"], {"documents": 9, "stream_tokens": 12300, "text_tokens": 12291, "samples": 6})
        self.assertEqual(s["splits"]["train"]["stream_tokens"], 4100)
        self.assertEqual(s["one_pass_train_loss_tokens"], 4094)

    def test_inventory_not_validation_as_training(self):
        s = M.combine_stats({"old": M.summarize(fixture()), "new": M.summarize(fixture("second"))})
        self.assertEqual(s["splits"]["train"]["stream_tokens"], 8200)
        self.assertEqual(s["total"]["stream_tokens"], 24600)
        self.assertEqual(s["natural_train_body_token_fraction"], {"first": .5, "second": .5})

    def test_overlap_cannot_double_count(self):
        with self.assertRaises(ValueError):
            M.combine_stats({"a": M.summarize(fixture()), "b": M.summarize(fixture())})

    def test_invalid_accounting_rejected(self):
        v = fixture()
        v["shards"][0]["tokens"] = 2
        with self.assertRaises(ValueError):
            M.summarize(v)

    def test_unknown_split_rejected(self):
        v = fixture()
        v["shards"][0]["split"] = "unknown"
        with self.assertRaises(ValueError):
            M.summarize(v)

    def test_ready_binding_and_mismatch(self):
        stats = M.summarize(fixture())
        ready = {"documents": 9, "text_tokens": 12291, "stream_tokens_with_eos": 12300,
                 "splits": copy.deepcopy(stats["splits"])}
        M.match_ready(stats, ready)
        ready["splits"]["train"]["stream_tokens"] += 1
        with self.assertRaises(ValueError):
            M.match_ready(stats, ready)


if __name__ == "__main__":
    unittest.main()
