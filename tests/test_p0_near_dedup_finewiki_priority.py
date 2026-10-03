import importlib.util
import io
import sys
import tempfile
from pathlib import Path
import unittest
from collections import Counter, defaultdict


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("pyarrow"),
                     "remote V100 Python dependencies required")
class FineWikiFullNearDedupTest(unittest.TestCase):
    def test_positive_control_and_cross_split_hit(self):
        from p0_near_dedup_ultradata_night8 import NearIndex, shingles
        from p0_near_dedup_finewiki_priority import PAIR, add_record, verify_pairs

        base = "。".join(f"第{i}章介绍概念{i}与实验{i}的数据" for i in range(100))
        edited = base.replace("第42章介绍概念42与实验42的数据",
                              "第42章介绍概念42与实验42的新数据")
        records, buckets, pair_bytes, counts = [], defaultdict(list), io.BytesIO(), Counter()
        index = NearIndex()
        add_record(index, buckets, records,
                   {"document_id": "a", "source": "wave1_zh", "split": "train",
                    "file": "x", "row": 0, "token_count": 0}, base, pair_bytes, counts)
        add_record(index, buckets, records,
                   {"document_id": "b", "source": "finewiki", "split": "test",
                    "file": "y", "row": 0, "token_count": 100}, edited, pair_bytes, counts)
        self.assertEqual(list(PAIR.iter_unpack(pair_bytes.getvalue())), [(0, 1)])
        with tempfile.TemporaryDirectory() as tmp:
            pairs = Path(tmp) / "pairs.bin"
            output = Path(tmp) / "hits.jsonl"
            pairs.write_bytes(pair_bytes.getvalue())
            result = verify_pairs(pairs, records, {0: shingles(base), 1: shingles(edited)}, output)
            self.assertEqual(result["near_hits:wave1_to_finewiki"], 1)
            self.assertEqual(result["cross_split_near_hits:wave1_to_finewiki"], 1)


if __name__ == "__main__":
    unittest.main()
