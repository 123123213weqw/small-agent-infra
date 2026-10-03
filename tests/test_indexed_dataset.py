from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_agent.data.indexed_dataset import IndexedTokenDataset  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_corpus(root: Path, *, corrupt_boundary: bool = False) -> Path:
    root.mkdir()
    streams = {
        "train": np.asarray([10, 11, 2, 12, 13, 14, 2, 15, 16, 17, 18, 2], dtype=np.uint16),
        "validation": np.asarray([20, 21, 2], dtype=np.uint16),
        "test": np.asarray([30, 31, 2], dtype=np.uint16),
    }
    spans = {
        "train": [
            {"document_id": "a", "start": 0, "end": 3},
            {"document_id": "b", "start": 3, "end": 7},
            {"document_id": "c", "start": 7, "end": 12},
        ],
        "validation": [{"document_id": "d", "start": 0, "end": 3}],
        "test": [{"document_id": "e", "start": 0, "end": 3}],
    }
    if corrupt_boundary:
        spans["train"][1]["start"] = 4
    split_info = {}
    for split, stream in streams.items():
        path = root / f"{split}.npy"
        np.save(path, stream, allow_pickle=False)
        split_info[split] = {
            "documents": len(spans[split]),
            "tokens_including_eos": len(stream),
            "eos_tokens": len(spans[split]),
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    manifest = {
        "stage": "internal_smoke_token_stream",
        "dtype": "uint16",
        "vocab_size": 64,
        "eos_token_id": 2,
        "splits": split_info,
        "document_spans": spans,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class IndexedTokenDatasetTests(unittest.TestCase):
    def test_overlapping_windows_preserve_next_token_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = IndexedTokenDataset(make_corpus(Path(temporary) / "data"), "train", 5)
            self.assertEqual(len(dataset), 2)
            first = dataset[0]
            second = dataset[1]
            self.assertEqual(first["input_ids"].tolist(), [10, 11, 2, 12, 13])
            self.assertEqual(second["input_ids"].tolist(), [13, 14, 2, 15, 16])
            self.assertEqual(first["labels"].tolist(), first["input_ids"].tolist())
            self.assertEqual(first["attention_mask"].tolist(), [1, 1, 1, 1, 1])
            self.assertEqual(first["input_ids"][-1], second["input_ids"][0])
            self.assertEqual(first["token_offset"].item(), 0)
            self.assertEqual(second["token_offset"].item(), 4)
            report = dataset.coverage_report()
            self.assertEqual(report["predicted_tokens"], 8)
            self.assertEqual(report["dropped_tail_tokens"], 3)

    def test_negative_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = IndexedTokenDataset(make_corpus(Path(temporary) / "data"), "train", 5)
            self.assertEqual(dataset[-1]["sample_id"].item(), 1)

    def test_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_corpus(Path(temporary) / "data")
            with (root / "train.npy").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                IndexedTokenDataset(root, "train", 5)

    def test_boundary_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_corpus(Path(temporary) / "data", corrupt_boundary=True)
            with self.assertRaisesRegex(ValueError, "non-contiguous"):
                IndexedTokenDataset(root, "train", 5)

    def test_short_stream_has_no_full_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = IndexedTokenDataset(make_corpus(Path(temporary) / "data"), "validation", 8)
            self.assertEqual(len(dataset), 0)

    def test_pad_tail_covers_all_validation_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = IndexedTokenDataset(
                make_corpus(Path(temporary) / "data"), "train", 5, tail_policy="pad"
            )
            self.assertEqual(len(dataset), 3)
            tail = dataset[2]
            self.assertEqual(tail["input_ids"].tolist(), [16, 17, 18, 2, 0])
            self.assertEqual(tail["labels"].tolist(), [16, 17, 18, 2, -100])
            self.assertEqual(tail["attention_mask"].tolist(), [1, 1, 1, 1, 0])
            self.assertEqual(tail["valid_tokens"].item(), 4)
            report = dataset.coverage_report()
            self.assertEqual(report["predicted_tokens"], 11)
            self.assertEqual(report["dropped_tail_tokens"], 0)
            self.assertEqual(report["prediction_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
