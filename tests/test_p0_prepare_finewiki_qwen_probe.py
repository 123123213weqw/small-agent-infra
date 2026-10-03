import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p0_prepare_finewiki_qwen_probe import prompt_token_count  # noqa: E402


class BatchEncodingLike:
    def __len__(self):
        return 2

    def __getitem__(self, key):
        return {"input_ids": [10, 11, 12, 13, 14], "attention_mask": [1] * 5}[key]


class PromptTokenCountTests(unittest.TestCase):
    def test_mapping_like_encoding_counts_ids_not_keys(self):
        self.assertEqual(prompt_token_count(BatchEncodingLike()), 5)

    def test_plain_list_encoding(self):
        self.assertEqual(prompt_token_count([10, 11, 12]), 3)


if __name__ == "__main__":
    unittest.main()
