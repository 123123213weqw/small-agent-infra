import random
import re
import unittest

from experiments.phase1_reader_v2 import make_batch, make_episode


class CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


class ReaderV2DataTests(unittest.TestCase):
    def test_target_is_in_one_old_record_not_query_binding(self):
        variants = (
            "standard", "heldout_ids", "paraphrase", "delay", "mixed",
            "unseen_natural", "delay64", "id_extreme",
            "record_paraphrase_only", "query_paraphrase_only",
        )
        for variant in variants:
            for seed in range(20):
                records, query, answer, target = make_episode(random.Random(seed), 8, variant)
                self.assertEqual(len(records), 8)
                self.assertEqual(len(set(re.findall(r"K\d+", " ".join(records)))), 8)
                self.assertEqual(len(answer), 1)
                self.assertIn("K", query)
                self.assertTrue(0 <= target < 8)
                self.assertRegex(records[target], rf"(?:value|digit|number) {answer}\.")
                self.assertNotIn(records[target], query)

    def test_batch_input_ends_before_answer(self):
        tokenizer = CharacterTokenizer()
        indices = [0, 1, 2]
        batch, episodes = make_batch(
            tokenizer, seed=123, indices=indices, slots=8,
            max_record_tokens=256, device="cpu", variant="delay",
        )
        input_ids, attention_mask, labels, _, _, target_slot, prefix_len = batch
        for b, (_, query, answer, target) in enumerate(episodes):
            expected = tokenizer.encode(query)
            self.assertEqual(input_ids[b, : prefix_len[b]].tolist(), expected)
            self.assertEqual(int(attention_mask[b].sum()), len(expected))
            self.assertEqual(int(labels[b]), ord(answer))
            self.assertEqual(int(target_slot[b]), target)


if __name__ == "__main__":
    unittest.main()
