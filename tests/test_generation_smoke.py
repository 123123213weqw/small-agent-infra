from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

MODULE = Path(__file__).parents[1] / "src/small_agent/evaluation/generation_smoke.py"
SPEC = importlib.util.spec_from_file_location("generation_smoke", MODULE)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
compare_cache_tokens = module.compare_cache_tokens
generation_metrics = module.generation_metrics
longest_identical_token_run = module.longest_identical_token_run
repeated_ngram_rate = module.repeated_ngram_rate
summarize_step = module.summarize_step


class GenerationSmokeMetricsTest(unittest.TestCase):
    def test_repeated_ngram_rate(self) -> None:
        self.assertEqual(repeated_ngram_rate([1, 2, 3], 4), 0.0)
        self.assertEqual(repeated_ngram_rate([1, 2, 3, 4, 1, 2, 3, 4], 4), 0.2)
        self.assertEqual(longest_identical_token_run([1, 2, 2, 2, 3]), 3)
        with self.assertRaises(ValueError):
            repeated_ngram_rate([1], 0)

    def test_generation_metrics_and_invalid_characters(self) -> None:
        result = generation_metrics([5, 5, 5, 5, 2], "a\ufffd\x01", eos_id=2)
        self.assertEqual(result["generated_tokens"], 5)
        self.assertTrue(result["eos_reached"])
        self.assertEqual(result["eos_position"], 5)
        self.assertEqual(result["replacement_character_count"], 1)
        self.assertEqual(result["control_character_count"], 1)
        self.assertEqual(result["longest_identical_token_run"], 4)
        self.assertTrue(result["utf8_encodable"])
        self.assertFalse(generation_metrics([1], "\ud800", eos_id=2)["utf8_encodable"])

    def test_cache_comparison(self) -> None:
        exact = compare_cache_tokens([1, 2, 3], [1, 2, 3])
        self.assertTrue(exact["exact_match"])
        self.assertIsNone(exact["first_divergence_index"])
        different = compare_cache_tokens([1, 2, 3], [1, 4, 3, 5])
        self.assertFalse(different["exact_match"])
        self.assertEqual(different["first_divergence_index"], 1)
        self.assertEqual(different["position_agreement_rate"], 0.5)

    def test_summary(self) -> None:
        def sample(ids: list[int]) -> dict:
            record = {"metrics": generation_metrics(ids, "text", eos_id=2)}
            return {"cached": record, "uncached": record,
                    "cache_comparison": compare_cache_tokens(ids, ids)}

        summary = summarize_step([sample([1, 2]), sample([3, 4])])
        self.assertEqual(summary["prompts"], 2)
        self.assertEqual(summary["cached"]["eos_reached_prompts"], 1)
        self.assertEqual(summary["cache_comparison"]["exact_match_prompts"], 2)
        with self.assertRaises(ValueError):
            summarize_step([])


if __name__ == "__main__":
    unittest.main()
