import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p0_candidate_pipeline", ROOT / "scripts" / "p0_candidate_pipeline.py"
)
PIPELINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PIPELINE)


class P0CandidatePipelineTests(unittest.TestCase):
    def test_sample_rank_is_stable_and_seeded(self):
        first = PIPELINE.stable_sample_rank(7, "document")
        self.assertEqual(first, PIPELINE.stable_sample_rank(7, "document"))
        self.assertNotEqual(first, PIPELINE.stable_sample_rank(8, "document"))
        self.assertEqual(len(first), 64)

    def test_shingles_work_for_chinese_code_and_english(self):
        for text in (
            "这是一个用于测试近重复检测的中文段落。",
            "def add(a, b):\n    return a + b",
            "This is an educational technical document about parsers.",
        ):
            self.assertTrue(PIPELINE.text_shingles(text, size=3))

    def test_minhash_deduper_rejects_near_copy(self):
        deduper = PIPELINE.MinHashDeduper(
            seed=3, shingle_size=3, num_perm=32, bands=8, threshold=0.75
        )
        base = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        near = base + " lambda"
        different = "red green blue yellow orange purple black white silver"
        self.assertIsNone(deduper.find_duplicate("a", base))
        self.assertEqual(deduper.find_duplicate("b", near), "a")
        self.assertIsNone(deduper.find_duplicate("c", different))

    def test_jaccard(self):
        self.assertEqual(PIPELINE.jaccard({1, 2}, {1, 2}), 1.0)
        self.assertEqual(PIPELINE.jaccard({1}, {2}), 0.0)

    def test_source_prescreen_fails_closed_and_is_opt_in(self):
        rule = {"min_score": 3.5}
        PIPELINE.validate_source_prescreen({"finemath": rule})
        self.assertTrue(PIPELINE.passes_source_prescreen('{"score":3.5}', rule))
        self.assertFalse(PIPELINE.passes_source_prescreen('{"score":3.49}', rule))
        for metadata in (None, "{}", "not-json", '{"score":null}', '{"score":"4"}', '{"score":NaN}'):
            self.assertFalse(PIPELINE.passes_source_prescreen(metadata, rule))
        with self.assertRaises(ValueError):
            PIPELINE.validate_source_prescreen({"finemath": {"min_score": float("nan")}})

    def test_candidate_manifest_compatibility(self):
        base = {
            "stage": "build_candidates",
            "run_id": "run",
            "source_registry_sha256": "registry",
            "pipeline_config_sha256": "pipeline",
            "tokenizer": {"tokenizer_json_sha256": "tokenizer"},
            "license_gate": {"allow_pending_for_smoke": True},
        }
        shared = PIPELINE.candidate_manifest_compatibility([base, dict(base)])
        self.assertEqual(shared["run_id"], "run")
        extended_registry = dict(base)
        extended_registry["source_registry_sha256"] = "registry-v2"
        shared = PIPELINE.candidate_manifest_compatibility([base, extended_registry])
        self.assertEqual(
            shared["source_registry_sha256_values"], ["registry", "registry-v2"]
        )
        incompatible = dict(base)
        incompatible["pipeline_config_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "pipeline_config_sha256"):
            PIPELINE.candidate_manifest_compatibility([base, incompatible])

    def test_exact_dedup_keeps_lowest_rank(self):
        common = {
            "normalized_sha256": "same",
            "family_id": "family",
            "split": "train",
            "domain": "general_en",
        }
        later = {**common, "sample_rank": "f", "document_id": "later", "source_id": "b"}
        first = {**common, "sample_rank": "0", "document_id": "first", "source_id": "a"}
        retained, removed = PIPELINE.exact_deduplicate_records([later, first])
        self.assertEqual([row["document_id"] for row in retained], ["first"])
        self.assertEqual(removed[0]["record"]["document_id"], "later")
        self.assertEqual(removed[0]["representative"]["document_id"], "first")

    def test_family_split_conflict_fails_closed(self):
        base = {
            "normalized_sha256": "a",
            "document_id": "one",
            "family_id": "shared-family",
            "split": "train",
        }
        conflict = {
            **base,
            "normalized_sha256": "b",
            "document_id": "two",
            "split": "validation",
        }
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            PIPELINE.validate_document_and_family_consistency([base, conflict])


if __name__ == "__main__":
    unittest.main()
