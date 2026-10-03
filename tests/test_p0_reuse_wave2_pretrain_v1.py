import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reuse_wave2", ROOT / "scripts/p0_reuse_wave2_pretrain_v1.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
POLICY = json.loads((ROOT / "configs/curation/p0_wave2_pretrain_reuse_v1.json").read_text())


class ReuseTests(unittest.TestCase):
    def test_natural_fact_noise_is_not_a_hard_veto(self):
        text = "A tutorial with a mistaken DNS number. " * 20
        self.assertIsNone(M.hard_reason(text, "https://example.com/article", POLICY))

    def test_isolated_promotional_sentence_not_a_veto(self):
        text = "This explains a mathematical concept. " * 20 + " Subscribe to our newsletter."
        self.assertIsNone(M.hard_reason(text, "https://example.com/article", POLICY))

    def test_missing_body(self):
        self.assertEqual(M.hard_reason("", "https://example.com/a", POLICY), "missing_or_tiny_body")

    def test_missing_url(self):
        self.assertEqual(M.hard_reason("text " * 100, "", POLICY), "missing_or_invalid_source_url")

    def test_bad_port(self):
        self.assertEqual(M.hard_reason("text " * 100, "https://example.com:bad/a", POLICY), "missing_or_invalid_source_url")

    def test_nul_is_not_silently_removed(self):
        text = "text " * 100 + "\0"
        self.assertEqual(M.hard_reason(text, "https://example.com/a", POLICY), "nul_encoding_artifact")
        self.assertTrue(text.endswith("\0"))

    def test_replacement_damage(self):
        self.assertEqual(M.hard_reason("text " * 100 + "\ufffd" * 50, "https://example.com/a", POLICY), "severe_encoding_damage")

    def test_one_replacement_character_is_tolerated(self):
        self.assertIsNone(M.hard_reason("text " * 100 + "\ufffd", "https://example.com/a", POLICY))

    def test_reserved_literals(self):
        for token in M.RESERVED:
            self.assertEqual(M.hard_reason("text " * 100 + token, "https://example.com/a", POLICY), "reserved_training_token_literal")

    def test_page_alias_joins_tracking_and_http_versions(self):
        self.assertEqual(M.page_alias("http://www.EXAMPLE.com/a/?utm=1#x"), M.page_alias("https://example.com/a"))

    def test_distinct_pages_and_nondefault_ports_remain_distinct(self):
        a = M.page_alias("https://example.com/a")
        self.assertNotEqual(a, M.page_alias("https://example.com/b"))
        self.assertNotEqual(a, M.page_alias("https://example.com:8443/a"))

    def test_same_alias_same_split(self):
        alias = M.page_alias("https://example.com/a")
        self.assertEqual(M.assign_split(alias, 20261001), M.assign_split(alias, 20261001))

    def test_safe_filename(self):
        for value in ("../x", "/x", ".", "..", ""):
            with self.assertRaises(ValueError):
                M.safe_filename(value)

    def test_hash_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "data"
            p.write_text("original")
            digest = M.sha(p)
            M.verify_file(p, digest)
            p.write_text("modified")
            with self.assertRaises(ValueError):
                M.verify_file(p, digest)

    def test_json_output_does_not_overwrite_history(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "manifest.json"
            M.write_json(p, {"first": True})
            with self.assertRaises(FileExistsError):
                M.write_json(p, {"first": False})
            self.assertTrue(json.loads(p.read_text())["first"])

    def test_policy_preserves_old_rejects_and_no_dual_requirement(self):
        self.assertFalse(POLICY["require_unanimous_dual_judgment"])
        self.assertFalse(POLICY["rescue_previously_rejected_records"])
        self.assertFalse(POLICY["new_model_calls"])
        self.assertFalse(POLICY["modify_document_text"])


@unittest.skipUnless(os.environ.get("P0_REUSE_TEST_TOKENIZER"), "integration uses the frozen tokenizer on V100")
class IntegrationTests(unittest.TestCase):
    def test_freeze_pack_verify_and_mutation_stop(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tokenizers import Tokenizer
        tok_root = Path(os.environ["P0_REUSE_TEST_TOKENIZER"])
        tok = Tokenizer.from_file(str(tok_root / "tokenizer.json"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wave, prior, source, pack = (root / x for x in ("wave", "prior", "source", "pack"))
            candidate_root = wave / "audit/acceptance_v1/candidates"
            candidate_root.mkdir(parents=True)
            old_dir = prior / "general_en"
            old_dir.mkdir(parents=True)
            text = "An educational passage about engineering and practical computation. " * 160
            prior_text = "A different old passage about statistics and examples. " * 160
            old_rows = [{"text": prior_text, "family_id": "old_family", "split": "validation",
                         "source_locator": "url=https://prior.example/old"}]
            old_path = old_dir / "part.parquet"
            pq.write_table(pa.Table.from_pylist(old_rows), old_path)
            M.write_json(prior / "manifest.json", {"stage": "accepted_real", "license_gate": {"training_eligible": True},
                "counts": {"accepted_documents": 1}, "shards": [{"domain": "general_en", "file": old_path.name,
                    "bytes": old_path.stat().st_size, "sha256": M.sha(old_path)}]})
            urls = {}
            for i in range(10000):
                url = f"https://new.example/article-{i}"
                split = M.assign_split(M.page_alias(url), POLICY["split_seed"])
                urls.setdefault(split, url)
                if len(urls) == 3:
                    break
            definitions = [(text + " train", urls["train"]), (text + " validation", urls["validation"]),
                           (text + " test", urls["test"]), (prior_text, "https://new.example/exact"),
                           (text + " prior family", "https://prior.example/old?tracking=1"),
                           (text + "\0", "https://new.example/nul"),
                           (text + "<|system|>", "https://new.example/reserved"),
                           (text + " same page another version", urls["train"] + "?tracking=1")]
            rows = []
            for i, (body, url) in enumerate(definitions):
                rows.append({"document_id": M.body_sha(body), "text": body, "url": url,
                    "source_id": "fineweb_edu_new", "dataset_revision": POLICY["source_revision"],
                    "source_file": "fixture.parquet", "raw_row": i, "upstream_dump": "FIXTURE_NOT_REAL_DATA",
                    "upstream_file_path": "fixture", "upstream_license_label": "odc-by", "bucket": "general_en",
                    "review_route": "qwen_candidate", "exact_text_tokens": len(tok.encode(body, add_special_tokens=False).ids),
                    "upstream_score": 3.5, "training_approved": False, "split": "unassigned"})
            candidate = candidate_root / "fixture.parquet"
            pq.write_table(pa.Table.from_pylist(rows), candidate)
            exclusion_schema = pa.schema([("source_file", pa.string()), ("raw_row", pa.int64())])
            near_root, ev_root = wave / "audit/near_dedup_v1/verified_v1", wave / "audit/eval_screen_v1"
            near_root.mkdir(parents=True)
            ev_root.mkdir(parents=True)
            near_index, ev_index = near_root / "wave2_exclusion_index.parquet", ev_root / "quarantine_index.parquet"
            for p in (near_index, ev_index):
                pq.write_table(pa.Table.from_pylist([], schema=exclusion_schema), p)
            near = {"outputs_sha256": {near_index.name: M.sha(near_index)}}
            ev = {"documents_scanned": len(rows), "hit_documents_quarantined": 0,
                  "quarantine_index_sha256": M.sha(ev_index), "reference_files": {}}
            M.write_json(near_root / "SUMMARY.json", near)
            M.write_json(ev_root / "SUMMARY.json", ev)
            recon = wave / "audit/acceptance_v1/reports/RECONCILIATION.json"
            M.write_json(recon, {"selected_documents": len(rows), "inherited_near_dedup": near,
                                "inherited_eval_screen": ev})
            M.write_json(wave / "manifest.json", {"dataset": POLICY["source_dataset"], "revision": POLICY["source_revision"],
                "upstream_license_label": "odc-by", "items": [{"path": "fixture.parquet", "sha256": "f" * 64}]})
            total = sum(r["exact_text_tokens"] for r in rows)
            M.write_json(wave / "audit/acceptance_v1/SUMMARY.json", {"documents": len(rows),
                "exact_text_tokens_excluding_eos": total, "tokenizer_sha256": POLICY["tokenizer_sha256"],
                "reconciliation_sha256": M.sha(recon), "shards": [{"file": candidate.name, "raw_sha256": "f" * 64,
                    "documents": len(rows), "text_tokens_excluding_eos": total,
                    "output_sha256": M.sha(candidate), "output_bytes": candidate.stat().st_size}]})
            policy_path = root / "policy.json"
            policy = dict(POLICY, expected_candidate_documents=len(rows), expected_candidate_text_tokens=total)
            M.write_json(policy_path, policy)
            args = SimpleNamespace(wave_root=wave, prior=[prior], tokenizer=tok_root, policy=policy_path, output=source)
            result = M.freeze(args)
            self.assertEqual(result["counts"]["accepted_documents"], 4)
            self.assertEqual(sum(result["excluded"].values()), 4)
            frozen_rows = pq.read_table(source / POLICY["domain"] / "fixture.parquet").to_pylist()
            self.assertEqual([r["text"] for r in frozen_rows], [definitions[i][0] for i in (0, 1, 2, 7)])
            self.assertEqual(frozen_rows[0]["split"], frozen_rows[-1]["split"])
            M.pack(SimpleNamespace(source=source, tokenizer=tok_root, output=pack))
            verified = M.verify(SimpleNamespace(source=source, pack=pack, output=root / "READY.json"))
            self.assertEqual(verified["documents"], 4)
            self.assertFalse(verified["training_started"])
            self.assertEqual(verified["new_api_calls"], 0)
            with self.assertRaises(FileExistsError):
                M.freeze(args)
            manifest_path = pack / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["source_manifest_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                M.verify(SimpleNamespace(source=source, pack=pack, output=root / "BAD_READY.json"))


if __name__ == "__main__":
    unittest.main()
