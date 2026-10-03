import copy
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from small_agent.curation.core import (canonical, decide, decoder_schema, evidence_catalog, lint, normalize_document, route, sha,
                                       summary, validate_annotation)
from small_agent.curation.store import Batch, check_release, file_sha, prepare, protocol, read_json, write_json, engine_files
from small_agent.curation.ledger import Ledger, micros

import importlib.util
worker_spec = importlib.util.spec_from_file_location("curation_worker", ROOT / "scripts/p0_curation_worker_v3.py")
worker = importlib.util.module_from_spec(worker_spec)
worker_spec.loader.exec_module(worker)
cli_spec = importlib.util.spec_from_file_location("curation_cli", ROOT / "scripts/p0_curation_v3.py")
cli = importlib.util.module_from_spec(cli_spec)
cli_spec.loader.exec_module(cli)
smoke_spec = importlib.util.spec_from_file_location("curation_smoke", ROOT / "scripts/p0_curation_online_smoke_v3.py")
smoke = importlib.util.module_from_spec(smoke_spec)
smoke_spec.loader.exec_module(smoke)
env_spec = importlib.util.spec_from_file_location("curation_env", ROOT / "scripts/p0_curation_deepseek_env_v3.py")
credential_env = importlib.util.module_from_spec(env_spec)
env_spec.loader.exec_module(credential_env)
report_spec = importlib.util.spec_from_file_location("curation_smoke_report", ROOT / "scripts/p0_curation_smoke_report_v3.py")
smoke_report = importlib.util.module_from_spec(report_spec)
report_spec.loader.exec_module(smoke_report)
resume_spec = importlib.util.spec_from_file_location("curation_bounded_resume", ROOT / "scripts/p0_curation_bounded_resume_v3.py")
bounded_resume = importlib.util.module_from_spec(resume_spec)
resume_spec.loader.exec_module(bounded_resume)

POLICY = read_json(ROOT / "configs/curation/p0_policy_v3.json")
SCHEMA = read_json(ROOT / "schemas/p0_teacher_quality_v3.json")
PROMPT = ROOT / "configs/curation/p0_quality_prompt_v3.txt"


def raw(text="This is a complete introduction to graph search. A queue stores pending nodes."):
    return {"document_id": sha(text), "document_text": text, "source_id": "fixture",
            "source_locator": "https://example.org/fixture", "language": "en", "token_count": 16,
            "current_bucket": "general_en"}


def label(doc, bucket="general_en"):
    return {"document_id": doc["document_id"], "body_sha256": doc["body_sha256"],
            "coverage": "full_document", "decision": "keep", "confidence": .95,
            "quality": 4, "completeness": 4, "educational_value": 4, "format_integrity": 4,
            "recommended_bucket": bucket, "anchors": [doc["document_text"][:60]], "findings": []}


def registry():
    r = read_json(ROOT / "configs/curation/p0_judges_v3.example.json")
    r["deployment_status"] = "offline_test_only"
    for j in r["judges"].values():
        j["model_revision"] = "offline-fixture-identity-not-production"
    return r


def reviews(a, b=None):
    return {"qwen": {"status": "valid", "annotation": a}, "deepseek": {"status": "valid", "annotation": b or a}}


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.doc = normalize_document(raw())

    def test_exact_body_hash_and_no_silent_normalization(self):
        with self.assertRaises(ValueError):
            normalize_document({**raw(), "document_text": raw()["document_text"] + " "})

    def test_full_body_and_legacy_payload_supported(self):
        r = raw()
        doc = normalize_document({"token_count": 16, "sampling_stratum": "long", "review_payload": canonical(r)})
        self.assertEqual(doc["document_text"], r["document_text"])
        self.assertEqual(doc["sampling_stratum"], "long")

    def test_model_is_blind_to_first_label_and_bucket(self):
        payload = json.loads(self.doc["review_payload"])
        self.assertNotIn("current_bucket", payload)
        self.assertNotIn("previous_bucket", payload)
        self.assertNotIn("lint_flags", payload)

    def test_evidence_exact_span(self):
        resolved = validate_annotation(label(self.doc), self.doc, SCHEMA)
        span = resolved["anchors"][0]
        self.assertEqual(self.doc["document_text"][span["start"]:span["end"]], span["quote"])

    def test_fabricated_quote_rejected(self):
        a = label(self.doc)
        a["anchors"] = ["This fabricated passage does not exist."]
        with self.assertRaises(ValueError):
            validate_annotation(a, self.doc, SCHEMA)

    def test_ambiguous_quote_rejected(self):
        doc = normalize_document(raw("Repeat this sentence. Repeat this sentence."))
        a = label(doc)
        a["anchors"] = ["Repeat this sentence."]
        with self.assertRaises(ValueError):
            validate_annotation(a, doc, SCHEMA)

    def test_wrong_body_version_rejected(self):
        a = label(self.doc)
        a["body_sha256"] = "a" * 64
        with self.assertRaises(ValueError):
            validate_annotation(a, self.doc, SCHEMA)

    def test_nan_rejected(self):
        a = label(self.doc)
        a["confidence"] = float("nan")
        with self.assertRaises(ValueError):
            validate_annotation(a, self.doc, SCHEMA)

    def test_missing_field_and_partial_coverage_rejected(self):
        from jsonschema import ValidationError
        a = label(self.doc)
        a.pop("format_integrity")
        with self.assertRaises(ValidationError):
            validate_annotation(a, self.doc, SCHEMA)
        a = label(self.doc)
        a["coverage"] = "head_only"
        with self.assertRaises(ValidationError):
            validate_annotation(a, self.doc, SCHEMA)

    def test_keep_contradicting_finding_rejected(self):
        a = label(self.doc)
        a["findings"] = [{"code": "incomplete", "severity": "reject", "quotes": a["anchors"], "explanation": "The passage lacks necessary context."}]
        with self.assertRaises(ValueError):
            validate_annotation(a, self.doc, SCHEMA)

    def test_drop_requires_adverse_evidence(self):
        a = label(self.doc)
        a["decision"] = "drop"
        with self.assertRaises(ValueError):
            validate_annotation(a, self.doc, SCHEMA)

    def test_registration_and_advert_are_not_rejection_by_themselves(self):
        a = label(self.doc)
        a["decision"] = "drop"
        a["findings"] = [{"code": "registration_prompt", "severity": "reject", "quotes": a["anchors"], "explanation": "Registration was misinterpreted as a paywall."}]
        with self.assertRaises(ValueError):
            validate_annotation(a, self.doc, SCHEMA)
        flags = lint("Join to answer this question. Join a community of educators.")
        self.assertTrue(flags)
        self.assertTrue(all(f["severity"] == "info" for f in flags))

    def test_legacy_merged_paywall_reason_not_in_v3(self):
        from jsonschema import ValidationError
        a = label(self.doc)
        a["findings"] = [{"code": "paywall_or_preview", "severity": "reject", "quotes": a["anchors"], "explanation": "Not an allowed v3 combined reason."}]
        with self.assertRaises(ValidationError):
            validate_annotation(a, self.doc, SCHEMA)

    def test_dns_internal_inconsistency_flag_and_good_control(self):
        bad = "IP address 192.0.2.50 has corresponding reverse node 22.214.171.124.in-addr.arpa."
        good = bad.replace("22.214.171.124.in-addr.arpa", "168.22.104.18.in-addr.arpa")
        self.assertEqual(lint(bad)[0]["code"], "dns_reverse_mismatch")
        self.assertEqual(lint(good), [])

    def test_dns_unlinked_numbers_are_not_paired(self):
        self.assertEqual(lint("192.0.2.50 is a server. Another zone uses 22.214.171.124.in-addr.arpa."), [])

    def test_empty_list_and_real_list_control(self):
        self.assertTrue(lint("Symptoms may include:\nWhat Causes This Condition?"))
        self.assertEqual(lint("Symptoms may include:\n- fatigue\n- dizziness\nWhat Causes This Condition?"), [])

    def test_repeated_introducers_and_valid_options(self):
        self.assertTrue(lint("There are options:\nFor example, one could:\nNext Section"))
        self.assertEqual(lint("There are options:\nPlant additional trees.\nMonitor soil."), [])

    def test_preview_not_paywall(self):
        flags = lint("The report is interesting. See the rest of the story at Business Insider.")
        self.assertEqual(flags[0]["code"], "explicit_continuation_preview")
        self.assertNotIn("paywall", flags[0]["code"])

    def test_truncated_question(self):
        self.assertEqual(lint("Which of the following is correct?a) An unfinished choice...")[0]["code"], "truncated_question_marker")

    def test_parse_code_only_and_never_execute(self):
        self.assertEqual(lint("def broken(:", content_kind="text", language="en"), [])
        self.assertTrue(lint("def broken(:", content_kind="complete_code_file", language="python"))
        self.assertEqual(lint("raise RuntimeError('must not execute')", content_kind="complete_code_file", language="python"), [])

    def test_quality_and_bucket_separated(self):
        a, b = label(self.doc), label(self.doc, "technical")
        out = decide(self.doc, reviews(a, b), registry()["judges"], POLICY)
        self.assertEqual(out["status"], "candidate_eligible")
        self.assertEqual(out["bucket"], "general_en")
        self.assertFalse(out["training_approved"])

    def test_incompatible_route_is_not_bad_text(self):
        out = decide(self.doc, reviews(label(self.doc), label(self.doc, "code_python")), registry()["judges"], POLICY)
        self.assertEqual(out["status"], "quarantine")
        self.assertEqual(out["quality_status"], "both_judges_pass_not_ground_truth")
        self.assertEqual(out["reasons"], ["routing_uncertain"])

    def test_quality_disagreement_excluded_without_deleting_body(self):
        a = label(self.doc)
        b = {**a, "decision": "drop"}
        out = decide(self.doc, reviews(a, b), registry()["judges"], POLICY)
        self.assertEqual(out["reasons"], ["quality_disagreement"])

    def test_score_threshold_still_applies_without_bucket_condition(self):
        a = label(self.doc, "technical")
        a["completeness"] = 3
        out = decide(self.doc, reviews(a), registry()["judges"], POLICY)
        self.assertEqual(out["status"], "excluded")

    def test_lint_cannot_be_overruled_by_teacher_optimism(self):
        doc = normalize_document(raw("These options may include:\nWhat Is the Next Topic?"))
        a = label(doc)
        out = decide(doc, reviews(a), registry()["judges"], POLICY)
        self.assertEqual(out["status"], "quarantine")

    def test_missing_review_stays_in_denominator(self):
        out = decide(self.doc, {"qwen": {"status": "valid", "annotation": label(self.doc)}}, registry()["judges"], POLICY)
        report = summary([out])
        self.assertEqual(report["documents"], 1)
        self.assertEqual(report["status_documents"], {"await_review": 1})

    def test_independence_is_not_two_seeds(self):
        r = registry()["judges"]
        r["deepseek"]["family"] = "qwen"
        with self.assertRaises(ValueError):
            decide(self.doc, reviews(label(self.doc)), r, POLICY)

    def test_unknown_language_no_speculative_fallback(self):
        self.assertEqual(route(["technical", "general_en"], "unknown")[0], None)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.input = self.root / "input.jsonl"
        self.input.write_text(canonical(raw()) + "\n", encoding="utf-8")
        self.registry_path = self.root / "judges.json"
        write_json(self.registry_path, registry())
        self.path = self.root / "batch"
        self.make_batch()
        self.batch = Batch(self.path)
        self.doc = next(iter(self.batch.docs.values()))

    def tearDown(self):
        self.batch.close()
        self.tmp.cleanup()

    def make_batch(self, output=None, registry_path=None):
        return prepare(self.input, output or self.path, ROOT / "configs/curation/p0_policy_v3.json",
                       ROOT / "schemas/p0_teacher_quality_v3.json", PROMPT, registry_path or self.registry_path)

    def receipt(self, judge="qwen", a=None):
        j = self.batch.registry["judges"][judge]
        return {**protocol(self.doc, self.batch.prompt, self.batch.schema, j),
                "status": "ok", "served_model": j["model"], "annotation": a or label(self.doc)}

    def import_one(self, result=None, judge="qwen", name="results.jsonl"):
        path = self.root / name
        path.write_text(canonical(result or self.receipt(judge)) + "\n", encoding="utf-8")
        return self.batch.import_results(judge, path)

    def both(self):
        self.import_one()
        self.import_one(judge="deepseek", name="second.jsonl")

    def test_prepare_cannot_overwrite_and_temp_dirs_cleaned(self):
        with self.assertRaises(FileExistsError):
            self.make_batch()
        self.assertEqual(list(self.root.glob(".curation-v3-*")), [])

    def test_example_registry_cannot_deploy(self):
        with self.assertRaises(ValueError):
            self.make_batch(self.root / "invalid", ROOT / "configs/curation/p0_judges_v3.example.json")

    def test_online_report_keeps_unreviewed_documents_in_denominator(self):
        plan = self.root / "PLAN.json"
        write_json(plan, {"cases": [{"document_id": self.doc["document_id"], "smoke_group": "fixture"}]})
        result = smoke_report.report(plan, [self.path])
        self.assertEqual(result["documents"], 1)
        self.assertEqual(result["review_statuses"], {"queued": 2})
        self.assertEqual(result["statuses"], {"await_review": 1})
        self.assertFalse(result["training_approved"])
        self.assertFalse(result["independent_population_quality_accuracy_measured"])
        with self.assertRaises(ValueError):
            smoke_report.report(plan, [self.path, self.path])

    def test_bounded_supervisor_without_valid_pilot_calls_nothing(self):
        argv = ["supervisor", "--batch", str(self.path), "--judge", "qwen", "--ledger", str(self.root / "ledger.sqlite"),
                "--budget-cny", "0", "--output", str(self.root / "supervisor.json"), "--continue-known-validation-failures"]
        with patch.object(sys, "argv", argv), patch.object(bounded_resume.subprocess, "run") as child:
            with self.assertRaises(ValueError):
                bounded_resume.main()
            child.assert_not_called()

    def test_duplicate_body_rejected_before_publish(self):
        self.input.write_text((canonical(raw()) + "\n") * 2)
        with self.assertRaises(ValueError):
            self.make_batch(self.root / "duplicates")
        self.assertFalse((self.root / "duplicates").exists())

    def test_batch_memory_bound_enforced(self):
        p = self.root / "small_policy.json"
        policy = copy.deepcopy(POLICY)
        policy["maximum_batch_body_bytes"] = 8
        write_json(p, policy)
        with self.assertRaises(ValueError):
            prepare(self.input, self.root / "too_big", p, ROOT / "schemas/p0_teacher_quality_v3.json", PROMPT, self.registry_path)

    def test_duplicate_import_is_idempotent(self):
        self.assertEqual(self.import_one(), {"valid": 1})
        self.assertEqual(self.import_one(), {"already_imported": 1})
        self.assertEqual(self.batch.report()["job_statuses"], {"queued": 1, "valid": 1})

    def test_protocol_change_cannot_reuse_old_result(self):
        r = self.receipt()
        r["prompt_sha256"] = "a" * 64
        self.assertEqual(self.import_one(r), {"invalid": 1})
        self.assertEqual(len(self.batch.pending("qwen")), 1)

    def test_generation_change_cannot_reuse_result(self):
        r = self.receipt()
        r["inference_sha256"] = "b" * 64
        self.assertEqual(self.import_one(r), {"invalid": 1})

    def test_unexpected_served_model_rejected(self):
        r = self.receipt()
        r["served_model"] = "different-model"
        self.assertEqual(self.import_one(r), {"invalid": 1})

    def test_invalid_schema_can_retry_then_resume(self):
        r = self.receipt()
        r["annotation"].pop("format_integrity")
        self.assertEqual(self.import_one(r), {"invalid": 1})
        self.assertEqual(self.import_one(), {"valid": 1})
        self.batch.close()
        self.batch = Batch(self.path)
        self.assertEqual(self.batch.pending("qwen"), [])

    def test_conflicting_valid_labels_never_last_writer_wins(self):
        self.both()
        a = label(self.doc, "technical")
        self.assertEqual(self.import_one(self.receipt(a=a)), {"conflict": 1})
        self.assertEqual(self.batch.decisions()[0]["status"], "quarantine")
        self.import_one()
        self.assertEqual(self.batch.decisions()[0]["status"], "quarantine")

    def test_invalid_attempt_does_not_erase_valid_review(self):
        self.import_one()
        r = self.receipt()
        r["status"] = "error"
        self.import_one(r)
        self.assertEqual(self.batch.report()["job_statuses"], {"queued": 1, "valid": 1})

    def test_manifest_mutation_stops_resume(self):
        (self.path / "prompt.txt").write_text("tampered")
        with self.assertRaises(ValueError):
            Batch(self.path)

    def test_changed_engine_requires_new_batch_not_silent_gate_change(self):
        files = engine_files()
        files["src/small_agent/curation/core.py"] = "a" * 64
        with patch("small_agent.curation.store.engine_files", return_value=files), self.assertRaises(ValueError):
            Batch(self.path)

    def test_frozen_body_stored_once_payload_derived_identically(self):
        doc = json.loads((self.path / "documents.jsonl").read_text())
        self.assertNotIn("review_payload", doc)
        self.assertEqual(self.doc["review_payload"], normalize_document(raw())["review_payload"])

    def test_database_cannot_be_transplanted(self):
        with self.batch.db:
            self.batch.db.execute("UPDATE meta SET value='tampered'")
        with self.assertRaises(ValueError):
            Batch(self.path)

    def test_export_eligible_is_still_not_training_approved(self):
        self.both()
        report = self.batch.export(self.root / "snapshot")
        self.assertEqual(report["status_documents"], {"candidate_eligible": 1})
        d = json.loads((self.root / "snapshot/eligible.jsonl").read_text())
        self.assertFalse(d["training_approved"])

    def test_release_without_checks_is_blocked(self):
        self.both()
        snapshot = self.root / "snapshot"
        self.batch.export(snapshot)
        checks = self.root / "checks.json"
        write_json(checks, {"checks": []})
        result = check_release(self.batch, snapshot, checks)
        self.assertFalse(result["release_checks_bound"])
        self.assertFalse(result["training_ready"])
        self.assertTrue(any("independent_calibration" in e for e in result["errors"]))

    def test_new_conflict_invalidates_existing_snapshot(self):
        self.both()
        snapshot = self.root / "snapshot"
        self.batch.export(snapshot)
        self.import_one(self.receipt(a=label(self.doc, "technical")))
        checks = self.root / "checks.json"
        write_json(checks, {"checks": []})
        with self.assertRaises(ValueError):
            check_release(self.batch, snapshot, checks)

    def test_export_snapshot_cannot_be_modified(self):
        self.both()
        snapshot = self.root / "snapshot"
        self.batch.export(snapshot)
        (snapshot / "eligible.jsonl").write_text("tampered")
        checks = self.root / "checks.json"
        write_json(checks, {"checks": []})
        with self.assertRaises(ValueError):
            check_release(self.batch, snapshot, checks)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "budget.sqlite"
        self.kwargs = {"budget_cny": 2, "account_sha256": "a" * 64,
                       "pricing": {"uncached_input": 2, "output": 8, "margin": 1.15}}
        self.ledger = Ledger(self.path, **self.kwargs)

    def tearDown(self):
        self.ledger.close()
        self.tmp.cleanup()

    def test_reserve_before_call_and_budget_cap(self):
        self.assertTrue(self.ledger.reserve("job1", 1.2))
        with self.assertRaises(ValueError):
            self.ledger.reserve("job2", .9)
        self.assertEqual(self.ledger.report()["estimated_or_reserved_cny"], 1.2)

    def test_process_resume_does_not_resend_unknown_request(self):
        self.ledger.reserve("job1", .5)
        self.ledger.close()
        self.ledger = Ledger(self.path, **self.kwargs)
        self.assertFalse(self.ledger.reserve("job1", .5))
        self.assertIsNone(self.ledger.result("job1"))

    def test_result_is_durable_before_import(self):
        self.ledger.reserve("job1", .5)
        self.ledger.finish("job1", {"status": "ok", "annotation": {}}, observed_cost_cny=.1)
        self.assertEqual(self.ledger.result("job1")["status"], "ok")
        self.assertFalse(self.ledger.reserve("job1", .5))
        self.assertEqual(self.ledger.report()["estimated_or_reserved_cny"], .1)

    def test_missing_usage_keeps_full_reservation(self):
        self.ledger.reserve("job1", .5)
        self.ledger.finish("job1", {"status": "error"})
        self.assertEqual(self.ledger.report()["estimated_or_reserved_cny"], .5)

    def test_shared_ledger_does_not_multiply_budget_by_workers(self):
        other = Ledger(self.path, **self.kwargs)
        try:
            self.ledger.reserve("job1", 1.2)
            with self.assertRaises(ValueError):
                other.reserve("job2", 1.2)
        finally:
            other.close()

    def test_changed_account_cannot_reuse_campaign_budget(self):
        with self.assertRaises(ValueError):
            Ledger(self.path, **{**self.kwargs, "account_sha256": "b" * 64})

    def test_wrong_price_assumption_recorded_and_stops_future_work(self):
        self.ledger.reserve("job1", .5)
        self.ledger.finish("job1", {"status": "ok"}, observed_cost_cny=3)
        self.assertTrue(self.ledger.report()["cap_exceeded"])
        with self.assertRaises(ValueError):
            self.ledger.reserve("job2", .1)

    def test_rounds_up_and_rejects_nonfinite(self):
        self.assertEqual(micros("0.00000001"), 1)
        with self.assertRaises(ValueError):
            micros(float("nan"))


class WorkerTests(unittest.TestCase):
    def test_teacher_token_count_uses_ids_not_mapping_fields(self):
        for output in ([11, 12, 13], {"input_ids": [11, 12, 13], "attention_mask": [1, 1, 1]},
                       {"input_ids": [[11, 12, 13]], "attention_mask": [[1, 1, 1]]}):
            tokenizer = Mock()
            tokenizer.apply_chat_template.return_value = output
            self.assertEqual(worker.teacher_input_length(tokenizer, [], {}), 3)
            self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs["return_dict"])

    def test_teacher_unknown_shape_fails_closed(self):
        for output in ({"attention_mask": [1, 1]}, [[1], [2]], "not token IDs", [True], [-1]):
            tokenizer = Mock()
            tokenizer.apply_chat_template.return_value = output
            with self.assertRaises((ValueError, KeyError)):
                worker.teacher_input_length(tokenizer, [], {})

    def test_qwen_mapping_context_overflow_makes_zero_network_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            r = registry(); r["deployment_status"] = "verified"
            r["judges"]["qwen"]["endpoint"]["maximum_context_tokens"] = 4096
            reg = root / "registry.json"; write_json(reg, r)
            source = root / "source.jsonl"; source.write_text(canonical(raw()) + "\n")
            path = root / "batch"
            prepare(source, path, ROOT / "configs/curation/p0_policy_v3.json", ROOT / "schemas/p0_teacher_quality_v3.json", PROMPT, reg)
            tokenizer = Mock(); tokenizer.apply_chat_template.return_value = {"input_ids": [1] * 4096, "attention_mask": [1] * 4096}
            argv = ["worker", "--batch", str(path), "--judge", "qwen", "--allow-api", "--ledger", str(root / "local.sqlite"), "--budget-cny", "0", "--limit", "1"]
            with patch.object(sys, "argv", argv), patch.object(worker, "tokenizer_for", return_value=tokenizer), patch.object(worker, "send") as sender:
                self.assertEqual(worker.main(), 3)
                self.assertEqual(sender.call_count, 0)

    def run_fixture(self, maximum_input_tokens=32768):
        """Mock network only. Never a real API/GPU call or production registry."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            r = registry()
            r["deployment_status"] = "verified"  # scoped fixture; live transport is mocked
            r["judges"]["deepseek"]["endpoint"]["maximum_input_tokens"] = maximum_input_tokens
            r["judges"]["deepseek"]["endpoint"]["pricing_cny_per_million"] = {"uncached_input": 2, "output": 8, "margin": 1.15}
            reg = root / "registry.json"
            write_json(reg, r)
            source = root / "source.jsonl"
            source.write_text(canonical(raw()) + "\n")
            path = root / "batch"
            prepare(source, path, ROOT / "configs/curation/p0_policy_v3.json", ROOT / "schemas/p0_teacher_quality_v3.json", PROMPT, reg)
            doc = normalize_document(raw())
            response = {"status": "completed", "model": "deepseek-flash", "id": "offline-response",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": canonical(label(doc))}]}],
                "usage": {"input_tokens": 500, "output_tokens": 100}}
            key = "offline-test-not-a-real-key"
            argv = ["worker", "--batch", str(path), "--judge", "deepseek", "--allow-api", "--ledger", str(root / "campaign.sqlite"),
                "--budget-cny", "1", "--expected-key-sha256", sha(key), "--limit", "1"]
            with patch.object(sys, "argv", argv), patch.dict("os.environ", {"DEEPSEEK_API_KEY": key}), patch.object(worker, "send", return_value=response) as sender:
                first = worker.main()
                second = worker.main()
                calls = sender.call_count
            batch = Batch(path)
            try:
                statuses = batch.report()["job_statuses"]
            finally:
                batch.close()
            return first, second, calls, statuses

    def test_mock_worker_end_to_end_and_resume_no_duplicate_call(self):
        first, second, calls, statuses = self.run_fixture()
        self.assertEqual((first, second, calls), (0, 0, 1))
        self.assertEqual(statuses, {"queued": 1, "valid": 1})

    def test_context_overflow_no_network_no_partial_approval(self):
        first, second, calls, statuses = self.run_fixture(maximum_input_tokens=1)
        self.assertEqual((first, second, calls), (3, 3, 0))
        self.assertEqual(statuses, {"queued": 1, "invalid": 1})

    def test_api_requires_explicit_enable_before_loading_batch(self):
        argv = ["worker", "--batch", "/does/not/exist", "--judge", "deepseek", "--ledger", "/does/not/exist", "--budget-cny", "0"]
        with patch.object(sys, "argv", argv), self.assertRaises(ValueError):
            worker.main()

    def test_no_tools_no_truncation_and_full_payload(self):
        doc = normalize_document(raw())
        judges = registry()["judges"]
        body = worker.request_body(doc, "prompt", SCHEMA, judges["deepseek"])
        self.assertEqual(body["tool_choice"], "none")
        self.assertNotIn("tools", body)
        self.assertEqual(body["input"][0]["content"], doc["review_payload"])
        self.assertEqual(body["reasoning"], {"effort": "none"})

    def test_chat_output_truncation_rejected(self):
        with self.assertRaises(ValueError):
            worker.extract({"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}, "chat")

    def test_responses_incomplete_rejected(self):
        with self.assertRaises(ValueError):
            worker.extract({"status": "incomplete", "output": []}, "responses")

    def test_normal_shape_extracts_json(self):
        self.assertEqual(worker.extract({"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"a":1}'}]}]}, "responses"), {"a": 1})

    def test_usage_estimate_conservative_does_not_assume_cache_savings(self):
        price = {"uncached_input": 2, "output": 8, "margin": 1}
        self.assertAlmostEqual(float(worker.usage_cost({"input_tokens": 1000, "output_tokens": 100, "input_tokens_details": {"cached_tokens": 1000}}, price)), .0028)
        self.assertIsNone(worker.usage_cost({}, price))

    def test_receipt_atomic_idempotent_and_not_indented_json(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "receipt.jsonl"
            worker.write_receipt(p, {"status": "ok", "annotation": {}})
            worker.write_receipt(p, {"status": "ok", "annotation": {}})
            self.assertEqual(len(p.read_text().splitlines()), 1)
            with self.assertRaises(ValueError):
                worker.write_receipt(p, {"status": "changed"})


class PartitionTests(unittest.TestCase):
    def test_streaming_partitions_preserve_all_records_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text("".join(canonical(raw(f"Document {i}: a complete enough teaching example.")) + "\n" for i in range(5)))
            report = cli.partition(source, root / "parts", 2, 10000)
            self.assertEqual([p["documents"] for p in report["parts"]], [2, 2, 1])
            self.assertEqual(report["partitioned_documents"], 5)
            for item in report["parts"]:
                self.assertEqual(file_sha(root / "parts" / item["path"]), item["sha256"])
            with self.assertRaises(ValueError):
                cli.partition(source, root / "parts", 2, 10000)

    def test_oversize_not_silently_truncated_or_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text(canonical(raw("a" * 100)) + "\n" + canonical(raw("small but complete")) + "\n")
            report = cli.partition(source, root / "parts", 2, 50)
            self.assertEqual(report["oversize_quarantined_documents"], 1)
            held = json.loads((root / "parts/oversize.quarantine.jsonl").read_text())
            self.assertEqual(held["document_text"], "a" * 100)

    def test_replay_reports_classification_separately_without_v3_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, results = root / "input.jsonl", root / "results.jsonl"
            source.write_text(canonical(raw()) + "\n")
            a = label(normalize_document(raw()), "technical")
            a["reason_codes"] = []
            results.write_text(canonical({"status": "ok", "document_id": a["document_id"], "annotation": a}) + "\n")
            report = cli.replay_legacy(source, [results], root / "REPLAY.json", None)
            self.assertEqual(report["categories"], {"bucket_only_mismatch_not_quality_verdict": 1})
            self.assertEqual(report["new_v3_valid_reviews"], 0)
            self.assertEqual(report["new_training_approved_documents"], 0)


class SmokePlanTests(unittest.TestCase):
    def test_control_labels_are_not_in_model_payload(self):
        docs = smoke.controls()
        self.assertEqual(len(docs), 8)
        self.assertEqual(len({d["document_id"] for d in docs}), 8)
        for d in docs:
            p = json.loads(normalize_document(d)["review_payload"])
            self.assertNotIn("expected_control_decision", p)
            self.assertNotIn("control_name", p)
            self.assertNotIn("/good", p["source_locator"])
            self.assertNotIn("/bad", p["source_locator"])

    def test_narrow_mechanical_control_defects_are_observable(self):
        docs = {d["control_name"]: normalize_document(d) for d in smoke.controls()}
        for kind in ("dns", "list", "python"):
            self.assertFalse(any(f["severity"] == "hold" for f in docs[kind+"_good"]["lint_flags"]))
            self.assertTrue(any(f["severity"] == "hold" for f in docs[kind+"_bad"]["lint_flags"]))


class CredentialLoaderTests(unittest.TestCase):
    def test_loads_exactly_one_value_without_evaluating_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials.env"
            marker = Path(directory) / "must-not-exist"
            path.write_text(f"touch {marker}\nexport DEEPSEEK_API_KEY='fixture-secret'\n")
            self.assertEqual(credential_env.load_key(path), "fixture-secret")
            self.assertFalse(marker.exists())

    def test_duplicate_missing_or_whitespace_values_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials.env"
            for text in ("OTHER=x", "DEEPSEEK_API_KEY=", "DEEPSEEK_API_KEY=a b",
                         "DEEPSEEK_API_KEY=a\nDEEPSEEK_API_KEY=b"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    credential_env.load_key(path)


class BoundedResumeTests(unittest.TestCase):
    def test_only_known_completed_validation_failures_are_recoverable(self):
        expected = {"body": "frozen"}
        result = {**expected, "status": "error", "error": "ValidationError", "raw_annotation": {},
                  "response_id": "offline-fixture", "served_model": "fixture-model",
                  "usage": {"input_tokens": 100, "output_tokens": 10}}
        self.assertTrue(bounded_resume.recoverable(result, expected, ["fixture-model"]))
        completed_bad_json = {**result, "error": "JSONDecodeError", "raw_annotation": None}
        self.assertTrue(bounded_resume.recoverable(completed_bad_json, expected, ["fixture-model"]))
        self.assertFalse(bounded_resume.recoverable({**completed_bad_json, "usage": None}, expected, ["fixture-model"]))
        self.assertFalse(bounded_resume.recoverable({**result, "raw_annotation": None}, expected, ["fixture-model"]))
        self.assertFalse(bounded_resume.recoverable(None, expected, ["fixture-model"]))
        for field, value in (("body", "changed"), ("error", "HTTP 429"), ("raw_annotation", None),
                             ("served_model", "unexpected"), ("usage", None), ("response_id", None),
                             ("usage", {"input_tokens": -1, "output_tokens": 10})):
            bad = {**result, field: value}
            self.assertFalse(bounded_resume.recoverable(bad, expected, ["fixture-model"]))


class DecoderProjectionTests(unittest.TestCase):
    def test_literal_catalog_only_contains_unique_exact_body_substrings(self):
        for control in smoke.controls():
            text = control["document_text"]
            literals = evidence_catalog(text)
            self.assertEqual(len(literals), len(set(literals)))
            for q in literals:
                self.assertTrue(8 <= len(q) <= 320)
                self.assertGreaterEqual(text.find(q), 0)
                self.assertEqual(text.find(q), text.rfind(q))

    def test_literal_catalog_rejects_flattened_quote_without_repair(self):
        doc = normalize_document(smoke.controls()[0])
        judge = copy.deepcopy(read_json(ROOT / "configs/curation/p0_judges_v3.example.json")["judges"]["qwen"])
        judge["generation"]["decoder_schema_projection"] = "literal_catalog_v1"
        wire = decoder_schema(doc, SCHEMA, judge)
        a = label(doc)
        a["anchors"] = ["A reverse DNS name represents an IPv4 address by putting its four octets in reverse order and appending the suffix in-addr.arpa."]
        before = copy.deepcopy(a)
        with self.assertRaises(Exception):
            validate_annotation(a, doc, wire)
        self.assertEqual(a, before)
        a["anchors"] = [wire["properties"]["anchors"]["items"]["enum"][0]]
        validate_annotation(a, doc, wire)
        validate_annotation(a, doc, SCHEMA)
        self.assertTrue(SCHEMA["properties"]["anchors"]["uniqueItems"])

    def test_literal_catalog_large_or_no_unique_body_fails_closed(self):
        for text in ("tiny", "x" * 131073, "abcdefghij" * 1000):
            with self.assertRaises(ValueError):
                evidence_catalog(text)

    def test_projection_only_changes_wire_grammar_not_validation_schema(self):
        judge = copy.deepcopy(read_json(ROOT / "configs/curation/p0_judges_v3.example.json")["judges"]["qwen"])
        judge["generation"]["decoder_schema_projection"] = "omit_uniqueItems_v1"
        before = copy.deepcopy(SCHEMA)
        doc = normalize_document(raw())
        wire = worker.request_body(doc, "prompt", SCHEMA, judge)["response_format"]["json_schema"]["schema"]
        self.assertNotIn("uniqueItems", wire["properties"]["anchors"])
        self.assertEqual(SCHEMA, before)
        self.assertTrue(SCHEMA["properties"]["anchors"]["uniqueItems"])
        a = label(doc)
        a["anchors"] = [a["anchors"][0]] * 2
        with self.assertRaises(Exception):
            validate_annotation(a, doc, SCHEMA)

    def test_projection_changes_inference_identity(self):
        judge = copy.deepcopy(read_json(ROOT / "configs/curation/p0_judges_v3.example.json")["judges"]["qwen"])
        doc = normalize_document(raw())
        original = protocol(doc, "prompt", SCHEMA, judge)
        judge["generation"]["decoder_schema_projection"] = "omit_uniqueItems_v1"
        changed = protocol(doc, "prompt", SCHEMA, judge)
        self.assertNotEqual(original["inference_sha256"], changed["inference_sha256"])
        self.assertEqual(original["schema_sha256"], changed["schema_sha256"])

    def test_model_revision_and_endpoint_bound_into_receipt_identity(self):
        judge = copy.deepcopy(read_json(ROOT / "configs/curation/p0_judges_v3.example.json")["judges"]["qwen"])
        doc = normalize_document(raw())
        original = protocol(doc, "prompt", SCHEMA, judge)
        for field, value in (("model_revision", "different-local-weight-snapshot"),):
            changed = copy.deepcopy(judge)
            changed[field] = value
            self.assertNotEqual(original["judge_identity_sha256"], protocol(doc, "prompt", SCHEMA, changed)["judge_identity_sha256"])
        changed = copy.deepcopy(judge)
        changed["endpoint"]["base_url"] = "http://127.0.0.1:19999/v1"
        self.assertNotEqual(original["judge_identity_sha256"], protocol(doc, "prompt", SCHEMA, changed)["judge_identity_sha256"])


if __name__ == "__main__":
    unittest.main()
