"""Immutable batch inputs plus transactional, append-only review receipts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone

from .core import canonical, decide, decoder_schema, finite_json, fingerprint, normalize_document, sha, summary, validate_annotation


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"not an object: {path}:{number}")
                yield item


def file_sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def engine_files() -> dict:
    repo = Path(__file__).resolve().parents[3]
    names = ("src/small_agent/curation/core.py", "src/small_agent/curation/store.py",
             "src/small_agent/curation/ledger.py", "scripts/p0_curation_v3.py",
             "scripts/p0_curation_worker_v3.py")
    return {name: file_sha(repo / name) for name in names}


def payload_for(doc: dict) -> str:
    return canonical({k: doc[k] for k in ("document_id", "body_sha256", "source_id", "source_locator",
                                           "language", "content_kind", "document_text")})


def validate_config(policy: dict, registry: dict) -> None:
    finite_json(policy)
    finite_json(registry)
    if policy.get("version") != 3 or registry.get("version") != 3:
        raise ValueError("expected v3 policy and registry")
    if registry.get("deployment_status") not in ("verified", "offline_test_only"):
        raise ValueError("verify model deployment/revision first; example registry cannot run")
    judges = registry.get("judges", {})
    purpose = policy.get("review_purpose", "legacy_dual_document_review")
    if purpose == "source_quality_audit":
        if (set(judges) != {"deepseek"} or judges["deepseek"]["family"] != "deepseek"
                or policy.get("corpus_release_allowed") is not False
                or policy.get("require_independent_model_families") is not False):
            raise ValueError("source audit is single DeepSeek, diagnostic only, and cannot release a corpus")
    elif purpose == "legacy_dual_document_review":
        if len(judges) != 2 or len({j["family"] for j in judges.values()}) != 2:
            raise ValueError("exactly two different model families required")
    else:
        raise ValueError("unknown review purpose; cannot silently relax legacy policy")
    if policy.get("review_mode") != "full_document_only" or policy.get("auto_training") is not False:
        raise ValueError("only full-document review; automatic training forbidden")
    if not 0 < policy["minimum_confidence"] <= 1:
        raise ValueError("invalid confidence threshold")
    if set(policy["minimum_scores"]) != {"quality", "completeness", "educational_value", "format_integrity"}:
        raise ValueError("four quality thresholds required")
    if any(type(v) is not int or not 1 <= v <= 5 for v in policy["minimum_scores"].values()):
        raise ValueError("invalid score thresholds")
    required = {"source_provenance", "license_internal", "exact_and_near_dedup", "split_isolation",
                "eval_contamination_scoped", "independent_calibration"}
    if set(policy.get("required_release_checks", [])) != required:
        raise ValueError("cannot omit a required release check")
    for name, judge in judges.items():
        if not name.replace("_", "").isalnum():
            raise ValueError("invalid judge name")
        if not judge.get("model") or not judge.get("model_revision") or "REPLACE_" in judge["model_revision"]:
            raise ValueError("verified model identity/revision required")
        if not judge.get("served_models"):
            raise ValueError("served model aliases required")
        g = judge["generation"]
        if (g.get("temperature") != 0 or g.get("require_document_id_match") is not True
                or g.get("response_format") != "json_schema" or g.get("max_tokens", 0) < 1):
            raise ValueError("temperature 0, ID binding, schema mode, positive output cap required")
        projection = g.get("decoder_schema_projection", "identity")
        if projection not in ("identity", "omit_uniqueItems_v1", "literal_catalog_v1"):
            raise ValueError("unknown decoder schema projection")
        if projection != "identity" and judge.get("endpoint", {}).get("kind") != "chat":
            raise ValueError("decoder projection is only supported for local chat")


def protocol(doc: dict, prompt: str, schema: dict, judge: dict) -> dict:
    g = judge["generation"]
    inference = {k: g[k] for k in ("chat_template_kwargs", "response_format", "thinking", "max_tokens", "require_document_id_match")}
    if "decoder_schema_projection" in g:
        inference["decoder_schema_projection"] = g["decoder_schema_projection"]
    # Legacy-compatible component hashes, plus mandatory verified teacher identity.
    return {"source_id": canonical([doc["document_id"]]), "text_sha256": sha(doc["review_payload"]),
            "prompt_sha256": sha(prompt.strip()), "schema_sha256": fingerprint(schema),
            "model": judge["model"], "temperature": g["temperature"],
            "inference_sha256": fingerprint(inference), "judge_identity_sha256": fingerprint(judge),
            "wire_schema_sha256": fingerprint(decoder_schema(doc, schema, judge))}


def prepare(source: Path, output: Path, policy_path: Path, schema_path: Path, prompt_path: Path, registry_path: Path) -> dict:
    if output.exists():
        # Resume is explicit via status/import/export; never silently reinterpret input.
        raise FileExistsError(f"immutable batch exists: {output}")
    policy, schema, registry = read_json(policy_path), read_json(schema_path), read_json(registry_path)
    validate_config(policy, registry)
    from jsonschema import Draft202012Validator
    Draft202012Validator.check_schema(schema)
    if schema.get("$id") != "p0_teacher_quality_v3":
        raise ValueError("cannot migrate a legacy annotation schema into v3")
    source_hash = file_sha(source)
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("empty prompt")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=output.parent, prefix=".curation-v3-"))
    try:
        for name, value in (("policy.json", policy), ("schema.json", schema), ("judges.json", registry)):
            write_json(tmp / name, value)
        (tmp / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
        seen, doc_count, tokens, unknown, body_bytes = set(), 0, 0, 0, 0
        with (tmp / "documents.jsonl").open("w", encoding="utf-8") as f:
            for raw in read_jsonl(source):
                doc = normalize_document(raw)
                body_bytes += len(doc["document_text"].encode("utf-8"))
                if doc_count >= policy["maximum_batch_documents"] or body_bytes > policy["maximum_batch_body_bytes"]:
                    raise ValueError("bounded batch limit reached; partition upstream into immutable batches")
                if doc["document_id"] in seen:
                    raise ValueError("duplicate full-body hash; globally reconcile provenance before batch admission")
                seen.add(doc["document_id"])
                # Store the body once; deterministically derive the model payload.
                f.write(canonical({k: v for k, v in doc.items() if k != "review_payload"}) + "\n")
                doc_count += 1
                if doc["token_count"] is None:
                    unknown += 1
                else:
                    tokens += doc["token_count"]
        if doc_count == 0 or file_sha(source) != source_hash:
            raise ValueError("empty input or input changed while preparing")
        frozen = {name: file_sha(tmp / name) for name in ("policy.json", "schema.json", "judges.json", "prompt.txt", "documents.jsonl")}
        manifest = {"version": 3, "stage": "review_input_frozen", "source_jsonl_sha256": source_hash,
                    "files": frozen, "engine_files": engine_files(), "documents": doc_count, "reported_body_tokens": tokens,
                    "unknown_token_count_documents": unknown, "training_approved": False,
                    "created_at": datetime.now(timezone.utc).isoformat()}
        write_json(tmp / "MANIFEST.json", manifest)
        with sqlite3.connect(tmp / "state.sqlite") as db:
            db.executescript("""
                CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE jobs(document_id TEXT, judge TEXT, status TEXT NOT NULL,
                                  annotation TEXT, grounded TEXT, PRIMARY KEY(document_id,judge));
                CREATE TABLE attempts(result_sha256 TEXT PRIMARY KEY, document_id TEXT, judge TEXT,
                                      status TEXT NOT NULL, reason TEXT, result_json TEXT NOT NULL,
                                      imported_at TEXT NOT NULL);
            """)
            db.execute("INSERT INTO meta VALUES (?,?)", ("manifest_sha256", file_sha(tmp / "MANIFEST.json")))
            db.executemany("INSERT INTO jobs VALUES (?,?, 'queued',NULL,NULL)", ((d, j) for d in sorted(seen) for j in registry["judges"]))
        os.rename(tmp, output)
        return manifest
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


class Batch:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.manifest = read_json(root / "MANIFEST.json")
        self.manifest_hash = file_sha(root / "MANIFEST.json")
        if self.manifest.get("engine_files") != engine_files():
            raise ValueError("curation implementation changed/untracked; explicitly create a new versioned batch")
        if set(self.manifest["files"]) != {"policy.json", "schema.json", "judges.json", "prompt.txt", "documents.jsonl"}:
            raise ValueError("unexpected manifest files")
        for name, expected in self.manifest["files"].items():
            if file_sha(root / name) != expected:
                raise ValueError(f"immutable batch input changed: {name}")
        self.policy, self.schema, self.registry = [read_json(root / name) for name in ("policy.json", "schema.json", "judges.json")]
        validate_config(self.policy, self.registry)
        self.prompt = (root / "prompt.txt").read_text(encoding="utf-8").strip()
        self.docs = {d["document_id"]: d for d in read_jsonl(root / "documents.jsonl")}
        for doc in self.docs.values():
            doc["review_payload"] = payload_for(doc)
        # Do not create a new blank database when state disappeared.
        self.db = sqlite3.connect(f"file:{root / 'state.sqlite'}?mode=rw", uri=True, timeout=30)
        self.db.row_factory = sqlite3.Row
        meta = self.db.execute("SELECT value FROM meta WHERE key='manifest_sha256'").fetchone()
        if not meta or meta[0] != self.manifest_hash:
            self.db.close()
            raise ValueError("database belongs to a different frozen batch")

    def close(self):
        self.db.close()

    def pending(self, judge: str) -> list[dict]:
        if judge not in self.registry["judges"]:
            raise ValueError("unknown judge")
        ids = [r[0] for r in self.db.execute("SELECT document_id FROM jobs WHERE judge=? AND status IN ('queued','invalid') ORDER BY document_id", (judge,))]
        return [{"document_id": d, "body_sha256": d, "review_payload": self.docs[d]["review_payload"]} for d in ids]

    def import_results(self, judge_name: str, results: Path) -> dict:
        judge = self.registry["judges"][judge_name]
        tally = Counter()
        for result in read_jsonl(results):
            finite_json(result)
            receipt_sha = fingerprint({"judge": judge_name, "result": result})
            # Per-receipt transaction: killed import resumes without losing prior receipts.
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                if self.db.execute("SELECT 1 FROM attempts WHERE result_sha256=?", (receipt_sha,)).fetchone():
                    tally["already_imported"] += 1
                    continue
                doc_id, grounded, annotation = None, None, None
                try:
                    ids = json.loads(result["source_id"])
                    if not isinstance(ids, list) or len(ids) != 1 or ids[0] not in self.docs:
                        raise ValueError("unknown source ID")
                    doc_id = ids[0]
                    doc = self.docs[doc_id]
                    expected = protocol(doc, self.prompt, self.schema, judge)
                    if any(result.get(k) != v for k, v in expected.items()):
                        raise ValueError("protocol mismatch: body/prompt/schema/model/generation")
                    if result.get("status") != "ok":
                        raise ValueError("model request did not succeed")
                    if result.get("served_model") not in judge["served_models"]:
                        raise ValueError("unexpected served model")
                    annotation = result["annotation"]
                    grounded = validate_annotation(annotation, doc, self.schema)
                    validate_annotation(annotation, doc, decoder_schema(doc, self.schema, judge))
                    state, reason = "valid", None
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    state, reason = "invalid", str(exc)[:500]
                except Exception as exc:
                    from jsonschema import ValidationError
                    if not isinstance(exc, ValidationError):
                        raise
                    state, reason = "invalid", f"schema: {exc.message}"[:500]
                self.db.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?,?)", (
                    receipt_sha, doc_id, judge_name, state, reason, canonical(result), datetime.now(timezone.utc).isoformat()))
                if doc_id is not None:
                    old = self.db.execute("SELECT status,annotation FROM jobs WHERE document_id=? AND judge=?", (doc_id, judge_name)).fetchone()
                    if state == "valid":
                        if old["status"] == "conflict" or (old["annotation"] is not None and old["annotation"] != canonical(annotation)):
                            self.db.execute("UPDATE jobs SET status='conflict' WHERE document_id=? AND judge=?", (doc_id, judge_name))
                            state = "conflict"
                        else:
                            self.db.execute("UPDATE jobs SET status='valid',annotation=?,grounded=? WHERE document_id=? AND judge=?", (
                                canonical(annotation), canonical(grounded), doc_id, judge_name))
                    elif old["status"] not in ("valid", "conflict"):
                        self.db.execute("UPDATE jobs SET status='invalid' WHERE document_id=? AND judge=?", (doc_id, judge_name))
                tally[state] += 1
        return dict(tally)

    def decisions(self) -> list[dict]:
        reviews = {}
        for row in self.db.execute("SELECT * FROM jobs"):
            reviews.setdefault(row["document_id"], {})[row["judge"]] = {
                "status": row["status"], "annotation": json.loads(row["annotation"]) if row["annotation"] else None}
        return [decide(self.docs[d], reviews[d], self.registry["judges"], self.policy) for d in sorted(self.docs)]

    def report(self) -> dict:
        out = summary(self.decisions())
        out.update(review_purpose=self.policy.get("review_purpose", "legacy_dual_document_review"),
                   source_approved=False,
                   batch_manifest_sha256=self.manifest_hash,
                   review_state_sha256=self.state_fingerprint(),
                   job_statuses=dict(self.db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status").fetchall()),
                   receipt_statuses=dict(self.db.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status").fetchall()))
        return out

    def state_fingerprint(self) -> str:
        return fingerprint([dict(r) for r in self.db.execute("SELECT * FROM jobs ORDER BY document_id,judge")])

    def export(self, output: Path) -> dict:
        if self.policy.get("review_purpose") == "source_quality_audit":
            raise ValueError("source audit cannot export an eligible corpus; export its diagnostic report instead")
        if output.exists():
            raise FileExistsError("exports are immutable; choose a new snapshot directory")
        # Pin one transactional review-state snapshot while assembling the export.
        self.db.execute("BEGIN IMMEDIATE")
        rows = self.decisions()
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=output.parent, prefix=".curation-export-"))
        try:
            with (tmp / "decisions.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(canonical(r) + "\n")
            buckets = {r["document_id"]: r["bucket"] for r in rows if r["status"] == "candidate_eligible"}
            eligible = list(buckets)
            with (tmp / "eligible.jsonl").open("w", encoding="utf-8") as f:
                for d in eligible:
                    f.write(canonical({**{k: v for k, v in self.docs[d].items() if k != "review_payload"}, "bucket": buckets[d],
                                       "stage": "candidate_eligible", "training_approved": False}) + "\n")
            report = self.report()
            report.update(stage="candidate_snapshot_not_approved", eligible_ids_sha256=fingerprint(eligible),
                          files={n: file_sha(tmp / n) for n in ("decisions.jsonl", "eligible.jsonl")})
            write_json(tmp / "REPORT.json", report)
            os.rename(tmp, output)
            self.db.commit()
            return report
        except BaseException:
            self.db.rollback()
            shutil.rmtree(tmp, ignore_errors=True)
            raise


def check_release(batch: Batch, snapshot: Path, checks_path: Path) -> dict:
    """Validate scope/hash bindings of external checker receipts, not their truth.

    Checkers must independently run dedup, split/leakage, source/license and
    reference calibration. An arbitrary hand-written 'passed' is not a measurement.
    This function DOES NOT write a training-approved corpus or launch training.
    """
    if batch.policy.get("review_purpose") == "source_quality_audit":
        raise ValueError("single-teacher source audit can never serve as corpus release approval")
    report = read_json(snapshot / "REPORT.json")
    if (report["batch_manifest_sha256"] != batch.manifest_hash
            or report["review_state_sha256"] != batch.state_fingerprint()):
        raise ValueError("snapshot is for another batch or a stale review state")
    if set(report["files"]) != {"eligible.jsonl", "decisions.jsonl"}:
        raise ValueError("incomplete snapshot files")
    for name, expected in report["files"].items():
        if name not in ("eligible.jsonl", "decisions.jsonl") or file_sha(snapshot / name) != expected:
            raise ValueError("candidate snapshot changed")
    checks = read_json(checks_path)
    required = {"source_provenance", "license_internal", "exact_and_near_dedup", "split_isolation",
                "eval_contamination_scoped", "independent_calibration"}
    found, errors = set(), []
    if batch.registry["deployment_status"] != "verified":
        errors.append("offline test identities cannot release production data")
    if not report["status_documents"].get("candidate_eligible", 0):
        errors.append("no eligible documents")
    for receipt in checks.get("checks", []):
        name = receipt.get("name")
        if name not in required or name in found:
            errors.append(f"unknown/duplicate release check: {name}")
            continue
        found.add(name)
        if (receipt.get("status") != "passed" or not receipt.get("scope") or not receipt.get("checker_revision")
                or receipt.get("batch_manifest_sha256") != report["batch_manifest_sha256"]
                or receipt.get("eligible_ids_sha256") != report["eligible_ids_sha256"]
                or not receipt.get("evidence_files")):
            errors.append(f"missing/failed/stale check: {name}")
            continue
        for item in receipt["evidence_files"]:
            path = Path(item["path"])
            if not path.is_absolute() or not path.is_file() or file_sha(path) != item.get("sha256"):
                errors.append(f"missing/changed evidence file: {name}")
        if name == "independent_calibration":
            if (receipt.get("reference_type") != "independently_verified_reference"
                    or receipt.get("heldout_source_disjoint") is not True
                    or type(receipt.get("reference_documents")) is not int
                    or receipt["reference_documents"] < 1):
                errors.append("model agreement/regression fixtures cannot substitute for calibration reference")
    errors.extend(f"missing release check: {name}" for name in sorted(required - found))
    return {"stage": "release_checks_bound" if not errors else "release_blocked", "errors": errors,
            "release_checks_bound": not errors, "training_ready": False, "training_started": False,
            "next": "trusted measured checker receipts, freeze text, pack, verify hashes and run trainctl preflight",
            "interpretation": "hash binding alone does not authenticate or validate a checker's scientific conclusions"}
