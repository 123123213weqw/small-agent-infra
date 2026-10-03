#!/usr/bin/env python3
"""Small-batch supervisor: quarantine known invalid outputs, never API-resend them.

Explicit opt-in only. Unknown transport/billing, changed identity, absent pilot,
or conflicts stop the supervisor. This does not approve a corpus or train.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from small_agent.curation.core import fingerprint, sha
from small_agent.curation.store import Batch, file_sha, protocol, write_json


def recoverable(result: dict | None, expected: dict, served_models: list[str]) -> bool:
    if not isinstance(result, dict) or any(result.get(k) != v for k, v in expected.items()):
        return False
    usage = result.get("usage")
    # JSONDecodeError with response ID + usage is a returned response whose
    # inner output JSON could not be parsed, not an unknown transport request.
    known_invalid_output = (result.get("error") in ("ValueError", "ValidationError")
        and isinstance(result.get("raw_annotation"), dict)) or result.get("error") == "JSONDecodeError"
    return (result.get("status") == "error" and known_invalid_output and bool(result.get("response_id"))
        and result.get("served_model") in served_models and isinstance(usage, dict)
        and type(usage.get("input_tokens", usage.get("prompt_tokens"))) is int
        and type(usage.get("output_tokens", usage.get("completion_tokens"))) is int
        and min(usage.get("input_tokens", usage.get("prompt_tokens")),
                usage.get("output_tokens", usage.get("completion_tokens"))) >= 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--judge", required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--budget-cny", required=True)
    p.add_argument("--expected-key-sha256", default="")
    p.add_argument("--credentials-env", type=Path)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-windows", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--continue-known-validation-failures", action="store_true")
    a = p.parse_args()
    if not a.continue_known_validation_failures or not 1 <= a.max_windows <= 8 or a.output.exists():
        raise ValueError("explicit known-failure continuation, bounded windows and new report required")
    if a.credentials_env:
        spec = importlib.util.spec_from_file_location("private_env", ROOT / "scripts/p0_curation_deepseek_env_v3.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        key = module.load_key(a.credentials_env)
        if sha(key) != a.expected_key_sha256:
            raise ValueError("account fingerprint mismatch")
        os.environ["DEEPSEEK_API_KEY"] = key
    history, final, prior_queued = [], None, None
    for window in range(a.max_windows + 1):
        batch = Batch(a.batch)
        try:
            if len(batch.docs) > 40:
                raise ValueError("supervisor is smoke-only, maximum 40 documents")
            judge = batch.registry["judges"][a.judge]
            statuses = dict(batch.db.execute("SELECT status,COUNT(*) FROM jobs WHERE judge=? GROUP BY status", (a.judge,)).fetchall())
            if statuses.get("conflict") or not statuses.get("valid"):
                raise ValueError("conflict or no previously valid pilot: no continuation")
            db = sqlite3.connect(f"file:{a.ledger}?mode=ro", uri=True)
            try:
                if db.execute("SELECT COUNT(*) FROM requests WHERE state=?", ("inflight",)).fetchone()[0]:
                    raise ValueError("unknown inflight request: no automatic continuation")
                for row in batch.db.execute("SELECT document_id FROM jobs WHERE judge=? AND status=?", (a.judge, "invalid")):
                    expected = protocol(batch.docs[row[0]], batch.prompt, batch.schema, judge)
                    job = fingerprint({"batch": batch.manifest_hash, "judge": a.judge, "protocol": expected})
                    result_row = db.execute("SELECT result_json FROM requests WHERE job_key=?", (job,)).fetchone()
                    result = json.loads(result_row[0]) if result_row and result_row[0] else None
                    if not recoverable(result, expected, judge["served_models"]):
                        raise ValueError("invalid output not a known recoverable validation failure")
            finally:
                db.close()
            queued = statuses.get("queued", 0)
            if queued == 0:
                final = statuses
                break
            if window == a.max_windows or (prior_queued is not None and queued >= prior_queued):
                raise ValueError("bounded continuation exhausted or made no progress")
            prior_queued = queued
            history.append({"window": window, "queued_before": queued, "known_invalid_not_resent": statuses.get("invalid", 0),
                            "batch_manifest_sha256": batch.manifest_hash})
        finally:
            batch.close()
        command = [sys.executable, str(ROOT / "scripts/p0_curation_worker_v3.py"), "--allow-api",
            "--batch", str(a.batch), "--judge", a.judge, "--ledger", str(a.ledger),
            "--budget-cny", a.budget_cny, "--expected-key-sha256", a.expected_key_sha256,
            "--workers", str(a.workers), "--limit", "40"]
        history[-1]["worker_exit_code"] = subprocess.run(command, check=False).returncode
    out = {"stage": "all_jobs_attempted_quarantines_retained", "job_statuses": final,
        "history": history, "supervisor_sha256": file_sha(Path(__file__)),
        "automatic_retry_of_known_failed_api_calls": False, "training_approved": False, "training_started": False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.output, out)
    print(json.dumps(out, ensure_ascii=False))
    return 3 if final.get("invalid") else 0


if __name__ == "__main__":
    raise SystemExit(main())
