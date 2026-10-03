#!/usr/bin/env python3
"""Explicitly enabled teacher worker; consumes v3 jobs and never starts training.

No automatic ambiguous retries. One pilot precedes bounded concurrent windows.
Model services are managed separately; this script never kills GPU processes.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from small_agent.curation.core import canonical, decoder_schema, fingerprint, sha, validate_annotation
from small_agent.curation.ledger import Ledger
from small_agent.curation.store import Batch, file_sha, protocol, write_json


def request_body(doc: dict, prompt: str, schema: dict, judge: dict) -> dict:
    g = judge["generation"]
    schema = decoder_schema(doc, schema, judge)
    if judge["endpoint"]["kind"] == "responses":
        if g["thinking"] != "disabled" or g["chat_template_kwargs"]:
            raise ValueError("Responses worker supports reasoning=none and no local chat kwargs")
        return {"model": judge["model"], "instructions": prompt,
                "input": [{"role": "user", "content": doc["review_payload"]}],
                "reasoning": {"effort": "none"}, "temperature": g["temperature"],
                "max_output_tokens": g["max_tokens"], "stream": False, "tool_choice": "none",
                "text": {"format": {"type": "json_schema", "name": "quality_v3", "schema": schema}}}
    if judge["endpoint"]["kind"] != "chat":
        raise ValueError("unsupported endpoint kind")
    body = {"model": judge["model"], "messages": [{"role": "system", "content": prompt},
                {"role": "user", "content": doc["review_payload"]}], "temperature": g["temperature"],
            "max_tokens": g["max_tokens"], "stream": False,
            "response_format": {"type": "json_schema", "json_schema": {"name": "quality_v3", "schema": schema}}}
    if g["thinking"] != "omit":
        body["thinking"] = {"type": g["thinking"]}
    if g["chat_template_kwargs"]:
        body["chat_template_kwargs"] = g["chat_template_kwargs"]
    return body


def extract(response: dict, kind: str) -> dict:
    if kind == "responses":
        parts = [c["text"] for item in response.get("output", []) if item.get("type") == "message"
                 for c in item.get("content", []) if c.get("type") == "output_text"]
        if response.get("status") != "completed" or len(parts) != 1:
            raise ValueError("response incomplete or wrong output shape")
        return json.loads(parts[0])
    choices = response.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("chat did not finish normally; no partial labels")
    return json.loads(choices[0]["message"]["content"])


def send(base_url: str, kind: str, body: dict, key: str, timeout: int):
    url = base_url.rstrip("/") + ("/responses" if kind == "responses" else "/chat/completions")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = Request(url, data=canonical(body).encode("utf-8"), headers=headers)
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None  # never forward a credential to a redirected destination
    with build_opener(NoRedirect()).open(req, timeout=timeout) as response:
        # Bound output bytes too; keys and request headers are never recorded.
        raw = response.read(8 << 20)
        if response.read(1):
            raise ValueError("oversized API response")
        return json.loads(raw)


def usage_cost(usage: dict | None, pricing: dict):
    # Charge reservation if usage is absent. Cache discount is deliberately ignored
    # for this conservative budget, even though separate cache statistics may exist.
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    if type(input_tokens) is not int or type(output_tokens) is not int or min(input_tokens, output_tokens) < 0:
        return None
    return ((Decimal(input_tokens) * Decimal(str(pricing["uncached_input"]))
             + Decimal(output_tokens) * Decimal(str(pricing["output"])))
            * Decimal(str(pricing["margin"])) / 1_000_000)


def tokenizer_for(judge: dict):
    ep = judge["endpoint"]
    root = Path(ep["teacher_tokenizer_directory"])
    files = ep["teacher_tokenizer_files"]
    if not {"tokenizer.json", "tokenizer_config.json"} <= set(files):
        raise ValueError("verify teacher tokenizer and chat-template/config files first")
    for name, digest in files.items():
        p = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts or file_sha(p) != digest:
            raise ValueError("teacher tokenizer changed")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)


def teacher_input_length(tokenizer, messages: list, kwargs: dict) -> int:
    # Transformers 5 may return BatchEncoding by default; len(mapping) is the
    # number of fields, NOT the token count. Ask for a list and still validate
    # mappings/batched shapes for compatible versions. Never silently use 2.
    encoded = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, return_dict=False, **kwargs)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, (list, tuple)) and len(ids) == 1 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    if not isinstance(ids, (list, tuple)) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("unknown teacher tokenizer shape; refuse admission")
    return len(ids)


def atomic_status(path: Path, value):
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".status-")
    os.close(fd)
    try:
        write_json(Path(temp), value)
        os.replace(temp, path)
    finally:
        if Path(temp).exists():
            Path(temp).unlink()


def write_receipt(path: Path, value):
    content = canonical(value) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError("existing receipt differs from durable ledger result")
        return
    # Publish one compact record atomically; never leave a partial JSONL line.
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".receipt-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if Path(temp).exists():
            Path(temp).unlink()


def job_identity(batch, judge_name: str, base: dict) -> str:
    # Share exact-protocol receipts between source-audit rounds. Preserve legacy
    # per-batch identities so old invoices/receipts are never reinterpreted.
    if batch.policy.get("review_purpose") == "source_quality_audit":
        return fingerprint({"purpose": "source_quality_audit", "judge": judge_name, "protocol": base})
    return fingerprint({"batch": batch.manifest_hash, "judge": judge_name, "protocol": base})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--judge", required=True)
    p.add_argument("--allow-api", action="store_true", help="otherwise no network request is allowed")
    p.add_argument("--ledger", type=Path, required=True, help="one shared campaign ledger, not one cap per batch")
    p.add_argument("--budget-cny", required=True)
    p.add_argument("--expected-key-sha256", default="")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--timeout", type=int, default=240)
    args = p.parse_args()
    if not args.allow_api:
        raise ValueError("explicit --allow-api required; no teacher called")
    if not 1 <= args.workers <= 8 or not 1 <= args.limit <= 2000:
        raise ValueError("workers 1..8, limit 1..2000")
    batch = Batch(args.batch)
    try:
        if batch.registry["deployment_status"] != "verified":
            raise ValueError("offline/test registry may not call a live teacher")
        judge = batch.registry["judges"][args.judge]
        ep, kind = judge["endpoint"], judge["endpoint"]["kind"]
        work = args.batch / "worker" / args.judge
        work.mkdir(parents=True, exist_ok=True)
        lock = (work / "worker.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if kind == "responses":
            # Do not send a paid credential to arbitrary endpoints or downgrade TLS.
            if ep["base_url"].rstrip("/") != "https://api.deepseek.com":
                raise ValueError("paid endpoint must be official HTTPS DeepSeek")
            key = os.environ.get(ep["key_env"], "")
            if not key or sha(key) != args.expected_key_sha256:
                raise ValueError("configured account differs; key never printed")
            pricing = ep["pricing_cny_per_million"]
            if min(Decimal(str(pricing[k])) for k in ("uncached_input", "output")) <= 0 or Decimal(str(pricing["margin"])) < 1:
                raise ValueError("verified positive upper prices and margin >=1 required")
            account, tokenizer = sha(key), None
            bound = usage_cost({"input_tokens": ep["maximum_input_tokens"], "output_tokens": judge["generation"]["max_tokens"]}, pricing)
        elif kind == "chat":
            parsed = urlparse(ep["base_url"])
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("invalid local service URL")
            key, pricing = "", {"uncached_input": 0, "output": 0, "margin": 1}
            account, bound = fingerprint({"local_endpoint": ep["base_url"]}), 0
            tokenizer = tokenizer_for(judge)
        else:
            raise ValueError("unsupported endpoint")
        ledger = Ledger(args.ledger, budget_cny=args.budget_cny, account_sha256=account, pricing=pricing)
        try:
            pending = batch.pending(args.judge)[:args.limit]
            done, blocked, inflight_unknown = 0, 0, 0
            index = 0
            pilot = True
            atomic_status(work / "STATUS.json", {"stage": "running" if pending else "idle_no_jobs",
                "selected": len(pending), "ledger": ledger.report(), "batch": batch.report(), "training_started": False})
            while index < len(pending):
                window = pending[index:index + (1 if pilot else args.workers)]
                tasks = []
                for job in window:
                    doc = batch.docs[job["document_id"]]
                    body = request_body(doc, batch.prompt, batch.schema, judge)
                    base = protocol(doc, batch.prompt, batch.schema, judge)
                    job_key = job_identity(batch, args.judge, base)
                    old = ledger.result(job_key)
                    if old is not None:
                        # Even errors are not automatically paid for again. The operator
                        # inspects and explicitly versions/authorizes a new attempt.
                        receipt = work / (job_key + ".jsonl")
                        write_receipt(receipt, old)
                        batch.import_results(args.judge, receipt)
                        done += 1
                        continue
                    if kind == "chat":
                        input_tokens = teacher_input_length(tokenizer, body["messages"], judge["generation"]["chat_template_kwargs"])
                        too_long = input_tokens + judge["generation"]["max_tokens"] + 256 > ep["maximum_context_tokens"]
                    else:
                        # Conservative admission, not a DeepSeek tokenizer claim.
                        # Input JSON framing/schema included. Overlong -> hold, no truncation.
                        too_long = len(canonical(body).encode("utf-8")) + 4096 > ep["maximum_input_tokens"]
                    if too_long:
                        blocked += 1
                        result = {**base, "status": "error", "error_code": "full_body_context_overflow"}
                        receipt = work / (job_key + ".context-blocked.jsonl")
                        write_receipt(receipt, result)
                        batch.import_results(args.judge, receipt)
                        continue
                    if not ledger.reserve(job_key, bound):
                        inflight_unknown += 1
                        continue
                    tasks.append((job_key, doc, body, base))

                def call(task):
                    job_key, doc, body, base = task
                    result = {**base, "status": "error", "generated_at_unix": time.time()}
                    try:
                        response = send(ep["base_url"], kind, body, key, args.timeout)
                        result.update(usage=response.get("usage"), response_id=response.get("id"), served_model=response.get("model"))
                        a = extract(response, kind)
                        # Preserve rejected output for diagnosis, never as a valid label.
                        result["raw_annotation"] = a
                        validate_annotation(a, doc, batch.schema)
                        validate_annotation(a, doc, decoder_schema(doc, batch.schema, judge))
                        result.update(status="ok", annotation=a)
                    except HTTPError as exc:
                        result["error"] = f"HTTP {exc.code}"  # no provider echo of text/key
                    except Exception as exc:
                        result["error"] = type(exc).__name__  # no exception payload secrets
                    return job_key, result

                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    for job_key, result in executor.map(call, tasks):
                        cost = usage_cost(result.get("usage"), pricing) if kind == "responses" else 0
                        ledger.finish(job_key, result, observed_cost_cny=cost)
                        receipt = work / (job_key + ".jsonl")
                        write_receipt(receipt, result)
                        batch.import_results(args.judge, receipt)
                        done += 1
                index += len(window)
                status = {"attempted_or_recovered": done, "considered": index, "selected": len(pending),
                          "context_blocked": blocked, "unknown_inflight_not_resent": inflight_unknown,
                          "ledger": ledger.report(), "batch": batch.report(), "training_started": False}
                atomic_status(work / "STATUS.json", status)
                print(canonical({k: v for k, v in status.items() if k != "batch"}), flush=True)
                if pilot and tasks:
                    if batch.db.execute("SELECT status FROM jobs WHERE document_id=? AND judge=?", (tasks[0][1]["document_id"], args.judge)).fetchone()[0] != "valid":
                        raise ValueError("pilot failed; inspect receipt before increasing concurrency")
                    pilot = False
                if ledger.report()["cap_exceeded"]:
                    raise ValueError("pricing/reservation assumptions violated; stop further requests")
                # No retry storm on 429/timeout. Errors remain visible in denominator.
                if any(ledger.result(t[0]).get("status") != "ok" for t in tasks):
                    raise ValueError("request failure; fail closed instead of automatic repeated billing")
                if any(batch.db.execute("SELECT status FROM jobs WHERE document_id=? AND judge=?", (t[1]["document_id"], args.judge)).fetchone()[0] != "valid" for t in tasks):
                    raise ValueError("invalid/conflicting result; inspect before continuing")
            valid_selected = sum(batch.db.execute("SELECT status FROM jobs WHERE document_id=? AND judge=?", (j["document_id"], args.judge)).fetchone()[0] == "valid" for j in pending)
            complete = not blocked and not inflight_unknown and valid_selected == len(pending)
            atomic_status(work / "STATUS.json", {"stage": "complete" if complete else "incomplete_fail_closed",
                "selected": len(pending), "valid_selected": valid_selected, "context_blocked": blocked,
                "unknown_inflight_not_resent": inflight_unknown, "ledger": ledger.report(),
                "batch": batch.report(), "training_started": False})
            return 0 if complete else 3
        except BaseException as exc:
            atomic_status(work / "STATUS.json", {"stage": "failed", "failure_type": type(exc).__name__,
                "ledger": ledger.report(), "batch": batch.report(), "training_started": False})
            raise
        finally:
            ledger.close()
            lock.close()
    finally:
        batch.close()


if __name__ == "__main__":
    raise SystemExit(main())
