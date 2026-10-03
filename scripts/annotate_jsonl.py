#!/usr/bin/env python3
"""Annotate JSON/JSONL records through an OpenAI-compatible chat API.

The output is an append-only JSONL file. Successful records are skipped on
reruns when the source ID, text, prompt, schema, and model are unchanged.
This tool intentionally knows nothing about Label Studio.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from jsonschema import Draft202012Validator


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def get_field(record: dict[str, Any], dotted_path: str) -> Any:
    value: Any = record
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"missing field {dotted_path!r}")
        value = value[part]
    return value


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    elif path.suffix.lower() == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("input must be .json or .jsonl")
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise ValueError("input must contain a list of JSON objects")
    return records


def make_job(
    record: dict[str, Any],
    index: int,
    text_field: str,
    id_fields: list[str],
    carry_fields: list[str],
    prompt_hash: str,
    schema_hash: str,
    model: str,
    temperature: float | None = None,
    inference_hash: str | None = None,
) -> dict[str, Any]:
    text = get_field(record, text_field)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"record {index}: {text_field!r} must be nonempty text")
    source_id = canonical_json([get_field(record, field) for field in id_fields]) if id_fields else str(index)
    source = {field: get_field(record, field) for field in carry_fields}
    job = {
        "source_id": source_id,
        "source_index": index,
        "source": source,
        "text": text,
        "text_sha256": digest(text),
        "prompt_sha256": prompt_hash,
        "schema_sha256": schema_hash,
        "model": model,
    }
    if temperature is not None:
        job["temperature"] = temperature
    if inference_hash is not None:
        job["inference_sha256"] = inference_hash
    return job


def job_key(job: dict[str, Any]) -> tuple[Any, ...]:
    return (*tuple(job[key] for key in ("source_id", "text_sha256", "prompt_sha256", "schema_sha256", "model")), job.get("temperature"), job.get("inference_sha256"))


def completed_keys(path: Path) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                if result.get("status") == "ok":
                    keys.add(job_key(result))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"bad output line {line_number}: {exc}") from exc
    return keys


def call_model(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    text: str,
    max_tokens: int,
    timeout: float,
    thinking: str,
    temperature: float | None,
    response_format: dict[str, Any] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        "response_format": response_format or {"type": "json_object"},
        "max_tokens": max_tokens,
        "stream": False,
    }
    if thinking != "omit":
        payload["thinking"] = {"type": thinking}
    if temperature is not None:
        payload["temperature"] = temperature
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    request = Request(
        api_base.rstrip("/") + "/chat/completions",
        data=canonical_json(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "User-Agent": "small-agent-data-audit/1",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except HTTPError as exc:
        # Never put request/response bodies in logs: they may contain source text.
        raise RuntimeError(f"HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"network {type(exc.reason).__name__}") from exc
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("model returned no choices")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise ValueError("model output was truncated")
    content = (choice.get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned empty content")
    annotation = json.loads(content)
    if not isinstance(annotation, dict):
        raise ValueError("model output must be a JSON object")
    return annotation, body.get("usage") or {}, str(body.get("model") or model)


def score_job(
    job: dict[str, Any], *, api_base: str, api_key: str, prompt: str,
    validator: Draft202012Validator, max_tokens: int, timeout: float,
    max_retries: int, thinking: str, temperature: float | None,
    response_format: dict[str, Any] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    require_document_id_match: bool = False,
) -> dict[str, Any]:
    error = "unknown error"
    started = time.monotonic()
    for attempt in range(max_retries + 1):
        try:
            annotation, usage, served_model = call_model(
                api_base=api_base,
                api_key=api_key,
                model=job["model"],
                prompt=prompt,
                text=job["text"],
                max_tokens=max_tokens,
                timeout=timeout,
                thinking=thinking,
                temperature=temperature,
                response_format=response_format,
                chat_template_kwargs=chat_template_kwargs,
            )
            problems = list(validator.iter_errors(annotation))
            if problems:
                raise ValueError("schema: " + problems[0].message[:160])
            if require_document_id_match and not document_id_matches_source(annotation, job["source_id"]):
                raise ValueError("annotation document_id does not match source_id")
            return {
                **{k: v for k, v in job.items() if k != "text"},
                "status": "ok",
                "annotation": annotation,
                "served_model": served_model,
                "usage": usage,
                "attempts": attempt + 1,
                "latency_s": round(time.monotonic() - started, 3),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, RuntimeError, TimeoutError) as exc:
            error = str(exc)[:200]
            if error in ("HTTP 401", "HTTP 402", "HTTP 403"):
                break
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 8))
    return {
        **{k: v for k, v in job.items() if k != "text"},
        "status": "error",
        "error": error,
        "attempts": attempt + 1,
        "latency_s": round(time.monotonic() - started, 3),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def document_id_matches_source(annotation: dict[str, Any], source_id: str) -> bool:
    try:
        ids = json.loads(source_id)
    except (TypeError, ValueError):
        return False
    return isinstance(ids, list) and len(ids) == 1 and annotation.get("document_id") == ids[0]


def make_response_format(schema: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "json_object":
        return {"type": "json_object"}
    # vLLM's grammar backend does not implement uniqueItems. This copy is only
    # a generation constraint; the original full schema is validated afterward.
    constrained = json.loads(json.dumps(schema))
    constrained.pop("$schema", None)
    constrained.pop("$id", None)

    def remove_unsupported(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("uniqueItems", None)
            for child in value.values():
                remove_unsupported(child)
        elif isinstance(value, list):
            for child in value:
                remove_unsupported(child)

    remove_unsupported(constrained)
    return {"type": "json_schema", "json_schema": {"name": "quality_annotation", "schema": constrained, "strict": True}}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help=".json array or .jsonl records")
    p.add_argument("--output", type=Path, required=True, help="append-only results .jsonl")
    p.add_argument("--prompt-file", type=Path, required=True, help="system prompt text")
    p.add_argument("--schema-file", type=Path, required=True, help="JSON Schema for annotation object")
    p.add_argument("--text-field", required=True, help="dot path, e.g. excerpt or data.text")
    p.add_argument("--id-fields", default="", help="comma-separated dot paths; defaults to row number")
    p.add_argument("--carry-fields", default="", help="comma-separated source fields to include in output")
    p.add_argument("--api-base", default="https://api.deepseek.com")
    p.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    p.add_argument("--model", default="deepseek-flash")
    p.add_argument("--thinking", choices=("enabled", "disabled", "omit"), default="disabled")
    p.add_argument("--response-format", choices=("json_object", "json_schema"), default="json_object")
    p.add_argument("--chat-template-kwargs", default="{}", help="JSON object passed to compatible local model servers")
    p.add_argument("--require-document-id-match", action="store_true", help="require annotation.document_id to match the sole --id-fields value")
    p.add_argument("--temperature", type=float, default=None, help="0-2; omit to use API default")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="maximum pending records to run; 0 means all")
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--timeout", type=float, default=90)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must be different files")
    if args.workers < 1 or args.max_retries < 0 or args.limit < 0:
        raise ValueError("workers must be >=1; max-retries and limit must be >=0")
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        raise ValueError("temperature must be in [0, 2]")
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    if not prompt or "json" not in prompt.lower():
        raise ValueError("prompt must be nonempty and mention JSON for JSON output mode")
    schema = json.loads(args.schema_file.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    chat_template_kwargs = json.loads(args.chat_template_kwargs)
    if not isinstance(chat_template_kwargs, dict):
        raise ValueError("--chat-template-kwargs must be a JSON object")
    response_format = make_response_format(schema, args.response_format)
    inference_hash = None
    if chat_template_kwargs or args.response_format != "json_object" or args.require_document_id_match:
        inference_hash = digest(canonical_json({
            "chat_template_kwargs": chat_template_kwargs,
            "response_format": args.response_format,
            "thinking": args.thinking,
            "max_tokens": args.max_tokens,
            "require_document_id_match": args.require_document_id_match,
        }))
    prompt_hash, schema_hash = digest(prompt), digest(canonical_json(schema))
    id_fields = [x.strip() for x in args.id_fields.split(",") if x.strip()]
    if args.require_document_id_match and len(id_fields) != 1:
        raise ValueError("--require-document-id-match needs exactly one --id-fields value")
    carry_fields = [x.strip() for x in args.carry_fields.split(",") if x.strip()]
    records = read_records(args.input)
    jobs = [
        make_job(r, i, args.text_field, id_fields, carry_fields, prompt_hash, schema_hash, args.model, args.temperature, inference_hash)
        for i, r in enumerate(records)
    ]
    source_ids = [job["source_id"] for job in jobs]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("duplicate source IDs; use more --id-fields")
    done = completed_keys(args.output)
    pending = [job for job in jobs if job_key(job) not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"input={len(jobs)} completed={len(jobs)-len([j for j in jobs if job_key(j) not in done])} pending={len(pending)}", file=sys.stderr)
    if args.dry_run or not pending:
        return 0
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"missing API key environment variable {args.api_key_env}")
    os.umask(0o077)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ok = errors = 0
    with args.output.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        os.chmod(args.output, 0o600)
        futures = {
            pool.submit(
                score_job, job, api_base=args.api_base, api_key=api_key,
                prompt=prompt, validator=validator, max_tokens=args.max_tokens,
                timeout=args.timeout, max_retries=args.max_retries, thinking=args.thinking,
                temperature=args.temperature,
                response_format=response_format, chat_template_kwargs=chat_template_kwargs,
                require_document_id_match=args.require_document_id_match,
            ): job["source_id"] for job in pending
        }
        for future in as_completed(futures):
            result = future.result()
            handle.write(canonical_json(result) + "\n")
            handle.flush()
            if result["status"] == "ok":
                ok += 1
            else:
                errors += 1
                print(f"error source_id={result['source_id']} reason={result['error']}", file=sys.stderr)
            if (ok + errors) % 10 == 0 or ok + errors == len(pending):
                os.fsync(handle.fileno())
                print(f"progress={ok+errors}/{len(pending)} ok={ok} errors={errors}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
