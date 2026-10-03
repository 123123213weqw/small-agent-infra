#!/usr/bin/env python3
"""Low-priority llama.cpp wash gateway. Bind to loopback; never replace :8000 blindly.

All interactive callers must use this gateway for a strict preemption guarantee.
Direct callers of the upstream are detected as far as llama.cpp metrics permit,
but cannot be held while a large KV checkpoint is restored.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}
ANNOTATION_KEYS = {
    "document_id", "decision", "confidence", "quality", "completeness",
    "educational_value", "format_integrity", "recommended_bucket",
    "reason_codes", "evidence",
}
BUCKETS = {
    "general_zh", "general_en", "technical", "code_python", "code_shell",
    "code_other", "math_zh", "math_en", "drop",
}
REASONS = {
    "seo_or_ad", "template_or_noise", "stitched_content", "incomplete",
    "paywall_or_preview", "low_information", "wrong_bucket", "format_broken",
    "code_incomplete", "generated_or_vendored", "math_incomplete",
    "factual_or_logical_error", "sensitive_data", "uncertain",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jobs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    jobs = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            job = json.loads(line)
            doc_id = job["document_id"]
            if doc_id in seen or not isinstance(job.get("messages"), list):
                raise ValueError(f"duplicate or invalid wash job: {doc_id}")
            seen.add(doc_id)
            jobs.append(job)
    return jobs


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("status") == "ok":
                result.add(item["document_id"])
    return result


def validate_annotation(annotation: dict, document_id: str) -> None:
    if not isinstance(annotation, dict) or set(annotation) != ANNOTATION_KEYS:
        raise ValueError("annotation fields differ from v2 schema")
    if annotation["document_id"] != document_id:
        raise ValueError("annotation document_id mismatch")
    if annotation["decision"] not in ("keep", "drop") or annotation["recommended_bucket"] not in BUCKETS:
        raise ValueError("invalid decision or bucket")
    confidence = annotation["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("invalid confidence")
    for key in ("quality", "completeness", "educational_value", "format_integrity"):
        value = annotation[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"invalid {key}")
    reasons = annotation["reason_codes"]
    if not isinstance(reasons, list) or len(reasons) != len(set(reasons)) or any(x not in REASONS for x in reasons):
        raise ValueError("invalid reason_codes")
    evidence = annotation["evidence"]
    if not isinstance(evidence, str) or not 2 <= len(evidence) <= 240:
        raise ValueError("invalid evidence")


class Gateway:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = httpx.AsyncClient(timeout=None, trust_env=False)
        self.jobs = load_jobs(args.input)
        self.done = completed_ids(args.output)
        self.attempted_this_run: set[str] = set()
        self.active_users = 0
        self.last_user_at = time.monotonic()
        self.wash_task: asyncio.Task | None = None
        self.loop_task: asyncio.Task | None = None
        self.direct_bypass_seen = False
        self.last_reason = "starting"
        self.preemptions = 0
        self.failures = 0
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        self.loop_task = asyncio.create_task(self.run_loop())

    async def stop(self) -> None:
        if self.loop_task:
            self.loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.loop_task
        if self.wash_task:
            self.wash_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.wash_task
        await self.client.aclose()

    async def before_user(self) -> None:
        async with self.lock:
            self.active_users += 1
            self.last_user_at = time.monotonic()
            task = self.wash_task
            if task and not task.done():
                self.preemptions += 1
                task.cancel()
        if task and not task.done():
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(task, timeout=self.args.preempt_timeout)

    async def after_user(self) -> None:
        async with self.lock:
            self.active_users -= 1
            self.last_user_at = time.monotonic()

    async def upstream_state(self) -> tuple[int, int, int]:
        metrics = await self.client.get(self.args.upstream + "/metrics", timeout=3)
        metrics.raise_for_status()
        values = {}
        for line in metrics.text.splitlines():
            if line.startswith("llamacpp:requests_processing ") or line.startswith("llamacpp:requests_deferred "):
                key, value = line.split(" ", 1)
                values[key] = int(float(value))
        slot = await self.client.get(self.args.upstream + "/slots", timeout=3)
        slot.raise_for_status()
        cached = int(slot.json()[0].get("n_prompt_tokens", 0) or 0)
        return values.get("llamacpp:requests_processing", 1), values.get("llamacpp:requests_deferred", 1), cached

    async def run_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.args.poll_seconds)
                if not self.args.enable_wash:
                    self.last_reason = "wash_disabled"
                    continue
                if self.direct_bypass_seen:
                    self.last_reason = "direct_client_bypassed_gateway"
                    continue
                task = self.wash_task
                if task and not task.done():
                    # A direct :8000 caller can be deferred behind our request.
                    processing, deferred, _ = await self.upstream_state()
                    if deferred > 0:
                        self.direct_bypass_seen = True
                        task.cancel()
                        self.last_reason = "direct_client_preempted_wash_disabled"
                    continue
                if self.active_users or time.monotonic() - self.last_user_at < self.args.idle_seconds:
                    self.last_reason = "user_active_or_cooldown"
                    continue
                pending = next((j for j in self.jobs if j["document_id"] not in self.done
                                and j["document_id"] not in self.attempted_this_run), None)
                if pending is None:
                    self.last_reason = "queue_exhausted"
                    continue
                processing, deferred, cached = await self.upstream_state()
                if processing or deferred:
                    self.last_reason = "upstream_busy"
                    self.last_user_at = time.monotonic()
                    continue
                # One llama slot: background work would replace its interactive KV.
                # Fail closed rather than secretly destroying a large user cache.
                if cached > self.args.max_displace_cache_tokens:
                    self.last_reason = f"user_kv_protected:{cached}"
                    continue
                async with self.lock:
                    if self.active_users:
                        continue
                    self.attempted_this_run.add(pending["document_id"])
                    self.wash_task = asyncio.create_task(self.score_one(pending))
                    self.last_reason = "washing"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.failures += 1
                self.last_reason = f"monitor_error:{type(exc).__name__}"
                await asyncio.sleep(3)

    async def score_one(self, job: dict) -> None:
        started = time.monotonic()
        payload = {
            "model": self.args.model,
            "messages": job["messages"],
            "temperature": 0,
            "max_tokens": min(int(job.get("max_tokens", 256)), self.args.max_output_tokens),
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        try:
            response = await self.client.post(
                self.args.upstream + "/v1/chat/completions", json=payload,
                timeout=self.args.wash_timeout,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            annotation = json.loads(content)
            validate_annotation(annotation, job["document_id"])
            result = {
                "document_id": job["document_id"], "status": "ok",
                "annotation": annotation, "usage": body.get("usage", {}),
                "model": body.get("model"), "generated_at": utc_now(),
                "latency_s": round(time.monotonic() - started, 3),
            }
        except asyncio.CancelledError:
            # Never persist a partial or interrupted annotation. Retry next run.
            self.attempted_this_run.discard(job["document_id"])
            raise
        except Exception as exc:
            self.failures += 1
            result = {
                "document_id": job["document_id"], "status": "error",
                "error": type(exc).__name__, "generated_at": utc_now(),
                "latency_s": round(time.monotonic() - started, 3),
            }
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        with self.args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if result["status"] == "ok":
            self.done.add(job["document_id"])

    def status(self) -> dict:
        return {
            "enabled": self.args.enable_wash,
            "reason": self.last_reason,
            "active_users": self.active_users,
            "wash_inflight": bool(self.wash_task and not self.wash_task.done()),
            "queue_total": len(self.jobs), "completed": len(self.done),
            "preemptions": self.preemptions, "failures": self.failures,
            "direct_bypass_seen": self.direct_bypass_seen,
            "max_displace_cache_tokens": self.args.max_displace_cache_tokens,
        }


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    gateway = Gateway(args)
    app.state.gateway = gateway

    @app.on_event("startup")
    async def startup() -> None:
        await gateway.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await gateway.stop()

    @app.get("/idle-wash/status")
    async def status() -> JSONResponse:
        return JSONResponse(gateway.status())

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def proxy(path: str, request: Request):
        await gateway.before_user()
        upstream_response = None
        try:
            headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
            upstream_request = gateway.client.build_request(
                request.method, args.upstream + "/" + path,
                params=request.query_params, content=await request.body(), headers=headers,
            )
            upstream_response = await gateway.client.send(upstream_request, stream=True)
        except Exception as exc:
            await gateway.after_user()
            return JSONResponse({"error": type(exc).__name__}, status_code=502)

        async def body():
            try:
                if upstream_response.is_stream_consumed:
                    yield upstream_response.content
                else:
                    async for chunk in upstream_response.aiter_raw():
                        yield chunk
            finally:
                await upstream_response.aclose()
                await gateway.after_user()

        response_headers = {k: v for k, v in upstream_response.headers.items()
                            if k.lower() not in HOP_HEADERS}
        return StreamingResponse(body(), status_code=upstream_response.status_code,
                                 headers=response_headers)

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--upstream", default="http://127.0.0.1:8000")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="qwen3.8-27b-uncensored")
    p.add_argument("--enable-wash", action="store_true")
    p.add_argument("--idle-seconds", type=float, default=300)
    p.add_argument("--poll-seconds", type=float, default=0.5)
    p.add_argument("--preempt-timeout", type=float, default=3)
    p.add_argument("--wash-timeout", type=float, default=120)
    p.add_argument("--max-output-tokens", type=int, default=256)
    p.add_argument("--max-displace-cache-tokens", type=int, default=8192)
    return p.parse_args()


if __name__ == "__main__":
    import uvicorn
    options = parse_args()
    uvicorn.run(create_app(options), host=options.host, port=options.port,
                access_log=False, server_header=False)
