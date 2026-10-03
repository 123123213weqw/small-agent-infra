import argparse
import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest

import httpx

from scripts.v100_idle_wash_gateway import Gateway, create_app


def options(tmp: Path, *, enabled=False):
    return argparse.Namespace(
        host="127.0.0.1", port=8001, upstream="http://llama.test",
        input=tmp / "queue.jsonl", output=tmp / "results.jsonl",
        model="test", enable_wash=enabled, idle_seconds=0.01,
        poll_seconds=0.01, preempt_timeout=0.5, wash_timeout=5,
        max_output_tokens=32, max_displace_cache_tokens=8192,
    )


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_forwards_request_and_releases_user_count(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(options(Path(directory)))
            gateway = app.state.gateway

            async def upstream(request):
                self.assertEqual(request.url.path, "/v1/chat/completions")
                self.assertEqual(request.url.params["test"], "1")
                self.assertEqual(json.loads(request.content), {"hello": "user"})
                return httpx.Response(200, json={"answer": 42})

            await gateway.client.aclose()
            gateway.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
                response = await client.post("/v1/chat/completions?test=1", json={"hello": "user"})
            self.assertEqual(response.json(), {"answer": 42})
            self.assertEqual(gateway.active_users, 0)
            await gateway.client.aclose()

    async def test_user_cancels_background_job(self):
        with tempfile.TemporaryDirectory() as directory:
            gateway = Gateway(options(Path(directory)))
            cancelled = asyncio.Event()

            async def wash():
                try:
                    await asyncio.sleep(60)
                finally:
                    cancelled.set()

            gateway.wash_task = asyncio.create_task(wash())
            await asyncio.sleep(0)
            start = time.monotonic()
            await gateway.before_user()
            self.assertTrue(cancelled.is_set())
            self.assertLess(time.monotonic() - start, 0.5)
            self.assertEqual(gateway.preemptions, 1)
            await gateway.after_user()
            await gateway.client.aclose()

    async def test_large_user_kv_blocks_background(self):
        with tempfile.TemporaryDirectory() as directory:
            args = options(Path(directory), enabled=True)
            args.input.write_text(json.dumps({"document_id": "a", "messages": []}) + "\n")
            gateway = Gateway(args)
            gateway.last_user_at = time.monotonic() - 10

            async def state():
                return 0, 0, 167000

            gateway.upstream_state = state
            await gateway.start()
            await asyncio.sleep(0.06)
            self.assertEqual(gateway.last_reason, "user_kv_protected:167000")
            self.assertIsNone(gateway.wash_task)
            await gateway.stop()

    async def test_resumes_from_successful_result_only(self):
        with tempfile.TemporaryDirectory() as directory:
            args = options(Path(directory))
            args.input.write_text("\n".join([
                json.dumps({"document_id": "a", "messages": []}),
                json.dumps({"document_id": "b", "messages": []}),
            ]) + "\n")
            args.output.write_text(json.dumps({"document_id": "a", "status": "ok"}) + "\n")
            gateway = Gateway(args)
            self.assertEqual(gateway.done, {"a"})
            self.assertEqual(len(gateway.jobs), 2)
            await gateway.client.aclose()

    async def test_score_one_persists_only_matching_result(self):
        with tempfile.TemporaryDirectory() as directory:
            gateway = Gateway(options(Path(directory)))

            async def upstream(request):
                body = json.loads(request.content)
                self.assertEqual(body["temperature"], 0)
                self.assertEqual(body["max_tokens"], 32)
                return httpx.Response(200, json={
                    "model": "test", "usage": {"prompt_tokens": 10},
                    "choices": [{"message": {"content": json.dumps({
                        "document_id": "a", "decision": "drop", "confidence": 0.9,
                        "quality": 2, "completeness": 1, "educational_value": 2,
                        "format_integrity": 2, "recommended_bucket": "drop",
                        "reason_codes": ["incomplete"], "evidence": "文档不完整",
                    })}}],
                })

            await gateway.client.aclose()
            gateway.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
            await gateway.score_one({"document_id": "a", "messages": [], "max_tokens": 128})
            self.assertEqual(gateway.done, {"a"})
            result = json.loads(gateway.args.output.read_text())
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["annotation"]["decision"], "drop")
            await gateway.client.aclose()


if __name__ == "__main__":
    unittest.main()
