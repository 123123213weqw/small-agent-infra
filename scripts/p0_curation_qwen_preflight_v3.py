#!/usr/bin/env python3
"""CPU-only teacher context/decoder admission; never call an API or launch GPUs."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from small_agent.curation.store import Batch, write_json

spec = importlib.util.spec_from_file_location("curation_worker_preflight", ROOT / "scripts/p0_curation_worker_v3.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, nargs="+", required=True)
    p.add_argument("--judge", default="qwen")
    p.add_argument("--check-decoder", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("version admission reports, never overwrite")
    rows, manifests, tokenizer = [], {}, None
    for path in a.batch:
        batch = Batch(path)
        try:
            judge = batch.registry["judges"][a.judge]
            if judge["endpoint"]["kind"] != "chat":
                raise ValueError("preflight requires a local chat teacher")
            # Verify files before loading; share the loaded tokenizer only
            # across identical endpoint/teacher/generation configurations.
            if tokenizer is None:
                tokenizer = worker.tokenizer_for(judge)
                identity = judge
            elif identity != judge:
                raise ValueError("mixed teacher identity in one admission report")
            manifests[str(path)] = batch.manifest_hash
            for doc in batch.docs.values():
                body = worker.request_body(doc, batch.prompt, batch.schema, judge)
                n = worker.teacher_input_length(tokenizer, body["messages"], judge["generation"]["chat_template_kwargs"])
                maximum = judge["endpoint"]["maximum_context_tokens"]
                row = {"document_id": doc["document_id"], "teacher_input_tokens": n,
                    "reserved_output_tokens": judge["generation"]["max_tokens"], "framing_margin_tokens": 256,
                    "maximum_context_tokens": maximum,
                    "context_allowed": n + judge["generation"]["max_tokens"] + 256 <= maximum}
                if a.check_decoder:
                    from vllm.sampling_params import SamplingParams, StructuredOutputsParams
                    from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar
                    validate_xgrammar_grammar(SamplingParams(structured_outputs=StructuredOutputsParams(
                        json=body["response_format"]["json_schema"]["schema"])))
                    row["installed_xgrammar_admission"] = "passed"
                rows.append(row)
        finally:
            batch.close()
    out = {"stage": "cpu_admission_not_quality_verdict", "batch_manifests": manifests, "documents": rows,
        "api_calls": 0, "gpu_jobs_started": 0, "training_started": False,
        "context_blocked": sum(not r["context_allowed"] for r in rows)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.output, out)
    print(json.dumps({"documents": len(rows), "maximum_teacher_input_tokens": max(r["teacher_input_tokens"] for r in rows),
        "context_blocked": out["context_blocked"], "api_calls": 0, "training_started": False}))


if __name__ == "__main__":
    main()
