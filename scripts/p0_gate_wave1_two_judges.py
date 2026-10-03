#!/usr/bin/env python3
"""Fail-closed Wave1 Qwen/DeepSeek agreement gate (no text packing here)."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def strict_keep(a: dict, domain: str) -> bool:
    return (
        a.get("decision") == "keep"
        and a.get("confidence", 0) >= 0.85
        and a.get("quality", 0) >= 3
        and a.get("completeness", 0) >= 4
        and a.get("educational_value", 0) >= 3
        and a.get("format_integrity", 0) >= 4
        and a.get("recommended_bucket") == domain
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    expected = {}
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            doc_id = x["document_id"]
            if doc_id in expected:
                raise ValueError("duplicate input document ID")
            expected[doc_id] = {
                "domain": x["current_bucket"],
                "payload_sha256": hashlib.sha256(x["review_payload"].encode()).hexdigest(),
            }
    seen = {}
    conflicts = set()
    errors = Counter()
    protocol = None
    with args.results.open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            try:
                source_id = json.loads(x["source_id"])[0]
            except (KeyError, ValueError, TypeError, IndexError) as exc:
                raise ValueError("bad result source ID") from exc
            if source_id not in expected:
                raise ValueError("unexpected result document ID")
            if x["text_sha256"] != expected[source_id]["payload_sha256"]:
                raise ValueError("payload hash mismatch")
            key = (x["prompt_sha256"], x["schema_sha256"], x["model"],
                   x.get("temperature"), x["inference_sha256"])
            if protocol is None:
                protocol = key
            elif key != protocol:
                raise ValueError("mixed prompt/schema/model/inference settings")
            if x.get("status") != "ok":
                errors[source_id] += 1
                continue
            a = x["annotation"]
            if a.get("document_id") != source_id:
                raise ValueError("annotation/document ID mismatch")
            if source_id in seen and seen[source_id] != a:
                conflicts.add(source_id)
            else:
                seen[source_id] = a
    counts = Counter()
    with (args.output_dir / "two_judge_accepted_ids.jsonl").open("w", encoding="utf-8") as out:
        for doc_id, item in expected.items():
            domain = item["domain"]
            if doc_id in conflicts:
                reason = "conflicting_valid_results"
            elif doc_id not in seen:
                reason = "missing_valid_deepseek_result"
            elif not strict_keep(seen[doc_id], domain):
                reason = "deepseek_not_strict_same_domain_keep"
            else:
                reason = "two_judge_agreed_keep"
                out.write(json.dumps({"document_id": doc_id, "domain": domain}, sort_keys=True) + "\n")
            counts[(domain, reason)] += 1
    accepted = args.output_dir / "two_judge_accepted_ids.jsonl"
    summary = {
        "stage": "wave1_two_judge_quality_gate",
        "training_eligible": False,
        "reason": "text freeze, cross-corpus dedup and split audit still required",
        "input_sha256": sha256(args.input),
        "results_sha256": sha256(args.results),
        "accepted_ids_sha256": sha256(accepted),
        "input_documents": len(expected),
        "valid_deepseek_documents": len(seen),
        "conflicting_valid_documents": len(conflicts),
        "documents_with_error_attempts": len(errors),
        "accepted_documents": sum(v for (_, reason), v in counts.items() if reason == "two_judge_agreed_keep"),
        "decisions": [
            {"domain": domain, "reason": reason, "documents": n}
            for (domain, reason), n in sorted(counts.items())
        ],
        "protocol": {
            "prompt_sha256": protocol[0], "schema_sha256": protocol[1],
            "model": protocol[2], "temperature": protocol[3],
            "inference_sha256": protocol[4],
        } if protocol else None,
    }
    (args.output_dir / "gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
