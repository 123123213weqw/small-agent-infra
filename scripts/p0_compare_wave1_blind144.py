#!/usr/bin/env python3
"""Compare a frozen Wave1 provisional sample against blind DeepSeek ratings."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    p.add_argument("--review-dir", type=Path, required=True)
    args = p.parse_args()
    root = args.review_dir
    manifest_path = root / "deepseek_blind144.manifest.json"
    input_path = root / "deepseek_blind144.input.jsonl"
    qwen_path = root / "qwen_blind144.labels.jsonl"
    initial_path = root / "deepseek_blind144.initial_only.jsonl"
    retry_path = root / "deepseek_retry2.results.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(input_path) != manifest["input_sha256"]:
        raise ValueError("blind sample input changed")
    inputs = {row["document_id"]: row for row in read_jsonl(input_path)}
    sample = {row["document_id"]: row for row in manifest["items"]}
    qwen = {row["document_id"]: row for row in read_jsonl(qwen_path)}
    if not (set(inputs) == set(sample) == set(qwen)) or len(inputs) != 144:
        raise ValueError("sample/Qwen identity mismatch")
    deepseek = {}
    errors = []
    for path in (initial_path, retry_path):
        for row in read_jsonl(path):
            doc_id, = json.loads(row["source_id"])
            if doc_id not in inputs:
                raise ValueError("unknown DeepSeek document")
            if row["status"] != "ok":
                errors.append(doc_id)
                continue
            if doc_id in deepseek:
                raise ValueError("duplicate DeepSeek valid judgment in selected attempts")
            if row["annotation"]["document_id"] != doc_id:
                raise ValueError("DeepSeek annotation identity mismatch")
            expected_text_hash = hashlib.sha256(inputs[doc_id]["review_payload"].encode()).hexdigest()
            if row["text_sha256"] != expected_text_hash or qwen[doc_id]["text_sha256"] != expected_text_hash:
                raise ValueError("blind review text mismatch")
            if (row["prompt_sha256"], row["schema_sha256"]) != (
                qwen[doc_id]["prompt_sha256"], qwen[doc_id]["schema_sha256"]
            ):
                raise ValueError("Qwen/DeepSeek prompt or schema mismatch")
            deepseek[doc_id] = row
    counts = Counter()
    reasons = Counter()
    records = []
    for doc_id in sorted(inputs):
        domain = sample[doc_id]["domain"]
        bin_index = sample[doc_id]["length_bin"]
        if not strict_keep(qwen[doc_id]["annotation"], domain):
            raise ValueError("Qwen sample not in provisional strict keep pool")
        d = deepseek.get(doc_id)
        verdict = "deepseek_missing_or_invalid" if d is None else (
            "both_strict_same_domain_keep" if strict_keep(d["annotation"], domain)
            else "deepseek_disagrees"
        )
        counts[(domain, bin_index, verdict)] += 1
        if d and verdict == "deepseek_disagrees":
            for reason in d["annotation"].get("reason_codes", []):
                reasons[(domain, reason)] += 1
        records.append({
            "document_id": doc_id,
            "domain": domain,
            "length_bin": bin_index,
            "own_text_tokens": sample[doc_id]["own_text_tokens"],
            "verdict": verdict,
            "deepseek_decision": d["annotation"]["decision"] if d else None,
            "deepseek_recommended_bucket": d["annotation"]["recommended_bucket"] if d else None,
            "deepseek_quality": d["annotation"]["quality"] if d else None,
        })
    report = {
        "stage": "wave1_blind144_calibration_not_document_level_training_approval",
        "sample_documents": len(inputs),
        "deepseek_valid": len(deepseek),
        "deepseek_error_attempts": len(errors),
        "by_domain_bin_verdict": [
            {"domain": dom, "length_bin": bin_idx, "verdict": verdict, "count": count}
            for (dom, bin_idx, verdict), count in sorted(counts.items())
        ],
        "disagreement_reason_codes": [
            {"domain": dom, "reason": reason, "count": count}
            for (dom, reason), count in sorted(reasons.items())
        ],
        "inputs_sha256": {p.name: sha256(p) for p in (manifest_path, input_path, qwen_path, initial_path, retry_path)},
        "records": records,
    }
    output = root / "blind144_comparison.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("records", "inputs_sha256", "disagreement_reason_codes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
