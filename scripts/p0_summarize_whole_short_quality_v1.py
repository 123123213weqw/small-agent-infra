#!/usr/bin/env python3
"""Summarize the frozen, full-document (<6000 chars) Qwen quality comparison.

The input is a stratified audit sample, not a document-level approval list.
Partial or errored inference runs cannot produce a quality decision.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"malformed JSON at {path}:{lineno}") from e


def estimate(grouped: dict[str, list[dict]]) -> dict:
    total = sum(v[0]["population"] for v in grouped.values())
    p = 0.0
    variance = 0.0
    per_stratum = {}
    for name, sample in sorted(grouped.items()):
        population = sample[0]["population"]
        n = len(sample)
        kept = sum(x["keep"] for x in sample)
        q = kept / n
        weight = population / total
        p += weight * q
        if n > 1:
            variance += weight * weight * q * (1 - q) / (n - 1) * max(0, 1 - n / population)
        reasons = Counter(reason for x in sample for reason in x["reasons"])
        per_stratum[name] = {
            "population": population, "sample": n, "kept": kept,
            "keep_rate": q, "drop_reasons": dict(reasons),
        }
    se = math.sqrt(variance)
    return {
        "population": total, "sample": sum(map(len, grouped.values())),
        "weighted_keep_rate": p,
        "approx_95pct_interval": [max(0, p - 1.96 * se), min(1, p + 1.96 * se)],
        "note": "Stratified finite-population normal approximation; model labels are not ground truth.",
        "strata": per_stratum,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--input-manifest", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.input_manifest.read_text())
    if manifest["stage"] != "finewiki_short_whole_document_quality_input_audit_only":
        raise ValueError("unexpected input stage")
    if sha(args.input) != manifest["input_sha256"]:
        raise ValueError("audit input hash mismatch")
    input_rows = list(rows(args.input))
    by_id = {x["document_id"]: x for x in input_rows}
    if len(by_id) != len(input_rows) or len(input_rows) != sum(manifest["counts"].values()):
        raise ValueError("input ID/count mismatch")
    result_rows = list(rows(args.results))
    seen: dict[str, dict] = {}
    failures = []
    for r in result_rows:
        try:
            did = json.loads(r["source_id"])[0]
        except (KeyError, ValueError, IndexError, TypeError) as e:
            raise ValueError("invalid inference source_id") from e
        if did not in by_id:
            raise ValueError(f"unexpected inference ID: {did}")
        if r["status"] == "ok":
            if did in seen:
                raise ValueError(f"duplicate successful inference: {did}")
            if r["annotation"]["document_id"] != did:
                raise ValueError("annotation ID mismatch")
            if r["source"] != {k: by_id[did][k] for k in (
                "source", "stratum", "chars", "population_in_stratum",
                "sample_size_in_stratum", "prompt_tokens_est",
            )}:
                raise ValueError("carried source metadata mismatch")
            seen[did] = r
        else:
            failures.append(did)
    missing = set(by_id) - set(seen)
    if missing or failures:
        raise ValueError(f"incomplete audit: {len(seen)}/{len(by_id)} ok, {len(failures)} errors, {len(missing)} missing")
    grouped = defaultdict(lambda: defaultdict(list))
    all_reasons = Counter()
    anomalies = []
    for did, source in by_id.items():
        result = seen[did]
        a = result["annotation"]
        keep = a["decision"] == "keep"
        if a["decision"] not in {"keep", "drop"}:
            raise ValueError("invalid decision")
        if keep and (a["reason_codes"] or a["recommended_bucket"] == "drop"):
            anomalies.append(did)
        reasons = a["reason_codes"]
        all_reasons.update(reasons)
        grouped[source["source"]][source["stratum"]].append({
            "keep": int(keep), "population": source["population_in_stratum"],
            "reasons": reasons,
        })
    if anomalies:
        raise ValueError(f"inconsistent keep labels: {len(anomalies)}")
    for source, strata in grouped.items():
        for name, sample in strata.items():
            if len(sample) != by_id[next(k for k,v in by_id.items() if v["source"] == source and v["stratum"] == name)]["sample_size_in_stratum"]:
                raise ValueError(f"stratum count mismatch: {source}/{name}")
            if len({x["population"] for x in sample}) != 1:
                raise ValueError("stratum population changed")
    report = {
        "stage": "finewiki_short_whole_document_quality_report_audit_only",
        "input_manifest_sha256": sha(args.input_manifest),
        "results_sha256": sha(args.results),
        "result_rows": len(result_rows), "successful_documents": len(seen),
        "sources": {name: estimate(strata) for name, strata in sorted(grouped.items())},
        "all_reason_codes": dict(all_reasons),
        "training_eligible": False,
        "limitations": [
            "Only <6000-character documents are eligible for this source-level decision.",
            "Wave1's <2000-character stratum includes <500-character documents; overall rates are not strictly length matched.",
            "Single Qwen judge labels are not ground truth and do not approve each unsampled document.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
