#!/usr/bin/env python3
"""Verify/status the eight-GPU audit queue without approving training data."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def records(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"bad JSON: {path}:{number}") from exc


def verify_band(root: Path, name: str, meta: dict) -> tuple[dict, set[str]]:
    stats = Counter()
    by_source = defaultdict(Counter)
    all_expected = set()
    for shard_meta in meta["shards"]:
        shard = shard_meta["shard"]
        input_path = root / "inputs" / name / f"shard-{shard}.input.jsonl"
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != shard_meta["sha256"]:
            raise ValueError(f"input hash mismatch: {name}/{shard}")
        input_rows = list(records(input_path))
        expected = {row["document_id"]: row["source_id"] for row in input_rows}
        if len(expected) != len(input_rows) or len(expected) != shard_meta["documents"]:
            raise ValueError(f"input identity/count mismatch: {name}/{shard}")
        if all_expected & set(expected):
            raise ValueError(f"duplicate input ID across shards: {name}/{shard}")
        all_expected.update(expected)
        attempted, successful = set(), set()
        result_path = root / "results" / name / f"shard-{shard}.results.jsonl"
        for row in records(result_path):
            doc_id, = json.loads(row["source_id"])
            if doc_id not in expected:
                raise ValueError(f"unexpected result ID: {name}/{shard}/{doc_id}")
            attempted.add(doc_id)
            if row.get("status") == "ok":
                annotation = row.get("annotation", {})
                if doc_id in successful or annotation.get("document_id") != doc_id:
                    raise ValueError(f"duplicate/mismatched success: {name}/{shard}/{doc_id}")
                successful.add(doc_id)
                by_source[expected[doc_id]][annotation["decision"]] += 1
            else:
                stats["error_attempts"] += 1
        stats["expected"] += len(expected)
        stats["attempted"] += len(attempted)
        stats["ok"] += len(successful)
        stats["unattempted"] += len(set(expected) - attempted)
    if stats["expected"] != meta["documents"]:
        raise ValueError(f"band count mismatch: {name}")
    return {"name": name, **dict(stats),
            "by_source_decision": {src: dict(counts) for src, counts in sorted(by_source.items())}}, all_expected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--band", choices=("heavy", "medium", "light"))
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-missing-ratio", type=float, default=0.01)
    args = ap.parse_args()
    if sum((bool(args.band), args.final, args.status)) != 1:
        raise ValueError("choose exactly one of --band, --final, --status")
    manifest_path = args.root / "inputs" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["num_shards"] != 8 or sum(manifest["targets"].values()) != 150000:
        raise ValueError("unexpected frozen eight-GPU input plan")
    names = [args.band] if args.band else ["heavy", "medium", "light"]
    bands, all_expected = [], set()
    for name in names:
        band, ids = verify_band(args.root, name, manifest["bands"][name])
        if all_expected & ids:
            raise ValueError("duplicate ID across bands")
        all_expected.update(ids)
        bands.append(band)
    if not args.status:
        for band in bands:
            if band["unattempted"] or (band["expected"] - band["ok"]) / max(1, band["expected"]) > args.max_missing_ratio:
                raise ValueError(f"too many missing results in {band['name']}: {band}")
    summary = {
        "stage": "eight_gpu_quality_audit_not_training_approved",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "bands": bands,
        "expected": sum(b["expected"] for b in bands),
        "attempted": sum(b["attempted"] for b in bands),
        "ok": sum(b["ok"] for b in bands),
        "unattempted": sum(b["unattempted"] for b in bands),
        "error_attempts": sum(b.get("error_attempts", 0) for b in bands),
    }
    if args.status:
        print(json.dumps(summary, ensure_ascii=False))
        return
    path = args.root / ("FINISHED8.json" if args.final else f"results/{args.band}/verified.json")
    if args.final and path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
