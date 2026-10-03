#!/usr/bin/env python3
"""Verify all six-shard new-source probe outputs without approving training data."""

import argparse
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


def verify_band(root: Path, name: str, expected_count: int) -> dict:
    stats = {"name": name, "expected": expected_count, "attempted": 0,
             "ok": 0, "errors": 0, "unattempted": 0}
    for shard in range(6):
        input_path = root / "inputs" / name / f"shard-{shard}.input.jsonl"
        expected = {row["document_id"] for row in records(input_path)}
        if len(expected) != sum(1 for _ in records(input_path)):
            raise ValueError(f"duplicate input ID in {name}/{shard}")
        attempted, successful = set(), set()
        result_path = root / "results" / name / f"shard-{shard}.results.jsonl"
        for row in records(result_path):
            doc_id, = json.loads(row["source_id"])
            if doc_id not in expected:
                raise ValueError(f"unexpected document ID in {name}/{shard}")
            attempted.add(doc_id)
            if row.get("status") == "ok":
                if doc_id in successful or row.get("annotation", {}).get("document_id") != doc_id:
                    raise ValueError(f"duplicate or mismatched result in {name}/{shard}")
                successful.add(doc_id)
            else:
                stats["errors"] += 1
        stats["attempted"] += len(attempted)
        stats["ok"] += len(successful)
        stats["unattempted"] += len(expected - attempted)
    if stats["attempted"] + stats["unattempted"] != expected_count:
        raise ValueError(f"count mismatch in {name}")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--band", choices=("heavy", "medium", "light"))
    p.add_argument("--final", action="store_true")
    p.add_argument("--max-missing-ratio", type=float, default=0.01)
    args = p.parse_args()
    if args.final == bool(args.band):
        raise ValueError("choose exactly one of --band and --final")
    manifest_path = args.root / "inputs" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = list(manifest["bands"]) if args.final else [args.band]
    bands = [verify_band(args.root, name, manifest["bands"][name]["documents"]) for name in names]
    for band in bands:
        if band["unattempted"] or (band["expected"] - band["ok"]) / max(1, band["expected"]) > args.max_missing_ratio:
            raise ValueError(f"too many missing results in {band['name']}: {band}")
    summary = {
        "stage": "new_raw_source_probe_scores_not_training_approved",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "bands": bands,
        "expected": sum(b["expected"] for b in bands),
        "attempted": sum(b["attempted"] for b in bands),
        "ok": sum(b["ok"] for b in bands),
        "errors": sum(b["errors"] for b in bands),
    }
    path = args.root / ("FINISHED6.json" if args.final else f"results/{args.band}/verified.json")
    if path.exists() and args.final:
        raise ValueError("final verification marker already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
