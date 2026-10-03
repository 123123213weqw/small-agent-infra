#!/usr/bin/env python3
"""Verify eight append-only scorer outputs without approving any training data."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--write-complete", action="store_true")
    ap.add_argument("--write-finished", action="store_true", help="mark attempted-but-invalid documents quarantined")
    ap.add_argument("--max-missing-ratio", type=float, default=0.01)
    args = ap.parse_args()
    if args.write_complete and args.write_finished:
        raise ValueError("choose only one final marker")
    if not 0 <= args.max_missing_ratio <= 1:
        raise ValueError("max-missing-ratio must be in [0,1]")
    manifest = json.loads((args.data_dir / "input_shards" / "manifest.json").read_text())
    expected_total = ok_total = 0
    attempts_total = 0
    domain_ok: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    shard_stats = []
    for i, meta in enumerate(manifest["shards"]):
        input_path = args.data_dir / "input_shards" / meta["file"]
        if sha256_file(input_path) != meta["sha256"]:
            raise ValueError(f"input shard {i} changed")
        expected = {r["document_id"] for r in rows(input_path)}
        if len(expected) != meta["documents"]:
            raise ValueError(f"input shard {i} has duplicate IDs")
        ok: set[str] = set()
        attempted: set[str] = set()
        errors = 0
        attempts = 0
        for result in rows(args.data_dir / "results" / f"shard-{i}.results.jsonl"):
            doc_id = json.loads(result["source_id"])[0]
            if doc_id not in expected:
                raise ValueError(f"unexpected result document ID: {doc_id}")
            attempted.add(doc_id)
            attempts += 1
            if result["status"] == "ok":
                if doc_id in ok:
                    raise ValueError(f"duplicate successful result: {doc_id}")
                if result["annotation"]["document_id"] != doc_id:
                    raise ValueError(f"annotation ID mismatch: {doc_id}")
                ok.add(doc_id)
                domain_ok[result["source"]["current_bucket"]] += 1
            else:
                errors += 1
                error_counts[str(result.get("error", "unknown"))] += 1
        expected_total += len(expected)
        ok_total += len(ok)
        attempts_total += attempts
        shard_stats.append({"shard": i, "expected": len(expected), "ok": len(ok),
                            "missing": len(expected - ok), "unattempted": len(expected - attempted),
                            "error_attempts": errors,
                            "total_attempts": attempts})
    if expected_total != manifest["scoreable_documents"]:
        raise ValueError("manifest scoreable count mismatch")
    report = {
        "stage": "qwen_official_fp8_ratings_not_training_approved",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": sha256_file(args.data_dir / "input_shards" / "manifest.json"),
        "expected": expected_total, "ok": ok_total,
        "missing": expected_total - ok_total,
        "unattempted": sum(shard["unattempted"] for shard in shard_stats),
        "total_attempts": attempts_total,
        "error_attempts": sum(error_counts.values()),
        "errors": dict(sorted(error_counts.items())),
        "ok_by_domain": dict(sorted(domain_ok.items())),
        "shards": shard_stats,
    }
    if args.write_complete:
        if report["missing"]:
            raise ValueError(f"cannot mark complete: {report['missing']} documents missing")
        complete_path = args.data_dir / "COMPLETE.json"
        if complete_path.exists():
            raise ValueError("COMPLETE.json already exists")
        complete_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.write_finished:
        if report["unattempted"]:
            raise ValueError(f"cannot mark finished: {report['unattempted']} documents unattempted")
        if report["missing"] / max(1, expected_total) > args.max_missing_ratio:
            raise ValueError(f"cannot mark finished: missing ratio exceeds {args.max_missing_ratio}")
        finished_path = args.data_dir / "FINISHED.json"
        if finished_path.exists():
            raise ValueError("FINISHED.json already exists")
        finished_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
