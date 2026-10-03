#!/usr/bin/env python3
"""Fail-closed verification of eight-card, length-banded Qwen rating outputs."""

from __future__ import annotations

import argparse
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


def jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc


def verify_band(root: Path, band: dict) -> dict:
    name = band["name"]
    shards = []
    for index, meta in enumerate(band["shards"]):
        source = root / "input_bands8_v1" / meta["file"]
        if sha256_file(source) != meta["sha256"]:
            raise ValueError(f"{name}/{index}: input checksum mismatch")
        expected = {row["document_id"] for row in jsonl(source)}
        if len(expected) != meta["documents"]:
            raise ValueError(f"{name}/{index}: duplicate input ID")
        attempted: set[str] = set()
        good: set[str] = set()
        errors = 0
        for result in jsonl(root / "results8" / name / f"shard-{index}.results.jsonl"):
            try:
                doc_id, = json.loads(result["source_id"])
            except (KeyError, ValueError, TypeError):
                raise ValueError(f"{name}/{index}: malformed result source_id") from None
            if doc_id not in expected:
                raise ValueError(f"{name}/{index}: unknown result ID {doc_id}")
            attempted.add(doc_id)
            if result.get("status") == "ok":
                if doc_id in good:
                    raise ValueError(f"{name}/{index}: duplicate successful result ID {doc_id}")
                if result.get("annotation", {}).get("document_id") != doc_id:
                    raise ValueError(f"{name}/{index}: annotation ID mismatch")
                good.add(doc_id)
            else:
                errors += 1
        shards.append({"shard": index, "gpu": meta["gpu"], "expected": len(expected),
                       "attempted": len(attempted), "ok": len(good),
                       "unattempted": len(expected - attempted), "missing": len(expected - good),
                       "error_attempts": errors})
    report = {"band": name, "shards": shards}
    for field in ("expected", "attempted", "ok", "unattempted", "missing", "error_attempts"):
        report[field] = sum(shard[field] for shard in shards)
    if report["expected"] != band["documents"]:
        raise ValueError(f"{name}: band document count mismatch")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--band", choices=("heavy", "medium", "light"))
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--max-missing-ratio", type=float, default=0.01)
    args = ap.parse_args()
    if bool(args.band) == bool(args.final):
        raise ValueError("choose exactly one of --band and --final")
    if not 0 <= args.max_missing_ratio <= 1:
        raise ValueError("max-missing-ratio must be in [0,1]")
    manifest = json.loads((args.data_dir / "input_bands8_v1" / "manifest.json").read_text())
    selected = [band for band in manifest["bands"] if args.final or band["name"] == args.band]
    reports = [verify_band(args.data_dir, band) for band in selected]
    for report in reports:
        if report["unattempted"]:
            raise ValueError(f"{report['band']}: {report['unattempted']} documents unattempted")
        if report["missing"] / max(1, report["expected"]) > args.max_missing_ratio:
            raise ValueError(f"{report['band']}: missing ratio exceeds {args.max_missing_ratio}")
    summary = {"stage": "qwen_official_fp8_eight_gpu_ratings_not_training_approved",
               "generated_at": datetime.now(timezone.utc).isoformat(),
               "input_manifest_sha256": sha256_file(args.data_dir / "input_bands8_v1" / "manifest.json"),
               "bands": reports}
    for field in ("expected", "attempted", "ok", "unattempted", "missing", "error_attempts"):
        summary[field] = sum(report[field] for report in reports)
    if args.final:
        path = args.data_dir / "FINISHED8.json"
        if path.exists():
            raise ValueError("FINISHED8.json already exists")
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        path = args.data_dir / "results8" / args.band / "verified.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
