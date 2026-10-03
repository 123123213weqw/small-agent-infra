#!/usr/bin/env python3
"""Resume and verify pinned FineWiki zh files directly on the target server."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(entry: dict, *, raw_dir: Path, dataset: str, revision: str, endpoint: str) -> dict:
    name = Path(entry["path"]).name
    target = raw_dir / name
    partial = raw_dir / f"{name}.part"
    if target.exists():
        if target.stat().st_size != entry["bytes"] or sha256_file(target) != entry["sha256"]:
            raise ValueError(f"existing file failed verification: {target}")
        return {"file": name, "bytes": entry["bytes"], "sha256": entry["sha256"], "status": "verified_existing"}
    url = f"{endpoint.rstrip('/')}/datasets/{dataset}/resolve/{revision}/{entry['path']}"
    for attempt in range(1, 9):
        if partial.exists() and partial.stat().st_size > entry["bytes"]:
            raise ValueError(f"oversized partial: {partial}")
        if not partial.exists() or partial.stat().st_size < entry["bytes"]:
            # -C - resumes incomplete bytes. --retry-all-errors handles transient CDN stalls.
            result = subprocess.run([
                "curl", "--fail", "--location", "--silent", "--show-error",
                "--connect-timeout", "25", "--max-time", "1200",
                "--retry", "5", "--retry-delay", "3", "--retry-all-errors",
                "--continue-at", "-", "--output", str(partial), url,
            ])
            if result.returncode:
                print(f"download retry {name} attempt={attempt} exit={result.returncode}", flush=True)
                time.sleep(min(30, 3 * attempt))
                continue
        if partial.stat().st_size != entry["bytes"]:
            print(f"size retry {name}: {partial.stat().st_size}/{entry['bytes']}", flush=True)
            continue
        actual = sha256_file(partial)
        if actual != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch {name}: {actual} != {entry['sha256']}")
        partial.replace(target)
        return {"file": name, "bytes": entry["bytes"], "sha256": actual, "status": "downloaded_verified"}
    raise RuntimeError(f"failed after retries: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 5:
        raise ValueError("workers must be 1..5")
    spec = json.loads(args.spec.read_text())
    if spec["status"] != "audit_only_not_training_approved" or spec["license_status"] != "pending":
        raise ValueError("source status mismatch")
    raw_dir = args.root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, entry, raw_dir=raw_dir, dataset=spec["dataset"],
                        revision=spec["revision"], endpoint=args.endpoint): entry["path"]
            for entry in spec["files"]
        }
        for future in as_completed(futures):
            result = future.result()
            reports.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    manifest = {
        "stage": "raw_download_verified_audit_only",
        "spec_sha256": sha256_file(args.spec),
        "dataset": spec["dataset"], "revision": spec["revision"],
        "license_status": spec["license_status"],
        "files": sorted(reports, key=lambda x: x["file"]),
    }
    temporary = args.root / "raw_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.root / "raw_manifest.json")
    print(json.dumps({"complete": True, "files": len(reports),
                      "bytes": sum(x["bytes"] for x in reports)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
