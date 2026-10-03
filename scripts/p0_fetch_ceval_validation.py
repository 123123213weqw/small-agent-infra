#!/usr/bin/env python3
"""Freeze official C-Eval validation Parquets on V100 for audit only."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import time
from urllib.parse import quote


REPO = "ceval/ceval-exam"
REVISION = "617524a00b307ff6f9933702f724131fe12ca7ce"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--endpoint", default="https://hf-mirror.com")
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    import requests
    from huggingface_hub import HfApi

    info = HfApi(endpoint=args.endpoint).dataset_info(REPO, revision=REVISION,
                                                       files_metadata=True, timeout=30)
    if info.sha != REVISION:
        raise ValueError("C-Eval revision drift")
    files = sorted((x for x in info.siblings if x.rfilename.endswith("/val-00000-of-00001.parquet")),
                   key=lambda x: x.rfilename)
    if len(files) != 52 or any(x.lfs is None or x.size is None for x in files):
        raise ValueError("unexpected C-Eval validation file list")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ceval-val-", dir=args.output_dir.parent) as tmp:
        stage = Path(tmp) / args.output_dir.name
        stage.mkdir()
        def download(item):
            path = stage / item.rfilename
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"{args.endpoint}/datasets/{REPO}/resolve/{REVISION}/{quote(item.rfilename)}"
            for attempt in range(3):
                try:
                    with requests.get(url, timeout=45, stream=True) as response:
                        response.raise_for_status()
                        with path.open("wb") as out:
                            for block in response.iter_content(chunk_size=1024 * 1024):
                                if block:
                                    out.write(block)
                    actual = sha256(path)
                    if path.stat().st_size != item.size or actual != item.lfs.sha256:
                        raise ValueError(f"C-Eval file integrity mismatch: {item.rfilename}")
                    return {"file": item.rfilename, "bytes": item.size, "sha256": actual}
                except (requests.RequestException, ValueError):
                    path.unlink(missing_ok=True)
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            raise AssertionError("unreachable")

        with ThreadPoolExecutor(max_workers=8) as executor:
            reports = list(executor.map(download, files))
        print(f"verified {len(reports)} validation shards", flush=True)
        report = {
            "stage": "ceval_validation_frozen_for_decontamination_audit_only",
            "source_repository": REPO,
            "source_revision": REVISION,
            "endpoint": args.endpoint,
            "files": reports,
            "training_eligible": False,
            "usage": "evaluation/decontamination reference only; never training input",
        }
        (stage / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        stage.rename(args.output_dir)
    print(json.dumps({"complete": True, "files": len(reports), "bytes": sum(x["bytes"] for x in reports)}),
          flush=True)


if __name__ == "__main__":
    main()
