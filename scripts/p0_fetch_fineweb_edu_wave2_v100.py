#!/usr/bin/env python3
"""Fetch pinned FineWeb-Edu Wave2 shards directly on V100 via hf-mirror.

Downloads are resumable. No shard becomes visible to the prefilter until its
byte count and SHA-256 match the pinned manifest. This is raw ingestion only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["version"] != "p0_fineweb_edu_wave2_v1" or len(manifest["items"]) != 13:
        raise ValueError("unexpected manifest")
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for number, item in enumerate(manifest["items"], 1):
        name = Path(item["path"]).name
        final = raw_dir / name
        partial = raw_dir / f"{name}.download"
        if final.exists():
            if final.stat().st_size != item["bytes"] or sha256_file(final) != item["sha256"]:
                raise ValueError(f"existing final has wrong size or SHA-256: {final}")
            print(f"[{number}/13] {name}: existing final verified", flush=True)
            continue
        url = (f"https://hf-mirror.com/datasets/{manifest['dataset']}/resolve/"
               f"{manifest['revision']}/{item['path']}")
        for attempt in range(2):
            print(f"[{number}/13] {name}: aria2c hf-mirror download/resume", flush=True)
            command = ("aria2c", "--continue=true", "--file-allocation=none",
                       "--auto-file-renaming=false", "--allow-overwrite=false",
                       "--max-connection-per-server=8", "--split=8",
                       "--min-split-size=4M", "--connect-timeout=20", "--timeout=60",
                       "--max-tries=12", "--retry-wait=10", "--summary-interval=60",
                       "--download-result=hide", "--dir", str(raw_dir),
                       "--out", partial.name, url)
            subprocess.run(command, check=True)
            if partial.stat().st_size == item["bytes"] and sha256_file(partial) == item["sha256"]:
                break
            if attempt:
                raise ValueError(f"hf-mirror bytes fail pinned SHA-256: {name}")
            partial.unlink(missing_ok=True)
            partial.with_name(partial.name + ".aria2").unlink(missing_ok=True)
        partial.replace(final)
        # A 13 MiB partial from the abandoned V100<-L40 rsync attempt is not
        # part of this downloader; remove it only after the verified final is
        # safely present.
        final.with_name(final.name + ".part").unlink(missing_ok=True)
        print(f"[{number}/13] {name}: V100 verified {item['bytes']} bytes", flush=True)
    summary = {"status": "raw_ingested_not_training_approved", "shards": 13,
               "bytes": sum(x["bytes"] for x in manifest["items"]),
               "source": "hf-mirror.com", "revision": manifest["revision"]}
    (root / "INGESTED.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
