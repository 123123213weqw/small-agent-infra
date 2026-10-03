#!/usr/bin/env python3
"""L40 fetches pinned FineWeb-Edu shards via hf-mirror for V100 ingestion.

The first shard is completed by the existing Mac transfer. This worker waits
for its V100 receipt, then downloads the remaining 12 shards one at a time.
Each shard must match its upstream LFS SHA-256 before it is made visible to
the V100 ingest service. No GPU or training approval is involved.
"""

from __future__ import annotations

import argparse
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


def wait_receipt(path: Path, expected: str, deadline: float) -> None:
    while True:
        if path.exists():
            actual = path.read_text().strip()
            if actual != expected:
                raise ValueError(f"wrong V100 receipt: {path}")
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"waiting for V100 receipt: {path}")
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wait-timeout", type=int, default=86400)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["version"] != "p0_fineweb_edu_wave2_v1" or len(manifest["items"]) != 13:
        raise ValueError("unexpected manifest")
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    first = manifest["items"][0]
    first_receipt = raw / (Path(first["path"]).name + ".received.sha256")
    wait_receipt(first_receipt, first["sha256"], time.monotonic() + args.wait_timeout)
    print("[1/13] first shard received by V100; L40 mirror fetch starts", flush=True)
    for number, item in enumerate(manifest["items"][1:], 2):
        name = Path(item["path"]).name
        final = raw / name
        partial = raw / f"{name}.part"
        receipt = raw / f"{name}.received.sha256"
        if receipt.exists():
            if receipt.read_text().strip() != item["sha256"]:
                raise ValueError(f"wrong V100 receipt: {receipt}")
            print(f"[{number}/13] {name}: already received by V100", flush=True)
            continue
        if final.exists():
            if final.stat().st_size != item["bytes"] or sha256_file(final) != item["sha256"]:
                raise ValueError(f"staged final integrity mismatch: {final}")
            print(f"[{number}/13] {name}: staged final already verified", flush=True)
        else:
            url = (f"https://hf-mirror.com/datasets/{manifest['dataset']}/resolve/"
                   f"{manifest['revision']}/{item['path']}")
            for attempt in range(2):
                print(f"[{number}/13] {name}: downloading via hf-mirror", flush=True)
                subprocess.run(("curl", "--silent", "--show-error", "--fail", "--location",
                                "--retry", "8", "--retry-delay", "5", "--retry-all-errors",
                                "--connect-timeout", "20", "--continue-at", "-",
                                "--output", str(partial), url), check=True)
                if partial.stat().st_size == item["bytes"] and sha256_file(partial) == item["sha256"]:
                    break
                if attempt:
                    raise ValueError(f"hf-mirror bytes do not match pinned SHA-256: {name}")
                partial.unlink(missing_ok=True)
            partial.replace(final)
            print(f"[{number}/13] {name}: SHA-256 verified and staged", flush=True)
        wait_receipt(receipt, item["sha256"], time.monotonic() + args.wait_timeout)
        print(f"[{number}/13] {name}: V100 receipt verified", flush=True)
    summary = {"status": "all_shards_received_by_V100_not_training_approved",
               "shards": len(manifest["items"])}
    (root / "MIRROR_FINISHED.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
