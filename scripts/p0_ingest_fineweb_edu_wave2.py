#!/usr/bin/env python3
"""V100 pulls verified FineWeb-Edu staging shards from L40, then receipts them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def l40(host: str, command: str, *, missing_ok: bool = False) -> str:
    result = subprocess.run(("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                             "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
                             host, command), check=False, text=True, stdout=subprocess.PIPE)
    if not (missing_ok and result.returncode == 1):
        result.check_returncode()
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--stage-host", default="L40_Data_Pipeline")
    parser.add_argument("--wait-timeout", type=int, default=86400)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["version"] != "p0_fineweb_edu_wave2_v1" or len(manifest["items"]) != 13:
        raise ValueError("unexpected manifest")
    (root / "raw").mkdir(parents=True, exist_ok=True)
    for number, item in enumerate(manifest["items"], 1):
        name = Path(item["path"]).name
        final = root / "raw" / name
        partial = root / "raw" / f"{name}.part"
        staged = f"{args.stage_root.rstrip('/')}/raw/{name}"
        receipt = staged + ".received.sha256"
        if final.exists():
            if final.stat().st_size != item["bytes"] or sha256_file(final) != item["sha256"]:
                raise ValueError(f"V100 final integrity mismatch: {final}")
            print(f"[{number}/13] {name}: V100 already verified", flush=True)
        else:
            deadline = time.monotonic() + args.wait_timeout
            while True:
                if l40(args.stage_host, f"test -f {shlex.quote(staged)} && echo ready",
                       missing_ok=True) == "ready":
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"waiting for L40 staging: {staged}")
                time.sleep(30)
            for attempt in range(2):
                subprocess.run(("rsync", "--partial", "--append-verify", "-e",
                                "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=10 -o ServerAliveCountMax=2",
                                f"{args.stage_host}:{staged}", str(partial)), check=True)
                if partial.stat().st_size == item["bytes"] and sha256_file(partial) == item["sha256"]:
                    break
                if attempt:
                    raise ValueError(f"V100 partial integrity mismatch: {partial}")
                partial.unlink(missing_ok=True)
            partial.replace(final)
            print(f"[{number}/13] {name}: V100 SHA-256 verified", flush=True)
        # The receipt is written only after the final V100 file is verified.
        tmp_receipt = receipt + ".tmp"
        l40(args.stage_host,
            f"printf '%s\\n' {shlex.quote(item['sha256'])} > {shlex.quote(tmp_receipt)}"
            f" && mv -f {shlex.quote(tmp_receipt)} {shlex.quote(receipt)}"
            f" && rm -f -- {shlex.quote(staged)}")
        print(f"[{number}/13] {name}: L40 receipt written; staging copy removed", flush=True)
    summary = {"status": "raw_ingested_not_training_approved", "shards": 13,
               "bytes": sum(x["bytes"] for x in manifest["items"])}
    (root / "INGESTED.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
