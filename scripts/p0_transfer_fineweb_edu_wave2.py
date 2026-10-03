#!/usr/bin/env python3
"""Resume verified FineWeb-Edu shard transfer from Mac to L40 staging.

Only one shard is kept on Mac at a time. V100 independently pulls from L40,
verifies SHA-256, and writes a receipt on L40. Only then is the Mac copy
removed. This script does not approve data for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def run(*argv: str, capture: bool = False) -> str:
    result = subprocess.run(argv, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ""


def run_retry(*argv: str, capture: bool = False) -> str:
    """Retry transient SSH/Tailnet failures; never skip integrity checks."""
    for attempt in range(20):
        try:
            return run(*argv, capture=capture)
        except subprocess.CalledProcessError:
            if attempt == 19:
                raise
            delay = min(15 + attempt * 5, 90)
            print(f"remote command failed; retry {attempt + 1}/20 in {delay}s", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_digest(host: str, path: str) -> str | None:
    command = f"if test -f {shlex.quote(path)}; then sha256sum {shlex.quote(path)}; fi"
    out = run_retry("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                    "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
                    host, command, capture=True)
    return out.split()[0] if out else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--host", default="GPU_NODE")
    parser.add_argument("--limit", type=int, default=0, help="0 means all shards")
    args = parser.parse_args()
    # Keep the Mac awake while this LaunchAgent is active; caffeinate exits
    # automatically when the parent transfer process ends.
    subprocess.Popen(("/usr/bin/caffeinate", "-dimsu", "-w", str(os.getpid())),
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    manifest = json.loads(args.manifest.read_text())
    if manifest["version"] != "p0_fineweb_edu_wave2_v1":
        raise ValueError("unexpected manifest")
    if args.scratch.resolve() == Path("/"):
        raise ValueError("scratch cannot be filesystem root")
    args.scratch.mkdir(parents=True, exist_ok=True)
    remote_root = args.remote_root.rstrip("/")
    run_retry("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
              args.host, f"mkdir -p {shlex.quote(remote_root + '/raw')}")
    selected = manifest["items"][:args.limit or None]
    for number, item in enumerate(selected, 1):
        name = Path(item["path"]).name
        local = args.scratch / name
        remote = f"{remote_root}/raw/{name}"
        receipt = remote + ".received.sha256"
        print(f"[{number}/{len(selected)}] {name}: checking staging and V100 receipt", flush=True)
        receipt_value = run_retry("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                                  args.host,
                                  f"if test -f {shlex.quote(receipt)}; then cat {shlex.quote(receipt)}; fi",
                                  capture=True)
        if receipt_value == item["sha256"]:
            if local.exists():
                local.unlink()
            print(f"[{number}/{len(selected)}] {name}: already received by V100", flush=True)
            continue
        if receipt_value:
            raise ValueError(f"wrong V100 receipt for {name}")
        existing = remote_digest(args.host, remote)
        if existing is not None and existing != item["sha256"]:
            raise ValueError(f"remote final file has wrong SHA-256: {remote}")
        if existing is None:
            url = (f"https://huggingface.co/datasets/{manifest['dataset']}/resolve/"
                   f"{manifest['revision']}/{item['path']}")
            local_ok = (local.exists() and local.stat().st_size == item["bytes"]
                        and file_sha256(local) == item["sha256"])
            if not local_ok:
                for attempt in range(2):
                    run("curl", "--silent", "--show-error", "--fail", "--location", "--retry", "8",
                        "--retry-delay", "5", "--retry-all-errors", "--connect-timeout", "20",
                        "--continue-at", "-", "--output", str(local), url)
                    if local.stat().st_size == item["bytes"] and file_sha256(local) == item["sha256"]:
                        break
                    if attempt:
                        raise ValueError(f"local download failed integrity verification: {local}")
                    local.unlink()
            print(f"[{number}/{len(selected)}] {name}: Mac SHA-256 verified; staging", flush=True)
            remote_partial = remote + ".part"
            # macOS openrsync lacks --append-verify. SHA-256 is mandatory below.
            for attempt in range(2):
                run_retry("rsync", "--partial", "--append", "-e",
                          "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=10 -o ServerAliveCountMax=2",
                          str(local), f"{args.host}:{remote_partial}")
                if remote_digest(args.host, remote_partial) == item["sha256"]:
                    break
                if attempt:
                    raise ValueError(f"staged partial SHA-256 mismatch: {remote_partial}")
                run_retry("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                          args.host, f"rm -f -- {shlex.quote(remote_partial)}")
            run_retry("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                      args.host, f"mv -n {shlex.quote(remote_partial)} {shlex.quote(remote)}")
            if remote_digest(args.host, remote) != item["sha256"]:
                raise ValueError(f"staged final SHA-256 mismatch: {remote}")
        print(f"[{number}/{len(selected)}] {name}: staged; waiting for V100 verified receipt", flush=True)
        while True:
            receipt_value = run_retry(
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", args.host,
                f"if test -f {shlex.quote(receipt)}; then cat {shlex.quote(receipt)}; fi", capture=True)
            if receipt_value == item["sha256"]:
                break
            if receipt_value:
                raise ValueError(f"wrong V100 receipt for {name}")
            time.sleep(30)
        local.unlink(missing_ok=True)
        print(f"[{number}/{len(selected)}] {name}: V100 verified; Mac copy removed", flush=True)
    print(json.dumps({"transferred_or_verified": len(selected),
                      "status": "raw_only_not_training_approved"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
