#!/usr/bin/env python3
"""Fetch pinned HF Parquet shards via this host, relay to V100, verify both copies.

Downloads are serial. Relay outages can leave multiple verified local shards for retry.
This does not normalize or approve data.
Rerunning is safe: remotely verified shards are skipped; partial local downloads resume.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import time


def say(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S %z"), message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(command, 124, "", f"timeout after {timeout}s")


def ssh(host: str, command: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return run([
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
        "-o", "ControlMaster=no", "-o", "ControlPath=none",
        host, command,
    ], timeout=timeout)


def remote_verified(host: str, path: str, size: int, expected_hash: str) -> bool:
    check = ssh(host, "if test -f " + shlex.quote(path) + "; then "
                "stat -c %s " + shlex.quote(path) + " && sha256sum " + shlex.quote(path) +
                "; else echo MISSING; fi", timeout=300)
    if check.returncode != 0:
        return False
    lines = check.stdout.splitlines()
    return len(lines) >= 2 and lines[0] == str(size) and lines[1].split()[0] == expected_hash


def write_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def valid_item(item: dict) -> None:
    path = PurePosixPath(item["path"])
    if path.is_absolute() or ".." in path.parts or not path.name.endswith(".parquet"):
        raise ValueError(f"unsafe upstream path: {path}")
    if not item["source_id"].replace("_", "").isalnum():
        raise ValueError("unsafe source_id")
    if not isinstance(item["bytes"], int) or item["bytes"] <= 0:
        raise ValueError("bad expected byte count")
    if len(item["revision"]) != 40 or len(item["sha256"]) != 64:
        raise ValueError("missing pinned revision/hash")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--remote", default="CPU_NODE")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--hours", type=float, default=9.0)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("version") != "p0_overnight_upstream_v1" or not plan.get("items"):
        raise ValueError("wrong or empty download plan")
    for item in plan["items"]:
        valid_item(item)
    args.cache.mkdir(parents=True, exist_ok=True)
    lock = (args.cache / "downloader.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.cache / "progress.json"
    state = {"plan_sha256": sha256_file(args.plan), "completed": [], "failed": [], "started_at": time.time()}
    if state_path.exists():
        old = json.loads(state_path.read_text(encoding="utf-8"))
        if old.get("plan_sha256") != state["plan_sha256"]:
            raise ValueError("existing progress belongs to a different plan")
        state = old
    deadline = time.monotonic() + args.hours * 3600
    remote_root = args.remote_root.rstrip("/")
    if ssh(args.remote, "mkdir -p " + shlex.quote(remote_root), timeout=60).returncode == 0:
        run(["rsync", "-av", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none",
             str(args.plan), args.remote + ":" + remote_root + "/download_plan_v1.json"], timeout=180)
    for index, item in enumerate(plan["items"], 1):
        if time.monotonic() > deadline:
            say("Deadline reached; resume with the same command later")
            return 2
        identifier = item["source_id"] + "/" + PurePosixPath(item["path"]).name
        remote_dir = remote_root + "/raw/" + item["source_id"]
        remote_path = remote_dir + "/" + PurePosixPath(item["path"]).name
        say(f"[{index}/{len(plan['items'])}] {identifier} ({item['bytes']/1e6:.1f} MB)")
        if remote_verified(args.remote, remote_path, item["bytes"], item["sha256"]):
            say("Remote already verified; skip")
            if identifier not in state["completed"]:
                state["completed"].append(identifier)
                write_state(state_path, state)
            continue
        local_path = args.cache / (item["source_id"] + "__" + PurePosixPath(item["path"]).name)
        if shutil.disk_usage(args.cache).free < item["bytes"] + 5_000_000_000:
            say("Insufficient local free space; stop safely")
            return 3
        url = ("https://huggingface.co/datasets/" + item["dataset"] + "/resolve/" +
               item["revision"] + "/" + item["path"])
        transferred = False
        for attempt in range(1, 6):
            if time.monotonic() > deadline:
                break
            if not local_path.is_file() or local_path.stat().st_size != item["bytes"]:
                result = run([
                    "curl", "--fail", "--location", "--silent", "--show-error",
                    "--retry", "8", "--retry-all-errors", "--retry-delay", "5",
                    "--connect-timeout", "30", "--max-time", "3600", "--continue-at", "-",
                    "--output", str(local_path), url,
                ], timeout=4200)
                if result.returncode != 0:
                    say(f"Download retry {attempt}: {result.stderr.strip()[-200:]}")
                    time.sleep(15)
                    continue
            if local_path.stat().st_size != item["bytes"]:
                say(f"Size mismatch retry {attempt}")
                if local_path.stat().st_size > item["bytes"]:
                    local_path.unlink()
                continue
            if sha256_file(local_path) != item["sha256"]:
                say(f"SHA256 mismatch retry {attempt}; discard only this local temp file")
                local_path.unlink()
                continue
            mkdir = ssh(args.remote, "mkdir -p " + shlex.quote(remote_dir), timeout=60)
            if mkdir.returncode != 0:
                say(f"Remote unavailable retry {attempt}: {mkdir.stderr.strip()[-160:]}")
                time.sleep(30)
                continue
            transfer = run([
                "rsync", "-av", "--partial", "--inplace", "--timeout=180",
                "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ControlMaster=no -o ControlPath=none",
                str(local_path), args.remote + ":" + remote_path,
            ], timeout=3600)
            if transfer.returncode != 0:
                say(f"Relay retry {attempt}: {transfer.stderr.strip()[-200:]}")
                time.sleep(30)
                continue
            if remote_verified(args.remote, remote_path, item["bytes"], item["sha256"]):
                transferred = True
                break
            say(f"Remote checksum retry {attempt}")
            time.sleep(15)
        if not transferred:
            say("FAILED this file; retain local partial for a later retry")
            if identifier not in state["failed"]:
                state["failed"].append(identifier)
            write_state(state_path, state)
            continue
        local_path.unlink()
        if identifier not in state["completed"]:
            state["completed"].append(identifier)
        state["failed"] = [value for value in state["failed"] if value != identifier]
        write_state(state_path, state)
        say("Verified on V100; local temp removed")
    remote_manifest = remote_root + "/download_plan_v1.json"
    if ssh(args.remote, "mkdir -p " + shlex.quote(remote_root), timeout=60).returncode == 0:
        run(["rsync", "-av", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none",
             str(args.plan), args.remote + ":" + remote_manifest], timeout=180)
    say(f"Done: {len(state['completed'])}/{len(plan['items'])} verified; failed={len(state['failed'])}")
    return 0 if len(state["completed"]) == len(plan["items"]) else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        say(f"FATAL {type(error).__name__}: {error}")
        raise
