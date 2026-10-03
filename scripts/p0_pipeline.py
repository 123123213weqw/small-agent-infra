#!/usr/bin/env python3
"""Fail-closed, resumable coordinator for the P0 curation stages.

The manifest deliberately contains commands, not credentials. A stage is only
recorded after its output files and optional read-only check have succeeded.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time


SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
               "-o", "ControlMaster=auto", "-o", "ControlPersist=600",
               "-o", "ControlPath=/tmp/p0-pipeline-%C"]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def remote(host: str, command: str, *, capture: bool = False,
           retry_transport: bool = False) -> subprocess.CompletedProcess:
    argv = ["bash", "-lc", command] if host == "local" else [
        "ssh", *SSH_OPTIONS, host,
        "bash", "-lc", shlex.quote(command)
    ]
    for attempt in range(3 if retry_transport else 1):
        completed = subprocess.run(argv, text=True, capture_output=capture)
        if completed.returncode == 0:
            return completed
        if host != "local" and completed.returncode == 255 and retry_transport and attempt < 2:
            time.sleep(2 * (attempt + 1))
            continue
        raise subprocess.CalledProcessError(completed.returncode, argv,
                                            output=completed.stdout, stderr=completed.stderr)
    raise AssertionError("unreachable")


def file_hash(host: str, path: str) -> str:
    # Hash only explicit provenance files (normally manifests), not whole data directories.
    command = "sha256sum -- " + shlex.quote(path)
    if host == "local":
        command = "shasum -a 256 -- " + shlex.quote(path)
    return remote(host, command, capture=True, retry_transport=True).stdout.split()[0]


def exists(host: str, path: str) -> bool:
    try:
        remote(host, "test -f " + shlex.quote(path), capture=True, retry_transport=True)
        return True
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 1:
            return False
        raise RuntimeError(f"could not check {host}:{path}; transport/remote exit {exc.returncode}") from exc


def verify(stage: dict) -> dict:
    for check in stage.get("checks", []):
        remote(check["host"], check["command"], capture=True, retry_transport=True)
    result = {}
    for item in stage["outputs"]:
        host, path = item["host"], item["path"]
        if not exists(host, path):
            raise RuntimeError(f"missing output: {host}:{path}")
        result[f"{host}:{path}"] = file_hash(host, path)
    return result


def input_hashes(stage: dict) -> dict:
    result = {}
    for item in stage.get("inputs", []):
        host, path = item["host"], item["path"]
        result[f"{host}:{path}"] = file_hash(host, path)
    return result


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".pipeline-", delete=False) as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        name = f.name
    os.replace(name, path)


def validate(config: dict) -> None:
    if config.get("version") != 1 or not isinstance(config.get("stages"), list):
        raise ValueError("expected version 1 and stages list")
    seen = set()
    for stage in config["stages"]:
        name = stage["name"]
        if name in seen or not name.replace("_", "").isalnum():
            raise ValueError(f"duplicate/invalid stage name: {name}")
        seen.add(name)
        if stage["kind"] not in ("command", "paid", "copy") or not stage.get("outputs"):
            raise ValueError(f"invalid stage: {name}")
        if stage["kind"] == "paid":
            if not stage.get("checks"):
                raise ValueError(f"paid stage requires a read-only coverage check: {name}")
            if not (stage.get("budget_usd", 0) > 0 and len(stage.get("expected_key_sha256", "")) == 64):
                raise ValueError(f"paid stage lacks explicit budget/account fingerprint: {name}")
            if "{budget_usd}" not in stage["command"] or "{expected_key_sha256}" not in stage["command"]:
                raise ValueError(f"paid command must pass both budget and key fingerprint: {name}")
        if stage["kind"] == "copy" and len(stage.get("inputs", [])) != 1:
            raise ValueError(f"copy stage needs exactly one source: {name}")


def copy_one(stage: dict) -> None:
    source = stage["inputs"][0]
    target = stage["outputs"][0]
    if len(stage["outputs"]) != 1:
        raise ValueError("copy stage needs exactly one destination")
    source_hash = file_hash(source["host"], source["path"])
    if exists(target["host"], target["path"]):
        if file_hash(target["host"], target["path"]) != source_hash:
            raise RuntimeError("destination exists with a different hash; refusing overwrite")
        return
    # scp uses SSH configuration and an atomic remote rename; no credentials in the manifest.
    if source["host"] == "local" or target["host"] == "local":
        raise ValueError("cross-host copy expects two SSH hosts")
    temporary = target["path"] + ".pipeline-part"
    remote(target["host"], "mkdir -p " + shlex.quote(str(Path(target["path"]).parent)))
    ssh_options = SSH_OPTIONS
    reader = subprocess.Popen(["ssh", *ssh_options, source["host"],
                               "cat -- " + shlex.quote(source["path"])], stdout=subprocess.PIPE)
    try:
        writer = subprocess.run(["ssh", *ssh_options, target["host"],
                                 "cat > " + shlex.quote(temporary)], stdin=reader.stdout, check=True)
        assert writer.returncode == 0
    finally:
        if reader.stdout:
            reader.stdout.close()
    if reader.wait() != 0 or file_hash(target["host"], temporary) != source_hash:
        raise RuntimeError("copy failed or checksum mismatch; partial file left for inspection")
    remote(target["host"], "mv -- " + shlex.quote(temporary) + " " + shlex.quote(target["path"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "status", "run", "adopt", "adopt-all"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--stage", help="Required for adopt: adopt exactly this existing stage")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--max-paid-usd", type=float)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate(config)
    if args.action == "adopt" and not args.stage:
        parser.error("adopt requires --stage")
    if args.action == "adopt-all" and args.stage:
        parser.error("adopt-all does not accept --stage")
    if args.stage and args.stage not in {stage["name"] for stage in config["stages"]}:
        parser.error("unknown --stage")
    if args.action == "plan":
        for stage in config["stages"]:
            print(stage["name"], stage["kind"], stage.get("host", ""),
                  f"cap=${stage['budget_usd']}" if stage["kind"] == "paid" else "")
        return
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with (args.state.parent / (args.state.name + ".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(args.state.read_text()) if args.state.exists() else {"version": 1, "stages": {}}
        pending_paid_cap = sum(stage["budget_usd"] for stage in config["stages"]
                               if stage["kind"] == "paid" and stage["name"] not in state["stages"])
        if args.action == "run" and pending_paid_cap and (
            not args.allow_paid or args.max_paid_usd is None or args.max_paid_usd < pending_paid_cap
        ):
            raise RuntimeError(f"pending paid cap ${pending_paid_cap}; explicit --allow-paid and "
                               "--max-paid-usd covering all pending stages required")
        for stage in config["stages"]:
            name = stage["name"]
            spec = digest(stage)
            prior = state["stages"].get(name)
            if prior:
                if prior["spec"] != spec or prior["inputs"] != input_hashes(stage) or prior["outputs"] != verify(stage):
                    raise RuntimeError(f"completed stage changed: {name}; new output version required")
                print(f"SKIP {name}: verified", flush=True)
                continue
            if args.action == "adopt" and name != args.stage:
                continue
            print(f"PENDING {name}", flush=True)
            if args.action == "status":
                continue
            inputs = input_hashes(stage)
            if args.action in ("adopt", "adopt-all"):
                if not stage.get("checks"):
                    raise RuntimeError(f"cannot adopt {name} without an independent read-only check")
                outputs = verify(stage)
                state["stages"][name] = {"spec": spec, "inputs": inputs, "outputs": outputs}
                atomic_json(args.state, state)
                print(f"ADOPT {name}: verified existing artifacts", flush=True)
                if args.action == "adopt":
                    return
                continue
            if all(exists(item["host"], item["path"]) for item in stage["outputs"]):
                raise RuntimeError(f"untracked outputs already exist for {name}; run adopt after independent verification")
            if stage["kind"] == "paid":
                if not args.allow_paid or args.max_paid_usd is None or args.max_paid_usd < stage["budget_usd"]:
                    raise RuntimeError(f"paid stage {name} requires --allow-paid and sufficient --max-paid-usd")
                command = stage["command"].replace("{budget_usd}", str(stage["budget_usd"])).replace(
                    "{expected_key_sha256}", stage["expected_key_sha256"])
                remote(stage["host"], command)
            elif stage["kind"] == "command":
                remote(stage["host"], stage["command"])
            else:
                copy_one(stage)
            outputs = verify(stage)
            state["stages"][name] = {"spec": spec, "inputs": inputs, "outputs": outputs}
            atomic_json(args.state, state)
            print(f"DONE {name}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PIPELINE_STOP: {exc}", file=sys.stderr)
        raise SystemExit(1)
