#!/usr/bin/env python3
"""Small persistent FIFO GPU queue for a host without a cluster scheduler."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def gpu_inventory():
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    uuid_to_index = {uuid.strip(): int(index.strip()) for index, uuid in (row.split(",", 1) for row in gpu_rows)}
    occupied = set()
    process_rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    for row in process_rows:
        if not row.strip() or "," not in row:
            continue
        uuid, _pid = row.split(",", 1)
        if uuid.strip() in uuid_to_index:
            occupied.add(uuid_to_index[uuid.strip()])
    return sorted(uuid_to_index.values()), occupied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(manifest["root"]).resolve()
    state_dir = Path(manifest["state_dir"]).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / "scheduler.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another scheduler already owns this queue")

    state_path = state_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    jobs = manifest["jobs"]
    children = {}
    for job in jobs:
        state.setdefault(job["id"], {"status": "pending"})
    atomic_json(state_path, state)

    while True:
        # Reconcile children through wrapper exit-code files, which also works
        # after an SSH disconnect or scheduler restart.
        for job in jobs:
            item = state[job["id"]]
            if item["status"] != "running":
                continue
            exit_path = state_dir / f"{job['id']}.exit"
            if exit_path.exists():
                code = int(exit_path.read_text().strip())
                item.update(status="done" if code == 0 else "failed", exit_code=code, finished_at=time.time())
                children.pop(job["id"], None)
                continue
            pid = int(item.get("pid", 0))
            if pid:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    item.update(status="failed", exit_code=None, finished_at=time.time(), reason="process vanished without exit marker")
                    children.pop(job["id"], None)

        done_ids = {job_id for job_id, item in state.items() if item["status"] == "done"}
        all_gpus, occupied = gpu_inventory()
        reserved = {int(item["gpu"]) for item in state.values() if item.get("status") == "running" and "gpu" in item}
        free_gpus = [gpu for gpu in all_gpus if gpu not in occupied and gpu not in reserved]

        for job in jobs:
            item = state[job["id"]]
            if item["status"] != "pending":
                continue
            if any(dependency not in done_ids for dependency in job.get("depends_on", [])):
                continue
            if any(not Path(path).exists() for path in job.get("ready_paths", [])):
                continue
            needs_gpu = bool(job.get("needs_gpu", True))
            if needs_gpu and not free_gpus:
                continue
            gpu = free_gpus.pop(0) if needs_gpu else None
            exit_path = state_dir / f"{job['id']}.exit"
            exit_path.unlink(missing_ok=True)
            log_path = state_dir / f"{job['id']}.log"
            command = job["command"]
            wrapped = f"set -o pipefail; {command}; rc=$?; printf '%s\\n' \"$rc\" > {exit_path}; exit $rc"
            env = os.environ.copy()
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                ["bash", "-lc", wrapped], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            children[job["id"]] = process
            item.update(status="running", pid=process.pid, started_at=time.time(), log=str(log_path))
            if gpu is not None:
                item["gpu"] = gpu
                occupied.add(gpu)
            atomic_json(state_path, state)

        atomic_json(state_path, state)
        if all(item["status"] in {"done", "failed"} for item in state.values()):
            break
        time.sleep(int(manifest.get("poll_seconds", 30)))


if __name__ == "__main__":
    main()
