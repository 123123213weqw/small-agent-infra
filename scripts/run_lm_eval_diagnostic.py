#!/usr/bin/env python3
"""Run pinned, offline lm-eval tasks against a non-distributable diagnostic export."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--task-dir", type=Path, required=True)
    p.add_argument("--eval-data", type=Path, required=True)
    p.add_argument("--eval-deps", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=["local_arc_easy"])
    p.add_argument("--limit", type=int, help="Smoke test only; omit for real task totals")
    p.add_argument("--fewshot", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", action="store_true", help="Verify inputs and show command without creating outputs")
    args = p.parse_args()

    model = args.model.resolve()
    task_dir = args.task_dir.resolve()
    deps = args.eval_deps.resolve()
    output = args.output.resolve()
    if not args.dry_run and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evaluation output already exists: {output}")
    export_path = model / "export_manifest.json"
    export = json.loads(export_path.read_text(encoding="utf-8"))
    if export["stage"] != "internal_non_distributable_hf_export":
        raise ValueError("this evaluator requires the internal diagnostic export")
    for name, item in export["files"].items():
        path = model / name
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"export file changed: {name}")
    local_manifest_path = task_dir / "manifest.json"
    local = json.loads(local_manifest_path.read_text(encoding="utf-8"))
    if local["stage"] != "local_mirror_of_official_lm_eval_tasks":
        raise ValueError("not a pinned local task mirror")
    data_root = args.eval_data.resolve()
    for name, item in local["files"].items():
        path = task_dir / name if name.endswith((".yaml", ".py")) else data_root / name
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"task or benchmark data changed: {name}")
    for task in args.tasks:
        if not task.startswith("local_"):
            raise ValueError(f"unapproved task name: {task}")
        yaml = task_dir / f"{task}.yaml"
        if not yaml.is_file() or sha256_file(yaml) != local["files"][yaml.name]["sha256"]:
            raise ValueError(f"local task changed: {task}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(deps) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HOME"] = str(args.cache_dir.resolve())
    env["HF_DATASETS_CACHE"] = str((args.cache_dir / "datasets").resolve())
    env["HF_DATASETS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    cmd = [
        sys.executable, "-m", "lm_eval", "run",
        "--model", "hf",
        "--model_args", f"pretrained={model},dtype=bfloat16,trust_remote_code=False",
        "--include_path", str(task_dir),
        "--tasks", ",".join(args.tasks),
        "--num_fewshot", str(args.fewshot),
        "--batch_size", "1",
        "--device", args.device,
        "--output_path", str(output / "results"),
        "--log_samples", "--seed", str(args.seed),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    isolated_versions = {
        dist.metadata["Name"].lower().replace("_", "-"): dist.version
        for dist in importlib.metadata.distributions(path=[str(deps)])
        if dist.metadata.get("Name")
    }
    versions = {}
    for package in ("torch", "transformers", "lm-eval", "datasets", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = isolated_versions.get(package)
    report = {
        "stage": "internal_diagnostic_lm_eval",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_step": export["checkpoint_step"],
        "export_manifest_sha256": sha256_file(export_path),
        "task_manifest_sha256": sha256_file(local_manifest_path),
        "tasks": args.tasks,
        "limit": args.limit,
        "pilot_only": args.limit is not None,
        "fewshot": args.fewshot,
        "seed": args.seed,
        "versions": versions,
        "command": cmd,
    }
    if args.dry_run:
        report["status"] = "preflight_passed"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "eval_manifest.json"
    write_json(manifest_path, report)
    with (output / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    report["status"] = "completed" if code == 0 else "failed"
    report["ended_at"] = datetime.now(timezone.utc).isoformat()
    report["exit_code"] = code
    report["result_files"] = [str(p.relative_to(output)) for p in sorted((output / "results").rglob("results_*.json"))]
    write_json(manifest_path, report)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
