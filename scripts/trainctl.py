#!/usr/bin/env python3
"""Fail-closed preflight and launcher for single-node L40 training runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Sequence


DEFAULT_ENVIRONMENT_ROOT = Path(
    "/srv/small-agent/experiments/small-agent-base-1b-v1-engineering"
)
DEFAULT_LOCK = Path("configs/gdn/p0_l40_fast_env_v1.json")
DEFAULT_PYTHON = Path("/opt/small-agent/venvs/small-agent-l40/bin/python")
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class PreflightError(RuntimeError):
    """A condition that must prevent launch."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PreflightError(f"missing JSON file: {path}") from error
    except json.JSONDecodeError as error:
        raise PreflightError(f"invalid JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise PreflightError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as error:
        raise PreflightError(f"missing file: {path}") from error
    return digest.hexdigest()


def sha256_source_tree(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((root / "src" / "small_agent").rglob("*.py"))
    paths += [root / "scripts" / "trainctl.py"]
    for path in paths:
        if not path.is_file():
            raise PreflightError(f"missing source file: {path}")
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def find_repository_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "src" / "small_agent").is_dir() and (
            candidate / "configs"
        ).is_dir():
            return candidate
    raise PreflightError(f"cannot locate repository root from {start}")


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def parse_visible_devices(value: str) -> list[int]:
    fields = [field.strip() for field in value.split(",") if field.strip()]
    if not fields:
        raise PreflightError("CUDA_VISIBLE_DEVICES is empty")
    if any(not field.isdigit() for field in fields):
        raise PreflightError("only physical numeric GPU indices are supported")
    devices = [int(field) for field in fields]
    if len(devices) != len(set(devices)):
        raise PreflightError("CUDA_VISIBLE_DEVICES contains duplicate indices")
    return devices


def parse_csv(text: str) -> list[list[str]]:
    return [[field.strip() for field in row] for row in csv.reader(text.splitlines()) if row]


def query_gpus(runner: CommandRunner = subprocess.run) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = runner(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightError(f"cannot query GPUs with nvidia-smi: {error}") from error
    rows = parse_csv(result.stdout)
    gpus: list[dict[str, Any]] = []
    for row in rows:
        if len(row) != 5:
            raise PreflightError(f"unexpected nvidia-smi GPU row: {row}")
        gpus.append(
            {
                "index": int(row[0]),
                "uuid": row[1],
                "name": row[2],
                "memory_total_mib": int(row[3]),
                "memory_used_mib": int(row[4]),
            }
        )
    return gpus


def query_compute_processes(runner: CommandRunner = subprocess.run) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = runner(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        # Some driver versions use a nonzero status when there are no processes.
        if not (error.stdout or "").strip():
            return []
        raise PreflightError(f"cannot query GPU processes: {error}") from error
    except OSError as error:
        raise PreflightError(f"cannot query GPU processes: {error}") from error
    processes = []
    for row in parse_csv(result.stdout):
        if len(row) != 4:
            raise PreflightError(f"unexpected nvidia-smi process row: {row}")
        processes.append(
            {
                "gpu_uuid": row[0],
                "pid": int(row[1]),
                "process_name": row[2],
                "used_memory_mib": int(row[3]),
            }
        )
    return processes


def validate_gpu_selection(
    gpus: list[dict[str, Any]],
    processes: list[dict[str, Any]],
    selected_indices: list[int],
    nproc_per_node: int,
    expected_name: str,
    *,
    allow_shared: bool = False,
    minimum_free_memory_mib: int = 0,
    maximum_existing_memory_mib: int = 0,
) -> list[dict[str, Any]]:
    if len(selected_indices) != nproc_per_node:
        raise PreflightError(
            f"selected {len(selected_indices)} GPUs but nproc_per_node={nproc_per_node}"
        )
    by_index = {gpu["index"]: gpu for gpu in gpus}
    missing = [index for index in selected_indices if index not in by_index]
    if missing:
        raise PreflightError(f"selected GPU indices do not exist: {missing}")
    selected = [by_index[index] for index in selected_indices]
    bad_names = [gpu for gpu in selected if gpu["name"] != expected_name]
    if bad_names:
        raise PreflightError(
            f"GPU model mismatch; expected {expected_name!r}, got "
            + ", ".join(gpu["name"] for gpu in bad_names)
        )
    selected_uuids = {gpu["uuid"] for gpu in selected}
    busy = [process for process in processes if process["gpu_uuid"] in selected_uuids]
    if busy and not allow_shared:
        summary = ", ".join(
            f"pid={item['pid']} gpu={item['gpu_uuid']} mem={item['used_memory_mib']}MiB"
            for item in busy
        )
        raise PreflightError(f"selected GPUs have active compute processes: {summary}")
    if allow_shared:
        if minimum_free_memory_mib < 1 or maximum_existing_memory_mib < 1:
            raise PreflightError("shared GPU limits must be positive")
        for gpu in selected:
            existing = [p for p in busy if p["gpu_uuid"] == gpu["uuid"]]
            free_mib = gpu["memory_total_mib"] - gpu["memory_used_mib"]
            existing_mib = sum(p["used_memory_mib"] for p in existing)
            if free_mib < minimum_free_memory_mib:
                raise PreflightError(
                    f"GPU {gpu['index']} has only {free_mib} MiB free for shared run"
                )
            if existing_mib > maximum_existing_memory_mib:
                raise PreflightError(
                    f"GPU {gpu['index']} has {existing_mib} MiB existing compute allocation"
                )
    return selected


def validate_configuration(
    root: Path, run_config_path: Path
) -> tuple[dict[str, Any], Path, dict[str, Any], Path, dict[str, Any], Path, Path]:
    run_config = load_json(run_config_path)
    required = {
        "model_spec",
        "data_dir",
        "output_root",
        "purpose",
        "distributable",
        "context_length",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "optimizer",
        "scheduler",
        "validation",
    }
    missing = sorted(required - run_config.keys())
    if missing:
        raise PreflightError(f"run config is missing fields: {missing}")
    for field in ("context_length", "micro_batch_size", "gradient_accumulation_steps", "max_steps"):
        if int(run_config[field]) < 1:
            raise PreflightError(f"run config {field} must be positive")
    hook = run_config.get("distributed", {}).get("communication_hook", "none")
    if hook not in {"none", "bf16"}:
        raise PreflightError(f"unsupported DDP communication hook: {hook}")

    model_path = resolve_path(root, run_config["model_spec"])
    model_spec = load_json(model_path)
    model_config = model_spec.get("config", {})
    if int(run_config["context_length"]) > int(model_config["max_position_embeddings"]):
        raise PreflightError("context length exceeds model max_position_embeddings")

    data_dir = resolve_path(root, run_config["data_dir"])
    manifest_path = data_dir / "manifest.json"
    data_manifest = load_json(manifest_path)
    reader = run_config.get("data_reader", "indexed_v1")
    expected_dtype = "uint16-le" if reader == "sharded_v1" else "uint16"
    if data_manifest.get("dtype") != expected_dtype:
        raise PreflightError(f"unsupported token dtype for {reader}: {data_manifest.get('dtype')!r}")
    if int(data_manifest.get("vocab_size", -1)) != int(model_config["vocab_size"]):
        raise PreflightError("data/model vocab_size mismatch")
    if int(data_manifest.get("eos_token_id", -1)) != int(model_config["eos_token_id"]):
        raise PreflightError("data/model eos_token_id mismatch")
    if reader == "sharded_v1":
        pipeline = run_config.get("data_pipeline", {})
        workers = int(pipeline.get("num_workers", 0))
        if workers < 0 or (workers and int(pipeline.get("prefetch_factor", 2)) < 1):
            raise PreflightError("invalid sharded data_pipeline workers/prefetch_factor")
        if data_manifest.get("version") != "sharded_token_stream_v1":
            raise PreflightError("unsupported sharded manifest version")
        if int(data_manifest.get("sequence_length", -1)) != int(run_config["context_length"]):
            raise PreflightError("shard sequence length differs from run context length")
        if data_manifest.get("index_dtype") != "<QII":
            raise PreflightError("unsupported shard index dtype")
        splits_seen: set[str] = set()
        for expected_id, entry in enumerate(data_manifest.get("shards", [])):
            if int(entry["shard_id"]) != expected_id:
                raise PreflightError("shard IDs are not ordered and contiguous")
            splits_seen.add(entry["split"])
            for kind, width, count_key in (("bin", 2, "tokens"), ("idx", 16, "samples")):
                filename = str(entry[kind])
                if Path(filename).name != filename or filename in {".", ".."}:
                    raise PreflightError(f"unsafe shard filename: {filename}")
                path = data_dir / filename
                if not path.is_file():
                    raise PreflightError(f"missing shard file: {path}")
                expected_bytes = int(entry[count_key]) * width
                if int(entry[f"{kind}_bytes"]) != expected_bytes or path.stat().st_size != expected_bytes:
                    raise PreflightError(f"shard {kind} byte count mismatch: {path}")
                if sha256_file(path) != entry[f"{kind}_sha256"]:
                    raise PreflightError(f"shard {kind} SHA-256 mismatch: {path}")
        if not {"train", "validation"}.issubset(splits_seen):
            raise PreflightError("sharded data is missing train/validation")
        tokenizer_json = run_config.get("tokenizer_json")
        if tokenizer_json and sha256_file(resolve_path(root, tokenizer_json)) != data_manifest.get("tokenizer_sha256"):
            raise PreflightError("tokenizer JSON SHA-256 differs from sharded data")
    elif reader == "indexed_v1":
        for split in ("train", "validation"):
            entry = data_manifest.get("splits", {}).get(split)
            if not isinstance(entry, dict):
                raise PreflightError(f"data manifest is missing {split!r} split")
            path = data_dir / entry["file"]
            if not path.is_file():
                raise PreflightError(f"missing {split} token stream: {path}")
            if path.stat().st_size != int(entry["bytes"]):
                raise PreflightError(f"{split} token stream byte count mismatch")
            actual_hash = sha256_file(path)
            if actual_hash != entry["sha256"]:
                raise PreflightError(f"{split} token stream SHA-256 mismatch")
    else:
        raise PreflightError(f"unsupported data reader: {reader}")
    eligible = data_manifest.get("license_gate", {}).get("training_eligible")
    stage = str(data_manifest.get("stage", ""))
    if eligible is False and not (
        run_config["distributable"] is False and "smoke" in stage
    ):
        raise PreflightError("data manifest is not training-eligible")
    output_root = resolve_path(root, run_config["output_root"])
    return (
        run_config,
        model_path,
        model_spec,
        manifest_path,
        data_manifest,
        data_dir,
        output_root,
    )


def resolve_source_revision(root: Path, explicit: str | None) -> str:
    if explicit:
        return explicit.strip()
    environment = os.environ.get("SMALL_AGENT_SOURCE_COMMIT")
    if environment:
        return environment.strip()
    marker = root / ".source_commit"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip():
        return marker.read_text(encoding="utf-8").strip()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightError(
            "source revision is unknown; pass --source-commit or create .source_commit"
        ) from error


def nearest_existing(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise PreflightError(f"cannot find an existing parent for {path}")
    return candidate


def check_disk(
    output_root: Path, minimum_free_gib: float, model_spec: dict[str, Any], keep: int
) -> dict[str, Any]:
    existing = nearest_existing(output_root)
    usage = shutil.disk_usage(existing)
    free_gib = usage.free / 2**30
    parameters = int(model_spec.get("expected_parameters", 0))
    # FP32 weights + gradients + two Adam moments, plus modest metadata overhead.
    estimated_checkpoint_gib = parameters * 12 * 1.01 / 2**30
    required_gib = max(minimum_free_gib, estimated_checkpoint_gib * (max(keep, 1) + 1))
    if free_gib < required_gib:
        raise PreflightError(
            f"insufficient disk: {free_gib:.1f} GiB free, {required_gib:.1f} GiB required"
        )
    return {
        "filesystem_path": str(existing),
        "free_gib": round(free_gib, 3),
        "required_gib": round(required_gib, 3),
        "estimated_full_checkpoint_gib": round(estimated_checkpoint_gib, 3),
    }


def build_environment(root: Path, environment_root: Path, cuda_devices: str) -> dict[str, str]:
    environment = os.environ.copy()
    prefixes = [environment_root / "deps", root / "src", root]
    environment.update(
        {
            "PYTHONPATH": ":".join(str(path) for path in prefixes),
            "PYTHONNOUSERSITE": "1",
            "TRITON_CACHE_DIR": str(environment_root / "triton-cache"),
            "TMPDIR": str(environment_root / "tmp"),
            "OMP_NUM_THREADS": environment.get("OMP_NUM_THREADS", "4"),
            "CUDA_VISIBLE_DEVICES": cuda_devices,
        }
    )
    return environment


def verify_runtime(
    python: Path,
    lock: dict[str, Any],
    environment: dict[str, str],
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    if not python.is_file() or not os.access(python, os.X_OK):
        raise PreflightError(f"pinned Python is missing or not executable: {python}")
    for variable in ("TRITON_CACHE_DIR", "TMPDIR"):
        Path(environment[variable]).mkdir(parents=True, exist_ok=True)
    probe = r'''
import json
from importlib.metadata import version
import torch
import fla
import causal_conv1d
from transformers.models.qwen3_next import modeling_qwen3_next as m

expected = json.loads(__import__("os").environ["SMALL_AGENT_EXPECTED_PACKAGES"])
versions = {name: version(name) for name in expected}
mismatches = {name: {"expected": expected[name], "actual": actual}
              for name, actual in versions.items() if actual != expected[name]}
result = {
    "packages": versions,
    "mismatches": mismatches,
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    "qwen3_next_fast_path": bool(getattr(m, "is_fast_path_available", False)),
    "fused_rms_norm_gated": getattr(m, "FusedRMSNormGated", None) is not None,
}
print(json.dumps(result, sort_keys=True))
'''
    child_environment = environment.copy()
    child_environment["CUDA_VISIBLE_DEVICES"] = environment["CUDA_VISIBLE_DEVICES"].split(",")[0]
    child_environment["SMALL_AGENT_EXPECTED_PACKAGES"] = json.dumps(lock["packages"])
    try:
        result = runner(
            [str(python), "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            env=child_environment,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "") or ""
        raise PreflightError(f"runtime probe failed: {stderr.strip() or error}") from error
    try:
        report = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise PreflightError(f"runtime probe returned invalid JSON: {result.stdout!r}") from error
    if report["mismatches"]:
        raise PreflightError(f"pinned package mismatch: {report['mismatches']}")
    for field in ("cuda_available", "qwen3_next_fast_path", "fused_rms_norm_gated"):
        if not report[field]:
            raise PreflightError(f"required runtime feature is disabled: {field}")
    if report["gpu_name"] != lock["gpu"]:
        raise PreflightError(
            f"runtime GPU mismatch: expected {lock['gpu']!r}, got {report['gpu_name']!r}"
        )
    return report


def build_launch_command(
    python: Path,
    nproc_per_node: int,
    run_config_path: Path,
    run_id: str,
    max_steps: int | None,
    resume: bool,
    save_final_checkpoint: bool,
) -> list[str]:
    command = [
        str(python),
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        "-m",
        "small_agent.training.trainer",
        "--run-config",
        str(run_config_path),
        "--run-id",
        run_id,
    ]
    if max_steps is not None:
        command += ["--max-steps", str(max_steps)]
    if resume:
        command.append("--resume")
    if save_final_checkpoint:
        command.append("--save-final-checkpoint")
    return command


def sanitized_unit(run_id: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip("-.")
    if not suffix:
        raise PreflightError("run id cannot be converted to a systemd unit name")
    return f"small-agent-{suffix[:180]}"


def run_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    run_config_path = args.run_config.resolve()
    root = find_repository_root(run_config_path)
    (
        run_config,
        model_path,
        model_spec,
        manifest_path,
        data_manifest,
        data_dir,
        output_root,
    ) = validate_configuration(root, run_config_path)
    run_dir = output_root / args.run_id
    if args.resume:
        if not run_dir.is_dir():
            raise PreflightError(f"resume run directory does not exist: {run_dir}")
        if not (run_dir / "run_manifest.json").is_file():
            raise PreflightError("resume run is missing run_manifest.json")
        if not (run_dir / "checkpoints" / "latest.json").is_file():
            raise PreflightError("resume run has no latest checkpoint")
    elif run_dir.exists():
        raise PreflightError(f"run id already exists: {run_dir}")

    lock_path = resolve_path(root, args.environment_lock)
    lock = load_json(lock_path)
    environment_root = args.environment_root.resolve()
    if not (environment_root / "deps").is_dir():
        raise PreflightError(f"dependency directory is missing: {environment_root / 'deps'}")
    source_revision = resolve_source_revision(root, args.source_commit)
    selected_indices = parse_visible_devices(args.cuda_visible_devices)
    selected_gpus = validate_gpu_selection(
        query_gpus(),
        query_compute_processes(),
        selected_indices,
        args.nproc_per_node,
        str(lock["gpu"]),
        allow_shared=args.allow_shared_gpus,
        minimum_free_memory_mib=int(args.minimum_free_gpu_memory_gib * 1024),
        maximum_existing_memory_mib=int(args.maximum_existing_gpu_memory_gib * 1024),
    )
    disk = check_disk(
        output_root,
        args.minimum_free_disk_gib,
        model_spec,
        int(run_config.get("checkpoint", {}).get("keep_full", 1)),
    )
    python = Path(lock.get("base_python", DEFAULT_PYTHON))
    environment = build_environment(root, environment_root, args.cuda_visible_devices)
    environment["SMALL_AGENT_SOURCE_COMMIT"] = source_revision
    runtime = verify_runtime(python, lock, environment)
    command = build_launch_command(
        python,
        args.nproc_per_node,
        run_config_path,
        args.run_id,
        args.max_steps,
        args.resume,
        args.save_final_checkpoint,
    )
    report = {
        "version": "trainctl_preflight_v1",
        "status": "passed",
        "checked_at": utc_now(),
        "run_id": args.run_id,
        "resume": args.resume,
        "repository_root": str(root),
        "source_revision": source_revision,
        "source_tree_sha256": sha256_source_tree(root),
        "run_config": {"path": str(run_config_path), "sha256": sha256_file(run_config_path)},
        "model_spec": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "data_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "stage": data_manifest.get("stage"),
            "training_eligible": data_manifest.get("license_gate", {}).get(
                "training_eligible"
            ),
            "data_dir": str(data_dir),
        },
        "environment": {
            "root": str(environment_root),
            "lock": str(lock_path),
            "lock_sha256": sha256_file(lock_path),
            "python": str(python),
        },
        "runtime": runtime,
        "distributed_socket_environment": {
            key: environment[key]
            for key in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME")
            if key in environment
        },
        "gpu": {
            "cuda_visible_devices": args.cuda_visible_devices,
            "nproc_per_node": args.nproc_per_node,
            "selected": selected_gpus,
            "allow_shared": args.allow_shared_gpus,
            "minimum_free_gpu_memory_gib": args.minimum_free_gpu_memory_gib,
            "maximum_existing_gpu_memory_gib": args.maximum_existing_gpu_memory_gib,
        },
        "disk": disk,
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "launch_command": command,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / ".preflight" / f"{args.run_id}.json"
    atomic_json(report_path, report)
    report["report_path"] = str(report_path)
    return report, environment, command


def launch_detached(
    root: Path, run_id: str, environment: dict[str, str], command: Sequence[str]
) -> str:
    unit = sanitized_unit(run_id)
    systemd_command = [
        "systemd-run",
        "--user",
        f"--unit={unit}",
        "--collect",
        f"--property=WorkingDirectory={root}",
    ]
    for key in (
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "TRITON_CACHE_DIR",
        "TMPDIR",
        "OMP_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "SMALL_AGENT_SOURCE_COMMIT",
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
    ):
        if key in environment:
            systemd_command.append(f"--setenv={key}={environment[key]}")
    systemd_command += list(command)
    try:
        subprocess.run(systemd_command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightError(f"failed to start systemd unit {unit}: {error}") from error
    return unit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("preflight", "launch"):
        child = subparsers.add_parser(action)
        child.add_argument("run_config", type=Path)
        child.add_argument("--run-id", required=True)
        child.add_argument("--nproc-per-node", type=int, default=8)
        child.add_argument(
            "--cuda-visible-devices",
            default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"),
        )
        child.add_argument("--allow-shared-gpus", action="store_true")
        child.add_argument("--minimum-free-gpu-memory-gib", type=float, default=24.0)
        child.add_argument("--maximum-existing-gpu-memory-gib", type=float, default=8.0)
        child.add_argument(
            "--environment-root",
            type=Path,
            default=Path(os.environ.get("SMALL_AGENT_GDN_ROOT", DEFAULT_ENVIRONMENT_ROOT)),
        )
        child.add_argument("--environment-lock", type=Path, default=DEFAULT_LOCK)
        child.add_argument("--source-commit")
        child.add_argument("--minimum-free-disk-gib", type=float, default=50.0)
        child.add_argument("--max-steps", type=int)
        child.add_argument("--resume", action="store_true")
        child.add_argument("--save-final-checkpoint", action="store_true")
        child.add_argument("--dry-run", action="store_true")
        if action == "launch":
            child.add_argument("--detach", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.nproc_per_node < 1:
        raise PreflightError("nproc_per_node must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise PreflightError("max_steps must be positive")
    report, environment, command = run_preflight(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.action == "preflight" or args.dry_run:
        return 0
    root = Path(report["repository_root"])
    if args.detach:
        unit = launch_detached(root, args.run_id, environment, command)
        print(
            json.dumps(
                {
                    "status": "launched",
                    "mode": "systemd-user",
                    "unit": unit,
                    "logs": f"journalctl --user -fu {unit}",
                }
            ),
            flush=True,
        )
        return 0
    os.chdir(root)
    os.execvpe(command[0], command, environment)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreflightError as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        raise SystemExit(2)
