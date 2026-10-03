"""Atomic, content-addressed full checkpoints for exact training resume."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_model, save_model

from small_agent.data import StatefulDistributedSampler
from small_agent.training.scheduler import TokenLRScheduler
from small_agent.training.kernel_state import capture_kernel_state, restore_kernel_state


VERSION = "small_agent_full_checkpoint_v1"
DISTRIBUTED_VERSION = "small_agent_distributed_full_checkpoint_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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


def _artifact(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def save_full_checkpoint(
    checkpoints_dir: Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: TokenLRScheduler,
    sampler: StatefulDistributedSampler,
    compatibility: dict[str, Any],
    keep: int = 2,
) -> tuple[Path, dict[str, Any]]:
    if step < 0 or keep < 1:
        raise ValueError("step must be non-negative and keep must be positive")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    final = checkpoints_dir / f"step_{step:08d}"
    if final.exists():
        raise FileExistsError(final)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=checkpoints_dir)
    )
    started = time.time()
    try:
        model_path = temporary / "model.safetensors"
        optimizer_path = temporary / "optimizer.pt"
        runtime_path = temporary / "runtime.pt"
        scheduler_path = temporary / "scheduler.json"
        sampler_path = temporary / "sampler.json"

        save_model(
            model,
            str(model_path),
            metadata={"format": "pt", "checkpoint_version": VERSION},
        )
        torch.save(optimizer.state_dict(), optimizer_path)
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        torch.save(
            {
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state(model_device)
                    if model_device.type == "cuda"
                    else None
                ),
                "kernel_autotune": capture_kernel_state(),
            },
            runtime_path,
        )
        atomic_json(scheduler_path, scheduler.state_dict())
        atomic_json(sampler_path, sampler.state_dict())
        artifacts = {
            path.name: _artifact(path)
            for path in (
                model_path,
                optimizer_path,
                runtime_path,
                scheduler_path,
                sampler_path,
            )
        }
        manifest = {
            "version": VERSION,
            "stage": "full_training_checkpoint",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "tokens_seen": scheduler.tokens_seen,
            "elapsed_seconds": round(time.time() - started, 3),
            "compatibility": compatibility,
            "artifacts": artifacts,
        }
        atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, final)
        atomic_json(
            checkpoints_dir / "latest.json",
            {
                "version": VERSION,
                "step": step,
                "directory": final.name,
                "manifest_sha256": sha256_file(final / "manifest.json"),
            },
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    completed = sorted(
        path
        for path in checkpoints_dir.glob("step_[0-9]*")
        if path.is_dir() and (path / "manifest.json").is_file()
    )
    for obsolete in completed[:-keep]:
        shutil.rmtree(obsolete)
    return final, manifest


def resolve_checkpoint(checkpoints_dir: Path, checkpoint: Path | None = None) -> Path:
    if checkpoint is not None:
        return checkpoint.resolve()
    latest_path = checkpoints_dir / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    resolved = checkpoints_dir / latest["directory"]
    if sha256_file(resolved / "manifest.json") != latest["manifest_sha256"]:
        raise ValueError("latest checkpoint manifest hash mismatch")
    return resolved


def load_full_checkpoint(
    checkpoint: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: TokenLRScheduler,
    sampler: StatefulDistributedSampler,
    compatibility: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION or manifest.get("stage") != "full_training_checkpoint":
        raise ValueError("unsupported checkpoint manifest")
    if manifest.get("compatibility") != compatibility:
        raise ValueError("checkpoint compatibility metadata changed")
    for name, expected in manifest["artifacts"].items():
        path = checkpoint / name
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"checkpoint artifact integrity failure: {name}")

    missing, unexpected = load_model(
        model, checkpoint / "model.safetensors", strict=True, device=str(device)
    )
    if missing or unexpected:
        raise ValueError(f"model checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    optimizer.load_state_dict(
        torch.load(
            checkpoint / "optimizer.pt",
            map_location=device,
            weights_only=False,
        )
    )
    scheduler.load_state_dict(
        json.loads((checkpoint / "scheduler.json").read_text(encoding="utf-8"))
    )
    sampler.load_state_dict(
        json.loads((checkpoint / "sampler.json").read_text(encoding="utf-8"))
    )
    runtime = torch.load(
        checkpoint / "runtime.pt", map_location="cpu", weights_only=False
    )
    torch.set_rng_state(runtime["torch_rng_state"])
    restore_kernel_state(runtime.get("kernel_autotune"))
    if runtime["cuda_rng_state"] is not None:
        if device.type != "cuda":
            raise ValueError("checkpoint contains CUDA RNG state but target device is not CUDA")
        torch.cuda.set_rng_state(runtime["cuda_rng_state"], device=device)
    if scheduler.tokens_seen != manifest["tokens_seen"]:
        raise ValueError("checkpoint scheduler token count mismatch")
    return manifest


def _distributed_context(rank: int, world_size: int) -> None:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed checkpointing requires an initialized process group")
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise ValueError("distributed checkpoint rank or world size mismatch")


def _rank_runtime_state(
    model: torch.nn.Module, loader_generator: torch.Generator
) -> dict[str, Any]:
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    return {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state(model_device)
            if model_device.type == "cuda"
            else None
        ),
        "loader_generator_state": loader_generator.get_state(),
        "kernel_autotune": capture_kernel_state(),
    }


def _restore_rank_runtime_state(
    runtime: dict[str, Any],
    *,
    loader_generator: torch.Generator,
    device: torch.device,
) -> None:
    torch.set_rng_state(runtime["torch_rng_state"])
    restore_kernel_state(runtime.get("kernel_autotune"))
    cuda_rng_state = runtime["cuda_rng_state"]
    if cuda_rng_state is not None:
        if device.type != "cuda":
            raise ValueError("checkpoint contains CUDA RNG state but target device is not CUDA")
        torch.cuda.set_rng_state(cuda_rng_state, device=device)
    loader_generator.set_state(runtime["loader_generator_state"])


def save_distributed_full_checkpoint(
    checkpoints_dir: Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: TokenLRScheduler,
    sampler: StatefulDistributedSampler,
    loader_generator: torch.Generator,
    compatibility: dict[str, Any],
    rank: int,
    world_size: int,
    keep: int = 2,
) -> tuple[Path, dict[str, Any]]:
    """Save one replicated DDP checkpoint with rank-local runtime state.

    Rank zero writes the shared model, optimizer and scheduler.  Every rank
    writes its own sampler and RNG state.  Collective error exchange keeps all
    ranks on the same control-flow path instead of leaving peers blocked at a
    barrier when one writer fails.
    """

    _distributed_context(rank, world_size)
    if step < 0 or keep < 1:
        raise ValueError("step must be non-negative and keep must be positive")
    if hasattr(sampler, "global_commit_fingerprint"):
        fingerprint = sampler.global_commit_fingerprint()
        fingerprints: list[Any] = [None] * world_size
        dist.all_gather_object(fingerprints, fingerprint)
        if any(item != fingerprint for item in fingerprints):
            raise ValueError("DDP ranks disagree on committed data position")

    final = checkpoints_dir / f"step_{step:08d}"
    initialization: list[Any] = [None, None]
    if rank == 0:
        try:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            if final.exists():
                raise FileExistsError(final)
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=checkpoints_dir)
            )
            initialization[0] = str(temporary)
        except BaseException as error:
            initialization[1] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(initialization, src=0)
    if initialization[1] is not None:
        raise RuntimeError(f"distributed checkpoint initialization failed: {initialization[1]}")
    temporary = Path(initialization[0])
    started = time.time()

    local_error: str | None = None
    try:
        torch.save(
            _rank_runtime_state(model, loader_generator),
            temporary / f"runtime_rank_{rank:05d}.pt",
        )
        atomic_json(
            temporary / f"sampler_rank_{rank:05d}.json", sampler.state_dict()
        )
    except BaseException as error:
        local_error = f"rank {rank}: {type(error).__name__}: {error}"
    rank_errors: list[str | None] = [None] * world_size
    dist.all_gather_object(rank_errors, local_error)
    failures = [error for error in rank_errors if error is not None]
    if failures:
        if rank == 0:
            shutil.rmtree(temporary, ignore_errors=True)
        raise RuntimeError("distributed rank-state save failed: " + "; ".join(failures))

    finalization: list[Any] = [None]
    if rank == 0:
        try:
            model_path = temporary / "model.safetensors"
            optimizer_path = temporary / "optimizer.pt"
            scheduler_path = temporary / "scheduler.json"
            save_model(
                model,
                str(model_path),
                metadata={"format": "pt", "checkpoint_version": DISTRIBUTED_VERSION},
            )
            torch.save(optimizer.state_dict(), optimizer_path)
            atomic_json(scheduler_path, scheduler.state_dict())
            artifact_paths = [model_path, optimizer_path, scheduler_path]
            artifact_paths.extend(
                temporary / f"runtime_rank_{current_rank:05d}.pt"
                for current_rank in range(world_size)
            )
            artifact_paths.extend(
                temporary / f"sampler_rank_{current_rank:05d}.json"
                for current_rank in range(world_size)
            )
            artifacts = {path.name: _artifact(path) for path in artifact_paths}
            manifest = {
                "version": DISTRIBUTED_VERSION,
                "stage": "distributed_full_training_checkpoint",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "step": step,
                "tokens_seen": scheduler.tokens_seen,
                "world_size": world_size,
                "elapsed_seconds": round(time.time() - started, 3),
                "compatibility": compatibility,
                "artifacts": artifacts,
            }
            atomic_json(temporary / "manifest.json", manifest)
            os.replace(temporary, final)
            atomic_json(
                checkpoints_dir / "latest.json",
                {
                    "version": DISTRIBUTED_VERSION,
                    "step": step,
                    "directory": final.name,
                    "manifest_sha256": sha256_file(final / "manifest.json"),
                },
            )
            completed = sorted(
                path
                for path in checkpoints_dir.glob("step_[0-9]*")
                if path.is_dir() and (path / "manifest.json").is_file()
            )
            for obsolete in completed[:-keep]:
                shutil.rmtree(obsolete)
        except BaseException as error:
            shutil.rmtree(temporary, ignore_errors=True)
            finalization[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(finalization, src=0)
    if finalization[0] is not None:
        raise RuntimeError(f"distributed checkpoint finalization failed: {finalization[0]}")
    manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    return final, manifest


def load_distributed_full_checkpoint(
    checkpoint: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: TokenLRScheduler,
    sampler: StatefulDistributedSampler,
    loader_generator: torch.Generator,
    compatibility: dict[str, Any],
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    """Restore shared DDP state plus the calling rank's sampler and RNG."""

    _distributed_context(rank, world_size)
    verification: list[Any] = [None, None]
    if rank == 0:
        try:
            manifest = json.loads(
                (checkpoint / "manifest.json").read_text(encoding="utf-8")
            )
            if (
                manifest.get("version") != DISTRIBUTED_VERSION
                or manifest.get("stage") != "distributed_full_training_checkpoint"
            ):
                raise ValueError("unsupported distributed checkpoint manifest")
            if manifest.get("world_size") != world_size:
                raise ValueError("distributed checkpoint world size changed")
            if manifest.get("compatibility") != compatibility:
                raise ValueError("checkpoint compatibility metadata changed")
            for name, expected in manifest["artifacts"].items():
                path = checkpoint / name
                if (
                    path.stat().st_size != expected["bytes"]
                    or sha256_file(path) != expected["sha256"]
                ):
                    raise ValueError(f"checkpoint artifact integrity failure: {name}")
            verification[0] = manifest
        except BaseException as error:
            verification[1] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(verification, src=0)
    if verification[1] is not None:
        raise RuntimeError(f"distributed checkpoint verification failed: {verification[1]}")
    manifest = verification[0]

    local_error: str | None = None
    try:
        missing, unexpected = load_model(
            model, checkpoint / "model.safetensors", strict=True, device=str(device)
        )
        if missing or unexpected:
            raise ValueError(
                f"model checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        optimizer.load_state_dict(
            torch.load(
                checkpoint / "optimizer.pt", map_location=device, weights_only=False
            )
        )
        scheduler.load_state_dict(
            json.loads((checkpoint / "scheduler.json").read_text(encoding="utf-8"))
        )
        sampler.load_state_dict(
            json.loads(
                (checkpoint / f"sampler_rank_{rank:05d}.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        runtime = torch.load(
            checkpoint / f"runtime_rank_{rank:05d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        _restore_rank_runtime_state(
            runtime, loader_generator=loader_generator, device=device
        )
        if scheduler.tokens_seen != manifest["tokens_seen"]:
            raise ValueError("checkpoint scheduler token count mismatch")
    except BaseException as error:
        local_error = f"rank {rank}: {type(error).__name__}: {error}"
    rank_errors: list[str | None] = [None] * world_size
    dist.all_gather_object(rank_errors, local_error)
    failures = [error for error in rank_errors if error is not None]
    if failures:
        raise RuntimeError("distributed checkpoint load failed: " + "; ".join(failures))
    if hasattr(sampler, "global_commit_fingerprint"):
        fingerprint = sampler.global_commit_fingerprint()
        fingerprints: list[Any] = [None] * world_size
        dist.all_gather_object(fingerprints, fingerprint)
        if any(item != fingerprint for item in fingerprints):
            raise ValueError("restored DDP ranks disagree on committed data position")
    return manifest
