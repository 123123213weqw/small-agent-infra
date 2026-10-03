"""Single-node pretraining trainer for one GPU or DDP."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import tempfile
import time
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_default_hooks
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from small_agent.data import (
    DeterministicMixtureSchedule,
    IndexedTokenDataset,
    MixtureDistributedSampler,
    ShardedTokenDataset,
    StatefulDistributedSampler,
)
from small_agent.evaluation import evaluate_causal_lm
from small_agent.models import build_model, load_model_spec
from small_agent.training.checkpoint import (
    load_distributed_full_checkpoint,
    load_full_checkpoint,
    resolve_checkpoint,
    save_distributed_full_checkpoint,
    save_full_checkpoint,
)
from small_agent.training.scheduler import TokenLRScheduler, WarmupCosineConfig
from small_agent.training.quality import (
    HealthWindow, LayerDiagnostics, ReferencePanel, domain_plan, enqueue_checkpoint,
    preserve_runtime, quick_validation, validate_optimized_adoption,
)


VERSION = "small_agent_trainer_m8_quality_v2"


def distributed_context() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed environment")
    if world_size > 1:
        dist.init_process_group("nccl")
    return world_size > 1, rank, local_rank, world_size


def reduce_sum_int(value: int, device: torch.device) -> int:
    tensor = torch.tensor(value, device=device, dtype=torch.int64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor)


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= dist.get_world_size()
    return result


def reduce_max_float(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_source_tree(root: Path) -> str:
    """Hash executable Python sources even when the remote snapshot has no .git."""
    digest = hashlib.sha256()
    source_root = root / "src" / "small_agent"
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(root).as_posix().encode("utf-8")
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


def find_repository_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "src" / "small_agent").is_dir() and (candidate / "configs").is_dir():
            return candidate
    raise ValueError(f"cannot locate repository root from {config_path}")


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def source_revision(root: Path) -> str:
    explicit = os.environ.get("SMALL_AGENT_SOURCE_COMMIT")
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def infinite_train_batches(
    dataset: IndexedTokenDataset,
    sampler: StatefulDistributedSampler,
    batch_size: int,
    loader_generator: torch.Generator,
) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            generator=loader_generator,
        )
        yielded = False
        generator_state = loader_generator.get_state()
        iterator = iter(loader)
        # Token datasets are stateless, with no worker augmentation. A new
        # iterator on resume must not consume the saved loader RNG a second time.
        loader_generator.set_state(generator_state)
        for batch in iterator:
            yielded = True
            yield batch
        if not yielded and sampler.cursor == 0:
            raise ValueError("no full training batch can be formed")
        sampler.set_epoch(sampler.epoch + 1)


def sharded_train_batches(
    dataset: ShardedTokenDataset,
    sampler: MixtureDistributedSampler,
    batch_size: int,
    loader_generator: torch.Generator,
    pipeline_config: dict[str, Any],
) -> Iterator[dict[str, torch.Tensor]]:
    """An infinite sampler; constructing the loader lazily permits checkpoint restore first."""
    workers = int(pipeline_config.get("num_workers", 0))
    if workers < 0:
        raise ValueError("data_pipeline.num_workers must be nonnegative")
    prefetch_factor = int(pipeline_config.get("prefetch_factor", 2))
    if workers and prefetch_factor < 1:
        raise ValueError("data_pipeline.prefetch_factor must be positive")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        prefetch_factor=prefetch_factor if workers else None,
        persistent_workers=bool(pipeline_config.get("persistent_workers", True)) if workers else False,
        pin_memory=bool(pipeline_config.get("pin_memory", True)),
        generator=loader_generator,
    )
    generator_state = loader_generator.get_state()
    iterator = iter(loader)
    loader_generator.set_state(generator_state)
    yield from iterator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-final-checkpoint", action="store_true")
    args = parser.parse_args()
    distributed, rank, local_rank, world_size = distributed_context()
    is_primary = rank == 0
    torch.cuda.set_device(local_rank)

    config_path = args.run_config.resolve()
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    repository_root = find_repository_root(config_path)
    model_spec_path = resolve_path(repository_root, run_config["model_spec"])
    validate_optimized_adoption(dict(run_config, model_spec=str(model_spec_path)))
    data_dir = resolve_path(repository_root, run_config["data_dir"])
    output_root = resolve_path(repository_root, run_config["output_root"])
    checkpoint_interval = int(run_config.get("checkpoint", {}).get("interval_steps", 0))
    checkpoint_steps = {int(value) for value in run_config.get("checkpoint", {}).get("steps", [])}
    if checkpoint_interval < 0 or any(value < 1 for value in checkpoint_steps):
        raise ValueError("checkpoint interval and explicit steps must be non-negative")
    run_dir = output_root / args.run_id
    if is_primary:
        if args.resume:
            if not run_dir.is_dir():
                raise FileNotFoundError(run_dir)
        else:
            if run_dir.exists():
                raise FileExistsError(run_dir)
            run_dir.mkdir(parents=True)
    if distributed:
        dist.barrier(device_ids=[local_rank])
    metrics_path = run_dir / "metrics.jsonl"
    status_path = run_dir / "status.json"
    writer = SummaryWriter(log_dir=run_dir / "tensorboard") if is_primary else None
    profiler: torch.profiler.profile | None = None

    max_steps = args.max_steps or int(run_config["max_steps"])
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    context_length = int(run_config["context_length"])
    micro_batch_size = int(run_config["micro_batch_size"])
    accumulation_steps = int(run_config["gradient_accumulation_steps"])
    data_pipeline_config = run_config.get("data_pipeline", {})
    measure_data_pipeline = bool(data_pipeline_config.get("measure", False))
    trace_sample_refs = bool(data_pipeline_config.get("trace_sample_refs", False))
    seed = int(run_config["seed"])
    if micro_batch_size < 1 or accumulation_steps < 1:
        raise ValueError("batch size and accumulation must be positive")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", local_rank)
    quality_config = run_config.get("training_quality", {})
    quality_enabled = bool(quality_config.get("enabled", False))
    if quality_enabled:
        if int(quality_config.get("fast_interval_steps", 500)) < 1 or int(quality_config.get("ability_interval_steps", 2000)) < 1:
            raise ValueError("positive training-quality intervals required")
        if not (args.save_final_checkpoint or max_steps in checkpoint_steps or (checkpoint_interval and max_steps % checkpoint_interval == 0)):
            raise ValueError("training quality requires a committed final checkpoint; pass --save-final-checkpoint")
    layer_diagnostics = None

    try:
        reader = run_config.get("data_reader", "indexed_v1")
        if reader == "sharded_v1":
            train_dataset = ShardedTokenDataset(data_dir, "train", context_length)
            validation_dataset = ShardedTokenDataset(data_dir, "validation", context_length)
            schedule = DeterministicMixtureSchedule(
                train_dataset,
                seed=seed,
                block_size=int(run_config.get("mixture_block_size", 1000)),
            )
            train_sampler = MixtureDistributedSampler(
                schedule,
                num_replicas=world_size,
                rank=rank,
                micro_batch_size=micro_batch_size,
                gradient_accumulation_steps=accumulation_steps,
            )
        elif reader == "indexed_v1":
            train_dataset = IndexedTokenDataset(
                data_dir, "train", context_length, tail_policy="drop"
            )
            validation_dataset = IndexedTokenDataset(
                data_dir, "validation", context_length, tail_policy="pad"
            )
            train_sampler = StatefulDistributedSampler(
                len(train_dataset),
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                shuffle=True,
                drop_last=True,
                samples_per_rank_multiple=micro_batch_size * accumulation_steps,
            )
        else:
            raise ValueError(f"unsupported data reader: {reader}")
        train_loader_generator = torch.Generator().manual_seed(seed + 10_000 + rank)
        if reader == "sharded_v1":
            train_batches = sharded_train_batches(
                train_dataset,
                train_sampler,
                micro_batch_size,
                train_loader_generator,
                data_pipeline_config,
            )
        else:
            train_batches = infinite_train_batches(
                train_dataset, train_sampler, micro_batch_size, train_loader_generator
            )
        validation_subset = Subset(
            validation_dataset, range(rank, len(validation_dataset), world_size)
        )
        validation_loader = DataLoader(
            validation_subset,
            batch_size=int(run_config["validation"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            generator=torch.Generator().manual_seed(seed + 20_000 + rank),
        )

        model_spec = load_model_spec(model_spec_path)
        model = build_model(model_spec).to(device)
        if run_config.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.config.use_cache = False
        model.train()
        layer_diagnostics = LayerDiagnostics(
            model, enabled=quality_enabled and is_primary and bool(quality_config.get("layer_diagnostics", False)),
            interval=int(quality_config.get("layer_interval_steps", 200)),
        )
        quick_plan = domain_plan(validation_dataset, int(quality_config.get("quick_max_samples_per_domain", 128))) if quality_enabled else None
        reference_panel = None
        reference_plan = None
        if quality_enabled and quality_config.get("reference_panel"):
            reference_panel = ReferencePanel(resolve_path(repository_root, quality_config["reference_panel"]),
                validation_dataset.manifest["tokenizer_sha256"], sha256_file(data_dir / "manifest.json"))
            reference_plan = domain_plan(reference_panel, int(quality_config.get("quick_reference_samples_per_domain", 128)))
        if quality_enabled and is_primary:
            atomic_json(run_dir / "quick_validation_plan.json", {
                "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
                "selection": "deterministic_evenly_spaced_per_raw_domain",
                "refs": quick_plan,
            })
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        training_model: torch.nn.Module = model
        distributed_config = run_config.get("distributed", {})
        communication_hook = distributed_config.get("communication_hook", "none")
        if communication_hook not in ("none", "bf16"):
            raise ValueError(f"unsupported DDP communication hook: {communication_hook}")
        if distributed:
            training_model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                bucket_cap_mb=float(distributed_config.get("bucket_cap_mb", 25)),
            )
            if communication_hook == "bf16":
                training_model.register_comm_hook(
                    dist.group.WORLD, ddp_default_hooks.bf16_compress_hook
                )
            # Keep initialization identical, then decorrelate stochastic layers.
            torch.manual_seed(seed + rank)
            torch.cuda.manual_seed_all(seed + rank)

        optimizer_config = run_config["optimizer"]
        optimizer = torch.optim.AdamW(
            training_model.parameters(),
            lr=float(optimizer_config["maximum_learning_rate"]),
            betas=tuple(optimizer_config["betas"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            fused=True,
        )
        scheduler_config = WarmupCosineConfig(
            maximum_learning_rate=float(optimizer_config["maximum_learning_rate"]),
            minimum_learning_rate=float(optimizer_config["minimum_learning_rate"]),
            warmup_tokens=int(run_config["scheduler"]["warmup_tokens"]),
            total_tokens=int(run_config["scheduler"]["total_tokens"]),
        )
        scheduler = TokenLRScheduler(optimizer, scheduler_config)

        compatibility = {
            "run_config_sha256": sha256_file(config_path),
            "model_spec_sha256": sha256_file(model_spec_path),
            "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
            "seed": seed,
            "context_length": context_length,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "gradient_checkpointing": bool(run_config.get("gradient_checkpointing", False)),
        }
        if reader == "sharded_v1":
            compatibility["data_reader"] = reader
            compatibility["mixture_block_size"] = schedule.block_size
        if quality_enabled:
            compatibility["source_tree_sha256"] = sha256_source_tree(repository_root)
        if distributed:
            compatibility.update(
                {
                    "distributed": True,
                    "world_size": world_size,
                    "ddp_bucket_cap_mb": float(distributed_config.get("bucket_cap_mb", 25)),
                    "ddp_communication_hook": communication_hook,
                }
            )

        manifest = {
            "version": VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "purpose": run_config["purpose"],
            "distributable": bool(run_config["distributable"]),
            "source_revision": source_revision(repository_root),
            "source_tree_sha256": sha256_source_tree(repository_root),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "parameter_count": parameter_count,
            "distributed": distributed,
            "world_size": world_size,
            "ddp_bucket_cap_mb": float(distributed_config.get("bucket_cap_mb", 25)),
            "ddp_communication_hook": communication_hook,
            "global_sequences_per_step": micro_batch_size * accumulation_steps * world_size,
            "max_steps": max_steps,
            "run_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "model_spec": {
                "path": str(model_spec_path),
                "sha256": sha256_file(model_spec_path),
            },
            "data_manifest": {
                "path": str(data_dir / "manifest.json"),
                "sha256": sha256_file(data_dir / "manifest.json"),
            },
            "train_coverage": train_dataset.coverage_report(),
            "train_sampler": (
                {
                    "type": "sharded_mixture_v1",
                    "mixture": train_dataset.manifest["mixture"],
                    "mixture_block_size": schedule.block_size,
                    "mixture_counts": schedule.counts,
                    "global_samples_per_optimizer_step": train_sampler.global_step_samples,
                    "cursor_semantics": "committed_global_position_only",
                }
                if reader == "sharded_v1"
                else {
                    "type": "stateful_distributed_v2",
                    "samples_per_rank": train_sampler.num_samples,
                    "global_samples_per_epoch": train_sampler.total_size,
                    "dropped_samples_per_epoch": len(train_dataset) - train_sampler.total_size,
                    "samples_per_rank_multiple": train_sampler.samples_per_rank_multiple,
                }
            ),
            "data_pipeline": data_pipeline_config if reader == "sharded_v1" else {},
            "training_quality": quality_config,
            "validation_coverage": validation_dataset.coverage_report(),
        }
        start_step = 0
        if args.resume:
            existing_manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            for field in ("run_config", "model_spec", "data_manifest"):
                if existing_manifest[field]["sha256"] != manifest[field]["sha256"]:
                    raise ValueError(f"run manifest changed across resume: {field}")
            checkpoint_path = resolve_checkpoint(run_dir / "checkpoints")
            if distributed:
                checkpoint_manifest = load_distributed_full_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=train_sampler,
                    loader_generator=train_loader_generator,
                    compatibility=compatibility,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                )
            else:
                checkpoint_manifest = load_full_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=train_sampler,
                    compatibility=compatibility,
                    device=device,
                )
            start_step = int(checkpoint_manifest["step"])
            if max_steps <= start_step:
                raise ValueError("max_steps must be greater than the resumed step")
            if is_primary:
                append_jsonl(
                    metrics_path,
                    {
                        "event": "resume",
                        "step": start_step,
                        "tokens_seen": scheduler.tokens_seen,
                        "checkpoint": checkpoint_path.name,
                    },
                )
        else:
            if is_primary:
                atomic_json(run_dir / "run_manifest.json", manifest)
                append_jsonl(metrics_path, {"event": "start", **manifest})
        if is_primary:
            atomic_json(
                status_path,
                {"state": "running", "step": start_step, "tokens_seen": scheduler.tokens_seen},
            )

        if not args.resume and run_config["validation"].get("at_start", True):
            with preserve_runtime(model, (train_loader_generator,)):
                validation = evaluate_causal_lm(model, validation_loader, device)
            row = {
                "event": "validation",
                "step": 0,
                "tokens_seen": 0,
                **validation.__dict__,
            }
            if is_primary:
                append_jsonl(metrics_path, row)
                writer.add_scalar("validation/loss", validation.loss, 0)
                writer.add_scalar("validation/perplexity", validation.perplexity, 0)
                print(json.dumps(row), flush=True)

        logging_interval = int(run_config["logging_interval_steps"])
        validation_interval = int(run_config["validation"]["interval_steps"])
        profiler_config = run_config.get("profiler", {})
        if is_primary and profiler_config.get("enabled", False):
            profiler_dir = run_dir / "profiler"
            profiler_dir.mkdir(parents=True, exist_ok=True)
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=int(profiler_config.get("wait_steps", 2)),
                    warmup=int(profiler_config.get("warmup_steps", 2)),
                    active=int(profiler_config.get("active_steps", 3)),
                    repeat=1,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profiler_dir)),
                record_shapes=bool(profiler_config.get("record_shapes", False)),
                profile_memory=bool(profiler_config.get("profile_memory", True)),
                with_stack=bool(profiler_config.get("with_stack", False)),
            )
            profiler.start()
        window_loss = torch.zeros((), device=device)
        health = HealthWindow(device, float(optimizer_config["gradient_clip"])) if quality_enabled else None
        window_tokens = 0
        window_steps = 0
        window_data_wait_seconds = 0.0
        window_h2d_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        window_sample_refs: list[int] = []
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        window_started = time.perf_counter()
        window_elapsed = 0.0

        for step in range(start_step + 1, max_steps + 1):
            step_started = time.perf_counter()
            layer_diagnostics.begin(step)
            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device)
            step_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
            step_tokens = 0
            for microstep in range(accumulation_steps):
                data_wait_started = time.perf_counter()
                batch = next(train_batches)
                if measure_data_pipeline:
                    window_data_wait_seconds += time.perf_counter() - data_wait_started
                if trace_sample_refs and reader == "sharded_v1":
                    window_sample_refs.extend(int(value) for value in batch["sample_ref"].tolist())
                predicted_tokens = int(
                    torch.count_nonzero(batch["labels"][:, 1:] != -100)
                )
                if measure_data_pipeline:
                    h2d_started = torch.cuda.Event(enable_timing=True)
                    h2d_finished = torch.cuda.Event(enable_timing=True)
                    h2d_started.record()
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                if measure_data_pipeline:
                    h2d_finished.record()
                    window_h2d_events.append((h2d_started, h2d_finished))
                should_sync = not distributed or microstep == accumulation_steps - 1
                synchronization = (
                    nullcontext() if should_sync else training_model.no_sync()
                )
                with synchronization:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = training_model(
                            input_ids=input_ids,
                            labels=labels,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                        loss = output.loss
                    if microstep == 0:
                        layer_diagnostics.close()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite training loss at step {step}")
                    (loss / accumulation_steps).backward()
                    step_loss += loss.detach() / accumulation_steps
                    if quality_enabled:
                        step_loss_sum += loss.detach().double() * predicted_tokens
                    step_tokens += predicted_tokens

            layer_values = layer_diagnostics.gradients()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                training_model.parameters(),
                float(optimizer_config["gradient_clip"]),
                error_if_nonfinite=True,
            )
            global_step_tokens = reduce_sum_int(step_tokens, device)
            learning_rate = scheduler.step(global_step_tokens)
            optimizer.step()
            if health is not None:
                health.add(step_loss_sum, step_tokens, grad_norm, time.perf_counter() - step_started)
            if is_primary and layer_values:
                for tag, value in layer_values.items():
                    writer.add_scalar(tag, value, step)
                append_jsonl(metrics_path, {"event": "layer_diagnostics", "step": step, "sampled_rank": 0, "values": layer_values})
            if reader == "sharded_v1":
                train_sampler.commit_step()
            if profiler is not None:
                profiler.step()
            window_loss += step_loss
            window_tokens += global_step_tokens
            window_steps += 1

            should_log = step % logging_interval == 0 or step == max_steps
            if should_log:
                torch.cuda.synchronize(device)
                elapsed = reduce_max_float(
                    window_elapsed + time.perf_counter() - window_started, device
                )
                mean_loss = reduce_mean(window_loss / window_steps)
                maximum_grad_norm = reduce_max_float(float(grad_norm), device)
                maximum_allocated = reduce_max_float(
                    torch.cuda.max_memory_allocated(device) / 2**30, device
                )
                maximum_reserved = reduce_max_float(
                    torch.cuda.max_memory_reserved(device) / 2**30, device
                )
                row = {
                    "event": "train",
                    "step": step,
                    "epoch": train_sampler.epoch,
                    "sampler_cursor": train_sampler.cursor,
                    "tokens_seen": scheduler.tokens_seen,
                    "loss": float(mean_loss),
                    "learning_rate": learning_rate,
                    "grad_norm": maximum_grad_norm,
                    "tokens_per_second": window_tokens / elapsed,
                    "peak_allocated_gib": maximum_allocated,
                    "peak_reserved_gib": maximum_reserved,
                }
                if health is not None:
                    row.update(health.report())
                if measure_data_pipeline:
                    max_wait = reduce_max_float(window_data_wait_seconds, device)
                    local_h2d_ms = sum(
                        begin.elapsed_time(end) for begin, end in window_h2d_events
                    )
                    max_h2d_ms = reduce_max_float(local_h2d_ms, device)
                    row.update(
                        {
                            "data_wait_ms": max_wait * 1000,
                            "h2d_copy_ms": max_h2d_ms,
                            "data_wait_fraction": max_wait / max(elapsed, 1e-9),
                            "h2d_copy_fraction": (max_h2d_ms / 1000) / max(elapsed, 1e-9),
                        }
                    )
                if trace_sample_refs and reader == "sharded_v1":
                    row["sample_refs_rank0"] = window_sample_refs.copy() if is_primary else []
                if is_primary:
                    append_jsonl(metrics_path, row)
                    print(json.dumps(row), flush=True)
                    for key in ("loss", "learning_rate", "grad_norm", "tokens_per_second"):
                        writer.add_scalar(f"train/{key}", row[key], step)
                    writer.add_scalar("train/tokens_seen", scheduler.tokens_seen, step)
                    writer.add_scalar("system/gpu_allocated_gib", row["peak_allocated_gib"], step)
                    writer.add_scalar("system/gpu_reserved_gib", row["peak_reserved_gib"], step)
                    if health is not None:
                        for key in health.report_keys:
                            writer.add_scalar(f"train/{key}", row[key], step)
                    if measure_data_pipeline:
                        writer.add_scalar("data/wait_ms", row["data_wait_ms"], step)
                        writer.add_scalar("data/h2d_copy_ms", row["h2d_copy_ms"], step)
                        writer.add_scalar("data/wait_fraction", row["data_wait_fraction"], step)
                    writer.flush()
                    atomic_json(
                        status_path,
                        {"state": "running", "step": step, "tokens_seen": scheduler.tokens_seen},
                    )
                window_loss.zero_()
                if health is not None:
                    health.reset()
                window_tokens = 0
                window_steps = 0
                window_data_wait_seconds = 0.0
                window_h2d_events.clear()
                window_sample_refs.clear()
                window_started = time.perf_counter()
                window_elapsed = 0.0

            if (checkpoint_interval and step % checkpoint_interval == 0) or step in checkpoint_steps:
                if window_steps:
                    torch.cuda.synchronize(device)
                    window_elapsed += time.perf_counter() - window_started
                if distributed:
                    checkpoint_path, checkpoint_manifest = save_distributed_full_checkpoint(
                        run_dir / "checkpoints",
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        sampler=train_sampler,
                        loader_generator=train_loader_generator,
                        compatibility=compatibility,
                        rank=rank,
                        world_size=world_size,
                        keep=int(run_config.get("checkpoint", {}).get("keep_full", 2)),
                    )
                else:
                    checkpoint_path, checkpoint_manifest = save_full_checkpoint(
                        run_dir / "checkpoints",
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        sampler=train_sampler,
                        compatibility=compatibility,
                        keep=int(run_config.get("checkpoint", {}).get("keep_full", 2)),
                    )
                if is_primary:
                    append_jsonl(
                        metrics_path,
                        {
                            "event": "checkpoint",
                            "step": step,
                            "tokens_seen": scheduler.tokens_seen,
                            "directory": checkpoint_path.name,
                            "elapsed_seconds": checkpoint_manifest["elapsed_seconds"],
                        },
                    )
                    if quality_enabled:
                        enqueue_checkpoint(run_dir, checkpoint_path, step, step == max_steps, quality_config)
                torch.cuda.synchronize(device)
                window_started = time.perf_counter()

            if quality_enabled and step % int(quality_config.get("fast_interval_steps", 500)) == 0:
                if window_steps:
                    torch.cuda.synchronize(device)
                    window_elapsed += time.perf_counter() - window_started
                quick = quick_validation(model, validation_dataset, quick_plan, device, rank, world_size,
                                         int(run_config["validation"]["batch_size"]), (train_loader_generator,))
                if is_primary:
                    append_jsonl(metrics_path, {"event": "validation_quick", "step": step,
                        "tokens_seen": scheduler.tokens_seen, **quick})
                    for domain, values in {"overall": quick["overall"], **quick["domains"]}.items():
                        for key in ("loss", "ppl", "predicted_tokens"):
                            writer.add_scalar(f"validation/quick/{domain}/{key}", values[key], step)
                    writer.add_scalar("validation/quick/seconds", quick["seconds"], step)
                    writer.flush()
                if reference_panel is not None:
                    reference = quick_validation(model, reference_panel, reference_plan, device, rank, world_size, 1, (train_loader_generator,))
                    if is_primary:
                        append_jsonl(metrics_path, {"event": "validation_reference", "step": step, **reference,
                            "scope": "external_reference_NLL_not_original_corpus_PPL"})
                        for domain, values in reference["domains"].items():
                            for key in ("loss", "ppl", "predicted_tokens", "samples"):
                                writer.add_scalar(f"validation/reference/{domain}/{key}", values[key], step)
                        writer.flush()
                torch.cuda.synchronize(device)
                window_started = time.perf_counter()

            if validation_interval > 0 and step % validation_interval == 0:
                if window_steps:
                    torch.cuda.synchronize(device)
                    window_elapsed += time.perf_counter() - window_started
                validation_started = time.monotonic()
                with preserve_runtime(model, (train_loader_generator,)):
                    validation = evaluate_causal_lm(model, validation_loader, device)
                row = {
                    "event": "validation",
                    "step": step,
                    "tokens_seen": scheduler.tokens_seen,
                    **validation.__dict__,
                    "seconds": time.monotonic() - validation_started,
                }
                if is_primary:
                    append_jsonl(metrics_path, row)
                    print(json.dumps(row), flush=True)
                    writer.add_scalar("validation/loss", validation.loss, step)
                    writer.add_scalar("validation/perplexity", validation.perplexity, step)
                    writer.add_scalar("validation/seconds", row["seconds"], step)
                    writer.flush()
                torch.cuda.synchronize(device)
                window_started = time.perf_counter()

        final_already_checkpointed = bool(
            (checkpoint_interval and max_steps % checkpoint_interval == 0)
            or max_steps in checkpoint_steps
        )
        if args.save_final_checkpoint and not final_already_checkpointed:
            if distributed:
                checkpoint_path, checkpoint_manifest = save_distributed_full_checkpoint(
                    run_dir / "checkpoints",
                    step=max_steps,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=train_sampler,
                    loader_generator=train_loader_generator,
                    compatibility=compatibility,
                    rank=rank,
                    world_size=world_size,
                    keep=int(run_config.get("checkpoint", {}).get("keep_full", 2)),
                )
            else:
                checkpoint_path, checkpoint_manifest = save_full_checkpoint(
                    run_dir / "checkpoints",
                    step=max_steps,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=train_sampler,
                    compatibility=compatibility,
                    keep=int(run_config.get("checkpoint", {}).get("keep_full", 2)),
                )
            if is_primary:
                append_jsonl(
                    metrics_path,
                    {
                        "event": "checkpoint",
                        "step": max_steps,
                        "tokens_seen": scheduler.tokens_seen,
                        "directory": checkpoint_path.name,
                        "elapsed_seconds": checkpoint_manifest["elapsed_seconds"],
                    },
                )

        if quality_enabled and is_primary:
            enqueue_checkpoint(run_dir, resolve_checkpoint(run_dir / "checkpoints"), max_steps, True, quality_config)
        with preserve_runtime(model, (train_loader_generator,)):
            validation = evaluate_causal_lm(model, validation_loader, device)
        row = {
            "event": "validation",
            "step": max_steps,
            "tokens_seen": scheduler.tokens_seen,
            **validation.__dict__,
        }
        if is_primary:
            append_jsonl(metrics_path, row)
            print(json.dumps(row), flush=True)
            writer.add_scalar("validation/loss", validation.loss, max_steps)
            writer.add_scalar("validation/perplexity", validation.perplexity, max_steps)
            writer.flush()

            atomic_json(
                status_path,
                {"state": "completed", "step": max_steps, "tokens_seen": scheduler.tokens_seen},
            )
        if profiler is not None:
            profiler.stop()
            (run_dir / "profiler_operator_table.txt").write_text(
                profiler.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=80
                )
                + "\n",
                encoding="utf-8",
            )
            profiler = None
    except BaseException as error:
        if is_primary:
            atomic_json(
                status_path,
                {"state": "failed", "error_type": type(error).__name__, "error": str(error)},
            )
        raise
    finally:
        if layer_diagnostics is not None:
            layer_diagnostics.close()
        if profiler is not None:
            profiler.stop()
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
