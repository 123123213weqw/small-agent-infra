#!/usr/bin/env python3
"""Explicit replicated-DDP 6→8 checkpoint transition, not a trainer rewrite.

Keep actual Adam update step 200 and token/global-data cursors. Shared state and
six existing rank RNG files are hardlinked without edits; two new ranks receive
documented seeded runtime state. This is not a bitwise-equivalent six-card run.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile

ROOT = Path("/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1")
SOURCE = Path("/srv/small-agent/experiments/small-agent-base-1b-wave2-reuse-v1")
SOURCE_RUN_ID = "prior-wave2-reuse-1040m-6l40-v1"
RUN_ID = "prior-wave2-full-1040m-8l40-v1"
CONFIG = "base_1b_prior_wave2_internal_full_8l40_v1.json"
TRAINER_SHA = "e418e6c69bcf2f6724a8223325190a89f6799a72d75d6a9e0f0d1b283d23af41"
CHECKPOINT_SHA = "faa973e047ca7dbadfee02d8dfa5184bfd88f3b84cb8aa5abea240cee23fb716"
DATA_SHA = "50c599d0e81ebfdf5e8955309c039940546b1549d0d48dd688d35a657d7be9bd"
MODEL_SHA = "51e728d13e95cf111825416b344092171d5e0cc266ab66c55b501bfd76753b1e"
TOKENIZER_SHA = "9d0b99ab9cf20d337217c96a5b39e9ff87590eb462abcbcc8f1119c3d43911a7"


def read(p: Path) -> dict:
    return json.loads(p.read_text())


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(p: Path, value: dict) -> None:
    if p.exists():
        raise FileExistsError(p)
    with p.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def transition_budget(samples: int, source_step: int, position: int, tokens: int,
                      world: int, micro: int, accumulation: int, length: int) -> dict:
    if min(samples, world, micro, accumulation) < 1 or length < 2 or source_step < 0:
        raise ValueError("invalid training budget")
    batch = world * micro * accumulation
    if not 0 <= position < samples or position % batch or tokens != position * (length - 1):
        raise ValueError("unaligned or inconsistent committed token/sample cursor")
    remaining_steps = math.ceil((samples - position) / batch)
    end_position = position + remaining_steps * batch
    return {"source_step": source_step, "source_tokens": tokens, "source_position": position,
            "world_size": world, "global_sequences_per_step": batch,
            "loss_tokens_per_step": batch * (length - 1), "remaining_steps": remaining_steps,
            "end_step": source_step + remaining_steps, "end_position": end_position,
            "total_loss_tokens": end_position * (length - 1),
            "tail_repeated_samples": end_position - samples,
            "unique_train_samples": samples}


def validate_config(old: dict, new: dict, b: dict) -> None:
    allowed = {"version", "purpose", "output_root", "max_steps", "checkpoint", "data_pipeline"}
    if set(old) != set(new) or any(old[k] != new[k] for k in old if k not in allowed):
        raise ValueError("transition changed model/data/tokenizer/optimizer/scheduler semantics")
    old_pipeline, new_pipeline = copy.deepcopy(old["data_pipeline"]), copy.deepcopy(new["data_pipeline"])
    old_pipeline.pop("trace_sample_refs", None)
    new_pipeline.pop("trace_sample_refs", None)
    if old_pipeline != new_pipeline or new["max_steps"] != b["end_step"]:
        raise ValueError("pipeline or remaining budget changed")
    if new["output_root"] != str(ROOT / "runs") or new["distributable"]:
        raise ValueError("wrong output root or scope")


def check_artifacts(path: Path, m: dict) -> None:
    for name, item in m["artifacts"].items():
        if Path(name).name != name or (path / name).is_symlink():
            raise ValueError("unsafe checkpoint artifact path")
        if (path / name).stat().st_size != item["bytes"] or sha(path / name) != item["sha256"]:
            raise ValueError(f"checkpoint integrity failure: {name}")


def artifacts(path: Path) -> dict:
    return {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(path.iterdir())}


def full_summary(events: list[dict], b: dict) -> dict:
    train = [e for e in events if e["event"] == "train"]
    val = [e for e in events if e["event"] == "validation"]
    if not train or train[-1]["step"] != b["end_step"] or train[-1]["tokens_seen"] != b["total_loss_tokens"]:
        raise ValueError("full budget not completed")
    if not any(e["event"] == "resume" and e["step"] == b["source_step"] and
               e["tokens_seen"] == b["source_tokens"] for e in events):
        raise ValueError("missing real migration restore event")
    if not val or val[-1]["step"] != b["end_step"]:
        raise ValueError("missing final validation")
    for e in train:
        expected = b["source_tokens"] + (e["step"] - b["source_step"]) * b["loss_tokens_per_step"]
        if e["step"] <= b["source_step"] or e["tokens_seen"] != expected:
            raise ValueError("token counter was reset or step accounting changed")
        if not all(math.isfinite(float(e[k])) for k in ("loss", "grad_norm", "learning_rate", "tokens_per_second")) or e["tokens_per_second"] <= 0:
            raise ValueError("non-finite or invalid training metric")
    if any(not math.isfinite(e["loss"]) for e in val):
        raise ValueError("non-finite validation loss")
    return {"last_training": train[-1], "final_validation": val[-1],
            "median_last_10_windows_loss_tokens_per_second": statistics.median(e["tokens_per_second"] for e in train[-10:])}


def verify_bindings() -> tuple[dict, Path]:
    sys.path.insert(0, str(ROOT / "code/scripts"))
    import trainctl
    if trainctl.sha256_source_tree(ROOT / "code") != TRAINER_SHA:
        raise ValueError("validated backend changed")
    cfg_path = ROOT / "code/configs/runs" / CONFIG
    cfg = read(cfg_path)
    if sha(Path(cfg["data_dir"]) / "manifest.json") != DATA_SHA or sha(ROOT / "code" / cfg["model_spec"]) != MODEL_SHA:
        raise ValueError("model/data binding changed")
    if sha(Path(cfg["tokenizer_json"])) != TOKENIZER_SHA:
        raise ValueError("tokenizer binding changed")
    return cfg, cfg_path


def prepare() -> None:
    import torch
    from small_agent.data import ShardedTokenDataset
    from small_agent.data.mixture_sampler import DeterministicMixtureSchedule, MixtureDistributedSampler
    from small_agent.training.checkpoint import _restore_rank_runtime_state
    from small_agent.training.trainer import sha256_source_tree
    cfg, cfg_path = verify_bindings()
    old_run = SOURCE / "runs" / SOURCE_RUN_ID
    checkpoint = old_run / "checkpoints/step_00000200"
    if sha(checkpoint / "manifest.json") != CHECKPOINT_SHA:
        raise ValueError("source checkpoint identity changed")
    old_m = read(checkpoint / "manifest.json")
    old_cfg_path = SOURCE / "code/configs/runs/base_1b_prior_wave2_internal_1040m_6l40_v1.json"
    old_cfg = read(old_cfg_path)
    if sha(old_cfg_path) != old_m["compatibility"]["run_config_sha256"]:
        raise ValueError("source run config changed")
    trial = read(SOURCE / "reports/TRIAL200.json")
    prepared = read(SOURCE / "reports/PREPARED.json")
    if trial["status"] != "trial200_passed_with_actual_checkpoint_resume" or prepared["data_manifest_sha256"] != DATA_SHA:
        raise ValueError("source trial not accepted")
    if (old_m["world_size"], old_m["step"], old_m["tokens_seen"]) != (6, 200, 9825600):
        raise ValueError("unexpected source checkpoint progress")
    if old_m["compatibility"]["model_spec_sha256"] != MODEL_SHA or old_m["compatibility"]["data_manifest_sha256"] != DATA_SHA:
        raise ValueError("source checkpoint model/data changed")
    check_artifacts(checkpoint, old_m)
    print("SOURCE_ALL_ARTIFACT_HASHES_VERIFIED", flush=True)
    scheduler = read(checkpoint / "scheduler.json")
    expected_scheduler = {"maximum_learning_rate": cfg["optimizer"]["maximum_learning_rate"],
                          "minimum_learning_rate": cfg["optimizer"]["minimum_learning_rate"],
                          "warmup_tokens": cfg["scheduler"]["warmup_tokens"],
                          "total_tokens": cfg["scheduler"]["total_tokens"]}
    if scheduler["tokens_seen"] != old_m["tokens_seen"] or scheduler["config"] != expected_scheduler:
        raise ValueError("LR schedule reset/changed")
    dataset = ShardedTokenDataset(Path(cfg["data_dir"]), "train", cfg["context_length"], verify_hashes=True)
    schedule = DeterministicMixtureSchedule(dataset, seed=cfg["seed"], block_size=cfg["mixture_block_size"])
    position = read(checkpoint / "sampler_rank_00000.json")["committed_global_position"]
    b = transition_budget(len(dataset), 200, position, old_m["tokens_seen"], 8,
                          cfg["micro_batch_size"], cfg["gradient_accumulation_steps"], cfg["context_length"])
    validate_config(old_cfg, cfg, b)
    for rank in range(6):
        old_sampler = MixtureDistributedSampler(schedule, num_replicas=6, rank=rank,
                        micro_batch_size=cfg["micro_batch_size"], gradient_accumulation_steps=cfg["gradient_accumulation_steps"])
        state = read(checkpoint / f"sampler_rank_{rank:05d}.json")
        old_sampler.load_state_dict(state)
        if state["committed_global_position"] != position:
            raise ValueError("source ranks disagree on committed data cursor")
    refs = {schedule.sample_ref_at(i) for i in range(len(dataset))}
    if len(refs) != len(dataset) or sum(schedule.counts.values()) != len(dataset):
        raise ValueError("training schedule is not one full unique pass")
    final_run = ROOT / "runs" / RUN_ID
    if final_run.exists():
        raise FileExistsError(final_run)
    ROOT.joinpath("runs").mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{RUN_ID}.incoming-", dir=ROOT / "runs"))
    try:
        cp = temporary / "checkpoints/step_00000200"
        cp.mkdir(parents=True)
        for name in ("model.safetensors", "optimizer.pt", "scheduler.json"):
            os.link(checkpoint / name, cp / name)
        first_refs = []
        for rank in range(8):
            sampler = MixtureDistributedSampler(schedule, num_replicas=8, rank=rank,
                          micro_batch_size=cfg["micro_batch_size"], gradient_accumulation_steps=cfg["gradient_accumulation_steps"])
            state = sampler.state_dict()
            state["committed_global_position"] = position
            state["next_reference"] = schedule.reference_at(position + rank * cfg["micro_batch_size"])
            sampler.load_state_dict(state)
            iterator = iter(sampler)
            for _ in range(cfg["micro_batch_size"] * cfg["gradient_accumulation_steps"]):
                first_refs.append(next(iterator))
            sampler.commit_step()
            if sampler.committed_global_position != position + b["global_sequences_per_step"]:
                raise ValueError("new rank commit diverged")
            write(cp / f"sampler_rank_{rank:05d}.json", state)
            runtime_path = cp / f"runtime_rank_{rank:05d}.pt"
            if rank < 6:
                os.link(checkpoint / runtime_path.name, runtime_path)
            else:
                # Preparation sees only physical GPUs 6 and 7; contexts exit
                # before the launcher's fresh eight-GPU free-card preflight.
                device = torch.device("cuda", rank - 6)
                runtime = {"torch_rng_state": torch.Generator().manual_seed(cfg["seed"] + rank).get_state(),
                           "cuda_rng_state": torch.Generator(device=device).manual_seed(cfg["seed"] + rank).get_state(),
                           "loader_generator_state": torch.Generator().manual_seed(cfg["seed"] + 10000 + rank).get_state()}
                torch.save(runtime, runtime_path)
                loader_generator = torch.Generator()
                _restore_rank_runtime_state(runtime, loader_generator=loader_generator, device=device)
                if not torch.equal(torch.cuda.get_rng_state(device), runtime["cuda_rng_state"]) or not torch.equal(loader_generator.get_state(), runtime["loader_generator_state"]):
                    raise ValueError("new rank RNG roundtrip failed")
        expected_refs = [schedule.sample_ref_at(i) for i in range(position, position + b["global_sequences_per_step"])]
        if len(set(first_refs)) != len(expected_refs) or set(first_refs) != set(expected_refs):
            raise ValueError("eight-rank first step drops/duplicates previously unread samples")
        transition = {"type": "explicit_replicated_ddp_world_size_transition_v1", "created_at": now(),
                      "source_checkpoint": str(checkpoint), "source_checkpoint_manifest_sha256": CHECKPOINT_SHA,
                      "source_world_size": 6, "target_world_size": 8,
                      "source_run_manifest_sha256": sha(old_run / "run_manifest.json"),
                      "source_trial_receipt_sha256": sha(SOURCE / "reports/TRIAL200.json"),
                      "model_optimizer_scheduler_bytes_preserved": True,
                      "actual_optimizer_step_preserved": 200, "tokens_seen_preserved": old_m["tokens_seen"],
                      "committed_global_position_preserved": position,
                      "existing_rank_rng_hardlinked": list(range(6)), "new_rank_rng_seeded": [6, 7],
                      "bitwise_six_card_equivalence": False,
                      "reason": "Global batch changes from 24 to 32; existing steps are not renumbered.", "budget": b}
        new_m = {"version": old_m["version"], "stage": old_m["stage"], "created_at": now(),
                 "step": 200, "tokens_seen": old_m["tokens_seen"], "world_size": 8,
                 "compatibility": {**old_m["compatibility"], "world_size": 8, "run_config_sha256": sha(cfg_path)},
                 "migration": transition, "artifacts": artifacts(cp)}
        write(cp / "manifest.json", new_m)
        write(temporary / "checkpoints/latest.json", {"version": old_m["version"], "step": 200,
              "directory": cp.name, "manifest_sha256": sha(cp / "manifest.json")})
        run_manifest = read(old_run / "run_manifest.json")
        run_manifest.update({"created_at": now(), "run_id": RUN_ID, "purpose": cfg["purpose"],
                             "world_size": 8, "global_sequences_per_step": 32, "max_steps": cfg["max_steps"],
                             "source_tree_sha256": sha256_source_tree(ROOT / "code"),
                             "run_config": {"path": str(cfg_path), "sha256": sha(cfg_path)},
                             "model_spec": {"path": str(ROOT / "code" / cfg["model_spec"]), "sha256": MODEL_SHA},
                             "data_pipeline": cfg["data_pipeline"], "initialization": transition})
        run_manifest["train_sampler"]["global_samples_per_optimizer_step"] = 32
        write(temporary / "run_manifest.json", run_manifest)
        write(temporary / "status.json", {"state": "prepared_transition_not_started", "step": 200, "tokens_seen": old_m["tokens_seen"]})
        write(temporary / "transition.json", transition)
        temporary.rename(final_run)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    receipt = {"status": "eight_card_transition_prepared_not_started", "generated_at": now(),
               "run": str(final_run), "config_sha256": sha(cfg_path), "data_manifest_sha256": DATA_SHA,
               "trainer_tree_sha256": TRAINER_SHA, "source_checkpoint_manifest_sha256": CHECKPOINT_SHA,
               "transition_checkpoint_manifest_sha256": sha(final_run / "checkpoints/step_00000200/manifest.json"),
               "transition_receipt_sha256": sha(final_run / "transition.json"),
               "budget": b, "all_source_artifact_hashes_verified": True,
               "full_unique_train_samples_checked": len(refs), "first_eight_rank_step_refs_checked": 32,
               "scheduler_config_unchanged": True, "shared_state_hardlinked": True,
               "model_capability_certified": False}
    write(ROOT / "reports/PREPARED.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def verify_prepared() -> dict:
    cfg, cfg_path = verify_bindings()
    receipt = read(ROOT / "reports/PREPARED.json")
    if receipt["config_sha256"] != sha(cfg_path) or cfg["max_steps"] != receipt["budget"]["end_step"]:
        raise ValueError("prepared config changed")
    if sha(Path(cfg["output_root"]) / RUN_ID / "transition.json") != receipt["transition_receipt_sha256"]:
        raise ValueError("transition receipt changed")
    return receipt


def complete() -> None:
    p = verify_prepared()
    run = ROOT / "runs" / RUN_ID
    if read(run / "status.json")["state"] != "completed":
        raise ValueError("run not completed")
    events = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    summary = full_summary(events, p["budget"])
    cp = run / "checkpoints" / f"step_{p['budget']['end_step']:08d}"
    m = read(cp / "manifest.json")
    if m["step"] != p["budget"]["end_step"] or m["tokens_seen"] != p["budget"]["total_loss_tokens"] or m["world_size"] != 8:
        raise ValueError("final checkpoint progress mismatch")
    check_artifacts(cp, m)
    for rank in range(8):
        if read(cp / f"sampler_rank_{rank:05d}.json")["committed_global_position"] != p["budget"]["end_position"]:
            raise ValueError("final rank data position mismatch")
    write(ROOT / "reports/COMPLETED.json", {"status": "full_training_completed_and_final_checkpoint_verified",
          "generated_at": now(), "budget": p["budget"], "summary": summary,
          "final_checkpoint": str(cp), "final_checkpoint_manifest_sha256": sha(cp / "manifest.json"),
          "all_artifact_hashes_verified": True, "test_split_used_for_training": False,
          "model_capability_certified": False})
    print("FULL_TRAINING_COMPLETED_AND_CHECKPOINT_VERIFIED", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "verify-prepared", "complete"))
    args = p.parse_args()
    sys.path.insert(0, str(ROOT / "code/src"))
    if args.mode == "prepare":
        prepare()
    elif args.mode == "complete":
        complete()
    else:
        print(json.dumps(verify_prepared(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
