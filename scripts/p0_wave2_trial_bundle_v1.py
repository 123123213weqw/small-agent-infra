#!/usr/bin/env python3
"""Prepare the existing trainer's Wave2 bundle and verify its 100→200 trial.

Reuses the deployed immutable pack combiner, sampler, and checkpoint format;
does not implement a new trainer or start training itself.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

RUN_ID = "prior-wave2-reuse-1040m-6l40-v1"
CONFIG = "base_1b_prior_wave2_internal_1040m_6l40_v1.json"
SPLITS = ("train", "validation", "test")


def read(p: Path) -> dict:
    return json.loads(p.read_text())


def sha(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            d.update(b)
    return d.hexdigest()


def write(p: Path, v: dict) -> None:
    if p.exists():
        raise FileExistsError(p)
    fd, tmp = tempfile.mkstemp(prefix=".trial-report-", dir=p.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(v, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def budget(samples: int, world: int, micro: int, accumulation: int, length: int) -> dict:
    if min(samples, world, micro, accumulation) < 1 or length < 2:
        raise ValueError("invalid training budget")
    global_batch = world * micro * accumulation
    steps = math.ceil(samples / global_batch)
    targets = global_batch * (length - 1)
    return {"world_size": world, "global_sequences_per_step": global_batch,
            "loss_tokens_per_step": targets, "steps": steps, "loss_tokens": steps * targets}


def trial_summary(events: list[dict], targets: int) -> dict:
    train = [e for e in events if e["event"] == "train"]
    validation = [e for e in events if e["event"] == "validation"]
    resumes = [e for e in events if e["event"] == "resume"]
    if not train or train[-1]["step"] != 200 or train[-1]["tokens_seen"] != 200 * targets:
        raise ValueError("trial has not completed exactly 200 steps")
    if not any(e["step"] == 100 and e["tokens_seen"] == 100 * targets for e in resumes):
        raise ValueError("no actual 100-step resume event")
    if not any(e["step"] == 200 for e in validation):
        raise ValueError("missing final validation")
    for e in train:
        if (not all(math.isfinite(float(e[f])) for f in ("loss", "grad_norm", "learning_rate", "tokens_per_second"))
                or e["tokens_seen"] != e["step"] * targets or e["tokens_per_second"] <= 0):
            raise ValueError("invalid training metric/token accounting")
    for e in validation:
        if not math.isfinite(e["loss"]):
            raise ValueError("non-finite validation")
    resumed = [e for e in train if e["step"] > 100]
    if not resumed or resumed[0]["step"] != 110:
        raise ValueError("unexpected post-resume logging boundary")
    steady = [e["tokens_per_second"] for e in train if e["step"] >= 150]
    return {"steps": 200, "loss_tokens": 200 * targets, "resume_events": resumes,
            "first_logged_loss": train[0]["loss"], "last_logged_loss": train[-1]["loss"],
            "validation": validation, "median_last_windows_loss_tokens_per_second": statistics.median(steady),
            "peak_allocated_gib": max(e["peak_allocated_gib"] for e in train),
            "peak_reserved_gib": max(e["peak_reserved_gib"] for e in train)}


def prepare(args) -> None:
    from small_agent.data.sharded_format import load_sharded_manifest
    from small_agent.data import ShardedTokenDataset
    from small_agent.data.mixture_sampler import DeterministicMixtureSchedule, MixtureDistributedSampler
    sys.path.insert(0, str(args.root / "code/scripts"))
    import trainctl
    cfg = read(args.root / "code/configs/runs" / CONFIG)
    receipt_path = args.data_root / "pretrain_reuse_wave2_v1/reports/L40_TRANSFER_STATS.json"
    receipt = read(receipt_path)
    if receipt["status"] != "transferred_and_data_integrity_verified":
        raise ValueError("transfer receipt not ready")
    if trainctl.sha256_source_tree(args.root / "code") != "e418e6c69bcf2f6724a8223325190a89f6799a72d75d6a9e0f0d1b283d23af41":
        raise ValueError("validated trainer snapshot changed")
    for name in ("prior_pack", "new_pack"):
        p = Path(receipt[name]["path"])
        if sha(p / "manifest.json") != receipt[name]["manifest_sha256"]:
            raise ValueError("source pack manifest changed")
    expected = receipt["combined_inventory"]
    b = budget(expected["splits"]["train"]["samples"], 6, cfg["micro_batch_size"],
               cfg["gradient_accumulation_steps"], cfg["context_length"])
    if cfg["max_steps"] != b["steps"] or cfg["scheduler"]["total_tokens"] != b["loss_tokens"]:
        raise ValueError("config budget mismatch")
    output = Path(cfg["data_dir"])
    if output.exists():
        raise FileExistsError(output)
    incoming = output.parent / ("." + output.name + ".incoming")
    if incoming.exists():
        raise FileExistsError(incoming)
    subprocess.run([sys.executable, str(args.root / "code/scripts/p0_combine_sharded_packs_v1.py"),
                    "--base", receipt["prior_pack"]["path"], "--add", receipt["new_pack"]["path"],
                    "--output", str(incoming), "--balance-by-train-samples"], check=True)
    manifest_path = incoming / "manifest.json"
    manifest = read(manifest_path)
    manifest["license_gate"] = {"training_eligible": True, "distribution_eligible": False,
                                "scope": "internal_pretraining_experiment_only",
                                "restriction": "Preserve both source packs' internal restrictions; no release authorization."}
    manifest["quality_gate"] = {"independent_factual_accuracy_certified": False,
                                "historical_strict_gate_unchanged": True, "no_new_model_review": True}
    manifest["readiness_receipts"] = {"wave2_l40_transfer_stats": {"path": str(receipt_path), "sha256": sha(receipt_path)}}
    for key in ("base", "add"):
        source = read(Path(manifest["source_packs"][key]["path"]) / "manifest.json")
        manifest["source_packs"][key]["license_gate"] = source["license_gate"]
    # The combined package is still private/unpublished during enrichment.
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    verified, _ = load_sharded_manifest(incoming, verify_hashes=True)
    by_split = {s: Counter() for s in SPLITS}
    counts = Counter()
    for entry in verified["shards"]:
        by_split[entry["split"]].update({"documents": entry["documents"], "stream_tokens": entry["tokens"], "samples": entry["samples"]})
        if entry["split"] == "train":
            counts[entry["domain"]] += entry["samples"]
        old = read(Path(verified["source_packs"][entry["source_pack"]]["path"]) / "manifest.json")
        # Content hashes are the immutable identity; inode equality proves no duplicate storage.
        source_entry = next(x for x in old["shards"] if x["bin_sha256"] == entry["bin_sha256"] and x["idx_sha256"] == entry["idx_sha256"])
        for kind in ("bin", "idx"):
            a = (incoming / entry[kind]).stat()
            p = Path(verified["source_packs"][entry["source_pack"]]["path"]) / source_entry[kind]
            c = p.stat()
            if (a.st_dev, a.st_ino) != (c.st_dev, c.st_ino):
                raise ValueError("combined file is not a hardlink")
    for split in SPLITS:
        for field in ("documents", "stream_tokens", "samples"):
            if by_split[split][field] != expected["splits"][split][field]:
                raise ValueError("combined inventory mismatch")
    if dict(counts) != verified["mixture"] or sum(counts.values()) != cfg["mixture_block_size"]:
        raise ValueError("mixture no longer follows one-pass sample counts")
    dataset = ShardedTokenDataset(incoming, "train", cfg["context_length"], verify_hashes=False)
    schedule = DeterministicMixtureSchedule(dataset, seed=cfg["seed"], block_size=cfg["mixture_block_size"])
    refs = {schedule.sample_ref_at(i) for i in range(len(dataset))}
    if len(refs) != len(dataset):
        raise ValueError("one-pass schedule duplicates/drops samples")
    restore_checks = 0
    first = []
    for rank in range(6):
        sampler = MixtureDistributedSampler(schedule, num_replicas=6, rank=rank, micro_batch_size=1, gradient_accumulation_steps=4)
        iterator = iter(sampler)
        first.extend(next(iterator) for _ in range(4))
        sampler.commit_step()
        state = sampler.state_dict()
        resumed = MixtureDistributedSampler(schedule, num_replicas=6, rank=rank, micro_batch_size=1, gradient_accumulation_steps=4)
        resumed.load_state_dict(state)
        if next(iter(resumed)) != next(iterator):
            raise ValueError("CPU sampler restore mismatch")
        restore_checks += 1
    if len(set(first)) != 24:
        raise ValueError("DDP first-step sample overlap")
    incoming.rename(output)
    result = {"status": "combined_pack_and_full_schedule_ready",
              "generated_at": datetime.now(timezone.utc).isoformat(), "config_sha256": sha(args.root / "code/configs/runs" / CONFIG),
              "data_dir": str(output), "data_manifest_sha256": sha(output / "manifest.json"),
              "source_transfer_receipt_sha256": sha(receipt_path), "trainer_tree_sha256": trainctl.sha256_source_tree(args.root / "code"),
              "inventory": expected, "budget": b, "full_unique_train_samples_checked": len(refs),
              "cpu_sampler_resume_ranks_checked": restore_checks, "hardlinks_verified": True,
              "all_bin_idx_hashes_and_indexes_verified": True, "training_started": False}
    write(args.root / "reports/PREPARED.json", result)
    print(json.dumps({k: result[k] for k in ("status", "data_dir", "budget", "full_unique_train_samples_checked")}), flush=True)


def report(args) -> None:
    run = args.root / "runs" / RUN_ID
    cfg = read(args.root / "code/configs/runs" / CONFIG)
    prepared = read(args.root / "reports/PREPARED.json")
    if prepared["config_sha256"] != sha(args.root / "code/configs/runs" / CONFIG):
        raise ValueError("run config changed during trial")
    if read(run / "status.json")["state"] != "completed":
        raise ValueError("trial run not completed")
    events = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    summary = trial_summary(events, prepared["budget"]["loss_tokens_per_step"])
    checkpoint_receipts = []
    for step in (100, 200):
        path = run / "checkpoints" / f"step_{step:08d}"
        m = read(path / "manifest.json")
        if m["step"] != step or m["world_size"] != 6 or m["tokens_seen"] != step * prepared["budget"]["loss_tokens_per_step"]:
            raise ValueError("checkpoint identity/token mismatch")
        for name, item in m["artifacts"].items():
            if Path(name).name != name or (path / name).stat().st_size != item["bytes"] or sha(path / name) != item["sha256"]:
                raise ValueError("checkpoint hash/size failure")
        if m["compatibility"]["data_manifest_sha256"] != prepared["data_manifest_sha256"]:
            raise ValueError("checkpoint data mismatch")
        for rank in range(6):
            s = read(path / f"sampler_rank_{rank:05d}.json")
            if s["committed_global_position"] != step * 24 or s["rank"] != rank:
                raise ValueError("checkpoint committed sampler mismatch")
        scheduler = read(path / "scheduler.json")
        if scheduler["config"]["total_tokens"] != cfg["scheduler"]["total_tokens"] or scheduler["tokens_seen"] != m["tokens_seen"]:
            raise ValueError("full LR schedule was shortened/reset")
        checkpoint_receipts.append({"step": step, "path": str(path), "manifest_sha256": sha(path / "manifest.json"), "all_artifact_hashes_verified": True})
    value = {"status": "trial200_passed_with_actual_checkpoint_resume",
             "generated_at": datetime.now(timezone.utc).isoformat(), "run": str(run),
             "prepared_sha256": sha(args.root / "reports/PREPARED.json"), **summary,
             "checkpoints": checkpoint_receipts, "gpu_indices": [0, 1, 3, 4, 5, 6],
             "full_budget_training_started": False, "model_capability_certified": False,
             "limitations": ["A 200-step warmup trial checks engineering, not general capabilities.",
                             "Successful restore checks state continuity; it is not a separate uninterrupted-run bitwise equivalence test."]}
    write(args.root / "reports/TRIAL200.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "report"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.root / "code/src"))
    (prepare if args.mode == "prepare" else report)(args)


if __name__ == "__main__":
    main()
