#!/usr/bin/env python3
"""Compare sharded DataLoader settings after identical L40 training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def train_rows(run_dir: Path) -> list[dict]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise ValueError(f"run is not completed: {run_dir}")
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows = [row for row in rows if row.get("event") == "train"]
    if not rows:
        raise ValueError(f"run has no training metrics: {run_dir}")
    return rows


def summarize(rows: list[dict], *, warmup_steps: int) -> dict:
    steady = [row for row in rows if int(row["step"]) > warmup_steps]
    if len(steady) < 3:
        raise ValueError("need at least three post-warmup training points")
    for field in ("data_wait_ms", "h2d_copy_ms", "data_wait_fraction", "sample_refs_rank0"):
        if any(field not in row for row in steady):
            raise ValueError(f"missing data-pipeline measurement: {field}")
    return {
        "steady_steps": len(steady),
        "median_tokens_per_second": statistics.median(row["tokens_per_second"] for row in steady),
        "median_data_wait_ms": statistics.median(row["data_wait_ms"] for row in steady),
        "max_data_wait_ms": max(row["data_wait_ms"] for row in steady),
        "median_data_wait_fraction": statistics.median(row["data_wait_fraction"] for row in steady),
        "median_h2d_copy_ms": statistics.median(row["h2d_copy_ms"] for row in steady),
        "peak_allocated_gib": max(row["peak_allocated_gib"] for row in rows),
    }


def compare(baseline: Path, candidate: Path, *, warmup_steps: int = 2) -> dict:
    left_manifest = json.loads((baseline / "run_manifest.json").read_text(encoding="utf-8"))
    right_manifest = json.loads((candidate / "run_manifest.json").read_text(encoding="utf-8"))
    for field in (
        "source_tree_sha256",
        "world_size",
        "parameter_count",
        "global_sequences_per_step",
        "model_spec",
        "data_manifest",
        "train_sampler",
    ):
        if left_manifest[field] != right_manifest[field]:
            raise ValueError(f"runs are not comparable: {field} differs")
    left, right = train_rows(baseline), train_rows(candidate)
    if [row["step"] for row in left] != [row["step"] for row in right]:
        raise ValueError("training step lists differ")
    trace_equal = all(
        row["sample_refs_rank0"] == other["sample_refs_rank0"]
        for row, other in zip(left, right)
    )
    if not trace_equal:
        raise ValueError("rank-0 sample trajectories differ")
    baseline_summary = summarize(left, warmup_steps=warmup_steps)
    candidate_summary = summarize(right, warmup_steps=warmup_steps)
    ratio = candidate_summary["median_tokens_per_second"] / baseline_summary["median_tokens_per_second"]
    return {
        "version": "sharded_data_profile_comparison_v1",
        "baseline_run": str(baseline.resolve()),
        "candidate_run": str(candidate.resolve()),
        "warmup_steps_excluded": warmup_steps,
        "rank0_sample_trajectory_equal": True,
        "maximum_absolute_loss_difference": max(
            abs(float(row["loss"]) - float(other["loss"]))
            for row, other in zip(left, right)
        ),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "candidate_throughput_ratio": ratio,
        "decision": (
            "keep_simple_loader_for_this_dataset"
            if baseline_summary["median_data_wait_fraction"] < 0.05 and ratio < 1.03
            else "investigate_worker_prefetch_with_larger_representative_data"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, warmup_steps=args.warmup_steps)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
