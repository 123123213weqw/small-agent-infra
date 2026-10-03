#!/usr/bin/env python3
"""Read-only, CPU-only TensorBoard export of existing training/benchmark evidence.

No trainer imports, model loads, CUDA context, hooks, collectives or file writes to
the source runs. Historical missing metrics are never filled in with zeroes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
from typing import Any

VERSION = "training_tensorboard_sidecar_v1"
NOTES = """### 监控口径与边界
- 本视图来自只读 CPU sidecar，不会改训练、模型、优化器、随机数或采样顺序。
- train/grad_norm 是裁剪前范数；正式训练历史只记录每个日志窗口最后一步，
  sampled_clip_indicator **不是整个训练的裁剪比例**。
- train/estimated_window_mean_step_seconds 从 token 增量与吞吐推算，
  不是逐步计时；历史记录不支持训练步耗时 p95。
- benchmark/performance 的统计只包括 measurement，排除 warmup、加载、验证和 profiler。
  benchmark/probe 是 128 条固定留出探针，不能当完整验证或长期质量验收。
- validation/rank0_batches 是原始记录中的本 rank 批次数；predicted_tokens 是全 rank 合计。
- hardware 的横轴是 sidecar 启动后的秒数，utilization_percent 是驱动采样，**不是 MFU**。
  per_gpu used_gib 包括其它进程；不能和模型 allocated_gib 混为一谈。
- 未采集的 NaN/Inf 次数、逐层激活/GDN 状态、分领域 loss、能力评测不伪造。
  梯度/探针门禁通过也不表示优化已获准用于正式训练。
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    if path.stat().st_size > 512 * 1024:
        raise ValueError("unexpected oversized metadata: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def safe_metadata(value: Any) -> Any:
    """Do not publish auth/environment fields in config text."""
    if isinstance(value, dict):
        return {key: ("[REDACTED]" if any(word in key.lower() for word in
                ("password", "secret", "api_key", "credential", "cookie", "authorization", "environment"))
                else safe_metadata(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_metadata(item) for item in value]
    return value


def scalar(writer, tag: str, value: Any, step: int) -> None:
    if finite_number(value):
        writer.add_scalar(tag, value, step)


def mapping(writer, prefix: str, row: dict, names: dict[str, str], step: int) -> None:
    for source, target in names.items():
        scalar(writer, prefix + "/" + target, row.get(source), step)


def clip_indicator(norm: Any, threshold: float | None) -> int | None:
    if threshold is None or not finite_number(norm) or norm < 0 or not finite_number(threshold) or threshold <= 0:
        return None
    # PyTorch 2.5 clip_grad_norm_: clamp(max_norm / (total_norm + 1e-6), max=1).
    return int(threshold / (norm + 1e-6) < 1)


def training_row(writer, row: dict, previous: dict | None, clip: float | None) -> dict | None:
    step = row.get("step", 0)
    if not isinstance(step, int) or step < 0:
        raise ValueError("invalid training step")
    event = row.get("event")
    if event == "resume":
        return {"step": step, "tokens_seen": row.get("tokens_seen")}
    if event == "train":
        mapping(writer, "train", row, {key: key for key in
            ("loss", "learning_rate", "grad_norm", "tokens_per_second", "tokens_seen", "epoch", "sampler_cursor")}, step)
        mapping(writer, "system", row, {"peak_allocated_gib": "gpu_allocated_gib",
                                           "peak_reserved_gib": "gpu_reserved_gib"}, step)
        mapping(writer, "data", row, {"data_wait_ms": "wait_ms", "h2d_copy_ms": "h2d_copy_ms",
                     "data_wait_fraction": "wait_fraction", "h2d_copy_fraction": "h2d_copy_fraction"}, step)
        scalar(writer, "train/sampled_clip_indicator", clip_indicator(row.get("grad_norm"), clip), step)
        if previous and all(finite_number(v) for v in
                (previous.get("step"), previous.get("tokens_seen"), row.get("tokens_seen"), row.get("tokens_per_second"))):
            ds = step - previous["step"]
            dt = row["tokens_seen"] - previous["tokens_seen"]
            if ds > 0 and dt > 0 and row["tokens_per_second"] > 0:
                scalar(writer, "train/estimated_window_mean_step_seconds", dt / row["tokens_per_second"] / ds, step)
        return {"step": step, "tokens_seen": row.get("tokens_seen")}
    if event == "validation":
        mapping(writer, "validation", row, {"loss": "loss", "perplexity": "perplexity",
                      "predicted_tokens": "predicted_tokens", "batches": "rank0_batches"}, step)
    if event == "checkpoint":
        mapping(writer, "checkpoint", row, {"elapsed_seconds": "save_seconds", "tokens_seen": "tokens_seen"}, step)
    return previous


def step_statistics(rows: list[dict], clip: float | None) -> dict:
    measured = [r for r in rows if r.get("phase") == "measurement"]
    if not measured:
        return {}
    for row in measured:
        if not finite_number(row.get("seconds")) or row["seconds"] <= 0 or not finite_number(row.get("loss_tokens")) or row["loss_tokens"] <= 0:
            raise ValueError("invalid measured timing/token record")
    seconds = sorted(r["seconds"] for r in measured)
    tokens = sum(r["loss_tokens"] for r in measured)
    result = {"mean_step_seconds": statistics.mean(seconds), "median_step_seconds": statistics.median(seconds),
              "p95_step_seconds": seconds[math.ceil(0.95 * len(seconds)) - 1],
              "maximum_step_seconds": max(seconds), "loss_tokens_per_second": tokens / sum(seconds),
              "measured_steps": len(measured), "predicted_tokens": tokens}
    indicators = [clip_indicator(r.get("grad_norm"), clip) for r in measured]
    if all(v is not None for v in indicators):
        result["clip_fraction"] = sum(indicators) / len(indicators)
    return result


def benchmark_step(writer, row: dict, clip: float | None) -> None:
    step = row["step"]
    if not isinstance(step, int) or step < 1 or row.get("phase") not in {"measurement", "warmup"}:
        raise ValueError("invalid benchmark step/phase")
    prefix = "benchmark/" + row["phase"]
    mapping(writer, prefix, row, {"loss": "loss", "grad_norm": "grad_norm", "seconds": "step_seconds",
                               "loss_tokens": "predicted_tokens"}, step)
    scalar(writer, prefix + "/clip_indicator", clip_indicator(row.get("grad_norm"), clip), step)
    if finite_number(row.get("seconds")) and row["seconds"] > 0:
        scalar(writer, prefix + "/tokens_per_second", row.get("loss_tokens", 0) / row["seconds"], step)


def benchmark_result(writer, result: dict, step: int) -> None:
    mapping(writer, "benchmark/performance", result, {key: key for key in
        ("loss_tokens_per_second", "median_step_seconds", "maximum_step_seconds", "measured_steps")}, step)
    mapping(writer, "benchmark/memory", result, {"peak_allocated_gib": "max_rank_allocated_gib",
                                                "peak_reserved_gib": "max_rank_reserved_gib"}, step)
    for name, x in (("initial_validation", 0), ("final_validation", step)):
        mapping(writer, "benchmark/probe", result.get(name, {}), {key: key for key in
            ("loss", "perplexity", "predicted_tokens", "samples")}, x)
    mapping(writer, "benchmark/numerics", result.get("gradient_check", {}), {key: key for key in
            ("relative_l2", "cosine", "maximum_absolute_difference", "compared_parameters")}, 0)
    mapping(writer, "benchmark/numerics", result.get("validation_screen", {}), {key: key for key in
            ("initial_loss_difference", "post40_step_loss_difference")}, step)
    for name, tag in (("gradient_check", "gradient_gate_passed"), ("validation_screen", "probe_gate_passed")):
        passed = result.get(name, {}).get("passed")
        if isinstance(passed, bool):
            scalar(writer, "benchmark/numerics/" + tag, int(passed), step)
    writer.add_text("metadata/result_scope", "```json\n" + json.dumps(safe_metadata(result), ensure_ascii=False, indent=2) + "\n```", step)


def complete_jsonl(path: Path) -> list[dict]:
    """A writer may be mid-line. Only parse newline-terminated JSON objects."""
    if not path.exists():
        return []
    content = path.read_bytes()
    lines = content.split(b"\n")[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def hardware_sample() -> list[dict]:
    fields = ["index", "memory.used", "memory.total", "utilization.gpu", "temperature.gpu", "power.draw", "clocks.sm"]
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"], text=True, timeout=5)
    records = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            raise ValueError("invalid nvidia-smi row")
        row = {"index": int(values[0])}
        for field, value in zip(fields[1:], values[1:]):
            try:
                number = float(value)
            except ValueError:  # Unsupported N/A is unavailable, not zero.
                continue
            if math.isfinite(number):
                row[field] = number
        records.append(row)
    return records


def hardware_write(writer, records: list[dict], elapsed: int) -> None:
    for row in records:
        prefix = "hardware/gpu_" + str(row["index"])
        mapping(writer, prefix, row, {"utilization.gpu": "utilization_percent", "temperature.gpu": "temperature_c",
                                     "power.draw": "power_w", "clocks.sm": "sm_clock_mhz"}, elapsed)
        for field, name in (("memory.used", "used_gib"), ("memory.total", "total_gib")):
            if finite_number(row.get(field)):
                scalar(writer, prefix + "/" + name, row[field] / 1024, elapsed)


class Exporter:
    """One fresh output per session; no replay into an existing event directory."""
    def __init__(self, output: Path, writer_factory, clip: float | None):
        if output.exists():
            raise FileExistsError("use a fresh output; existing evidence is immutable: " + str(output))
        output.mkdir(parents=True)
        self.output, self.writer_factory, self.clip = output, writer_factory, clip
        self.writers, self.seen, self.previous, self.rows = {}, {}, {}, {}

    def writer(self, name: str):
        if name not in self.writers:
            w = self.writer_factory(self.output / "runs" / name)
            w.add_text("metadata/metric_definitions", NOTES, 0)
            self.writers[name] = w
        return self.writers[name]

    def once(self, key: str, fingerprint: Any, callback) -> None:
        if self.seen.get(key) != fingerprint:
            callback()
            self.seen[key] = fingerprint

    def metadata(self, name: str, path: Path, tag: str = "config") -> None:
        if not path.exists():
            return
        fingerprint = sha(path)
        def emit():
            value = safe_metadata(read_json(path))
            self.writer(name).add_text("metadata/" + tag,
                "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```\nsha256=" + fingerprint, 0)
        self.once(name + str(path), fingerprint, emit)

    def training(self, run: Path) -> None:
        name = "formal_enhanced"
        self.metadata(name, run / "run_manifest.json", "run_manifest")
        manifest = read_json(run / "run_manifest.json")
        for key in ("run_config", "model_spec"):
            source = Path(manifest[key]["path"])
            if source.exists():
                if sha(source) != manifest[key]["sha256"]:
                    raise ValueError("bound metadata changed: " + str(source))
                self.metadata(name, source, key)
        all_rows = complete_jsonl(run / "metrics.jsonl")
        # Skip unchanged tail files; do not silently rewind/replay on truncation.
        old = self.rows.get(name, [])
        if all_rows[:len(old)] != old:
            raise ValueError("source training rows were rewritten/truncated")
        previous = self.previous.get(name)
        for row in all_rows[len(old):]:
            previous = training_row(self.writer(name), row, previous, self.clip)
        self.previous[name], self.rows[name] = previous, all_rows

    def campaign(self, campaign: Path) -> None:
        cfg = read_json(campaign / "config.json")
        variants = [v["id"] for v in cfg["variants"]]
        if (campaign / "selection.json").exists():
            variants.append("winner_confirmation")
        for variant in variants:
            name = campaign.name + "__" + variant
            directory = campaign / variant
            if not directory.exists():
                continue
            self.metadata(name, campaign / "config.json")
            self.metadata(name, campaign / "PREPARED.json", "prepared_identity")
            rows = complete_jsonl(directory / "steps.jsonl")
            old = self.rows.get(name, [])
            if rows[:len(old)] != old:
                raise ValueError("source benchmark rows rewritten/truncated")
            for row in rows[len(old):]:
                benchmark_step(self.writer(name), row, self.clip)
            self.rows[name] = rows
            if rows:
                def aggregate():
                    for tag, value in step_statistics(rows, self.clip).items():
                        scalar(self.writer(name), "benchmark/performance/" + tag, value, rows[-1]["step"])
                self.once(name + "aggregate", len(rows), aggregate)
            result_file = directory / "result.json"
            if result_file.exists():
                self.once(name + "result", sha(result_file), lambda: benchmark_result(self.writer(name), read_json(result_file), rows[-1]["step"] if rows else 40))
        for receipt in ("COMPLETED.json", "POSTCHECK.json", "CONTENTION.json", "GUARD_ERROR.json"):
            self.metadata(campaign.name + "__status", campaign / receipt, receipt.removesuffix(".json"))

    def flush(self):
        for writer in self.writers.values():
            writer.flush()

    def close(self):
        for writer in self.writers.values():
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, action="append", default=[])
    parser.add_argument("--quality-policy", type=Path)
    parser.add_argument("--follow-until-postcheck", type=Path)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--max-seconds", type=int, default=2700)
    parser.add_argument("--hardware", action="store_true")
    args = parser.parse_args()
    if args.interval < 5 or args.max_seconds < 1:
        parser.error("minimum polling interval 5 seconds; positive bound required")
    source_dirs = [args.training_run.resolve(), *(p.resolve() for p in args.campaign)]
    output = args.output.resolve()
    if any(output == p or p in output.parents or output in p.parents for p in source_dirs):
        parser.error("output must be separate from source runs")
    manifest = read_json(args.training_run / "run_manifest.json")
    config_path = Path(manifest["run_config"]["path"])
    if sha(config_path) != manifest["run_config"]["sha256"]:
        raise ValueError("training config hash changed")
    clip = read_json(config_path)["optimizer"]["gradient_clip"]
    # Only the SummaryWriter needs torch. Keep this process explicitly CPU-only.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from torch.utils.tensorboard import SummaryWriter
    exporter = Exporter(output, lambda path: SummaryWriter(log_dir=str(path), max_queue=100, flush_secs=10), clip)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    identity = {"version": VERSION, "script_sha256": sha(Path(__file__)),
                "started_at": datetime.now(timezone.utc).isoformat(), "training_run": str(args.training_run),
                "campaigns": [str(p) for p in args.campaign], "interval_seconds": args.interval,
                "cpu_only": True, "source_files_modified": False, "maximum_seconds": args.max_seconds,
                "gradient_clip": clip, "restart_policy": "fresh output directory, never replay into old events"}
    atomic_json(output / "STARTED.json", identity)
    status, polls = "running", 0
    try:
        while not stopped:
            tick = time.monotonic()
            exporter.training(args.training_run)
            for campaign in args.campaign:
                exporter.campaign(campaign)
            if args.quality_policy:
                exporter.metadata("quality_policy", args.quality_policy, "production_adoption_hold")
            if args.hardware:
                elapsed = int(time.monotonic() - started)
                try:
                    records = hardware_sample()
                    hardware_write(exporter.writer("live_hardware_seconds"), records, elapsed)
                    with (output / "hardware.jsonl").open("a") as f:
                        f.write(json.dumps({"captured_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": elapsed, "gpus": records}) + "\n")
                except (OSError, subprocess.SubprocessError, ValueError) as error:
                    exporter.writer("live_hardware_seconds").add_text("monitoring/hardware_unavailable", str(error), elapsed)
            polls += 1
            scalar(exporter.writer("sidecar_health"), "monitoring/poll_seconds", time.monotonic() - tick, polls)
            exporter.flush()
            atomic_json(output / "STATUS.json", dict(identity, state="running", polls=polls, exported_runs=len(exporter.writers),
                training_rows=len(exporter.rows.get("formal_enhanced", [])), updated_at=datetime.now(timezone.utc).isoformat()))
            if args.follow_until_postcheck and args.follow_until_postcheck.exists():
                status = "campaign_finished"; break
            if not args.follow_until_postcheck:
                status = "one_shot_completed"; break
            if time.monotonic() - started >= args.max_seconds:
                status = "time_limit_reached"; break
            for _ in range(math.ceil(args.interval * 5)):
                if stopped:
                    break
                time.sleep(0.2)
        if stopped:
            status = "stopped"
    except BaseException as error:
        status = "failed"
        atomic_json(output / "FAILED.json", {"status": status, "error": repr(error)})
        raise
    finally:
        exporter.close()
        atomic_json(output / "FINISHED.json", dict(identity, status=status, polls=polls,
            elapsed_seconds=time.monotonic() - started, exported_runs=len(exporter.writers),
            finished_at=datetime.now(timezone.utc).isoformat()))
    print(json.dumps({"status": status, "output": str(output), "runs": len(exporter.writers)}), flush=True)


if __name__ == "__main__":
    main()
