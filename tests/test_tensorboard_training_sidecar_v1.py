from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tensorboard_training_sidecar_v1 as m


class Writer:
    def __init__(self):
        self.scalars, self.texts = [], []
    def add_scalar(self, tag, value, step): self.scalars.append((tag, value, step))
    def add_text(self, tag, value, step): self.texts.append((tag, value, step))
    def flush(self): pass
    def close(self): pass
    def values(self): return {tag: value for tag, value, _ in self.scalars}


class TelemetryTest(unittest.TestCase):
    def test_scalar_missing_nan_inf_not_filled_with_zero(self):
        w = Writer()
        for value in (None, float("nan"), float("inf"), "1", True):
            m.scalar(w, "bad", value, 0)
        self.assertEqual(w.scalars, [])
        m.scalar(w, "valid_zero", 0, 0)
        self.assertEqual(w.scalars, [("valid_zero", 0, 0)])

    def test_clip_formula_matches_pytorch_epsilon(self):
        self.assertEqual(m.clip_indicator(1, 1), 1)
        self.assertEqual(m.clip_indicator(0.999998, 1), 0)
        self.assertEqual(m.clip_indicator(2, 1), 1)
        self.assertIsNone(m.clip_indicator(None, 1))
        self.assertIsNone(m.clip_indicator(-1, 1))

    def test_formal_gradient_only_sampled_no_fraction(self):
        w = Writer()
        row = {"event": "train", "step": 20, "tokens_seen": 2000,
               "tokens_per_second": 100, "grad_norm": 2, "h2d_copy_fraction": .02}
        previous = m.training_row(w, row, {"step": 10, "tokens_seen": 1000}, 1)
        values = w.values()
        self.assertEqual(values["train/estimated_window_mean_step_seconds"], 1)
        self.assertEqual(values["train/sampled_clip_indicator"], 1)
        self.assertEqual(values["data/h2d_copy_fraction"], .02)
        self.assertNotIn("train/clip_fraction", values)
        self.assertNotIn("train/p95_step_seconds", values)
        self.assertEqual(previous, {"step": 20, "tokens_seen": 2000})

    def test_resume_resets_window_cursor(self):
        w = Writer()
        cursor = m.training_row(w, {"event": "resume", "step": 200, "tokens_seen": 2000}, {"step": 1000}, 1)
        m.training_row(w, {"event": "train", "step": 210, "tokens_seen": 3000, "tokens_per_second": 50}, cursor, 1)
        self.assertEqual(w.values()["train/estimated_window_mean_step_seconds"], 2)

    def test_validation_tokens_global_batches_rank0(self):
        w = Writer()
        m.training_row(w, {"event": "validation", "step": 10, "loss": 2,
                       "perplexity": 7.4, "predicted_tokens": 999, "batches": 4}, None, 1)
        self.assertEqual(w.values()["validation/predicted_tokens"], 999)
        self.assertEqual(w.values()["validation/rank0_batches"], 4)
        self.assertNotIn("validation/global_batches", w.values())

    def test_checkpoint_duration_backfilled(self):
        w = Writer()
        m.training_row(w, {"event": "checkpoint", "step": 20, "elapsed_seconds": 12.3}, None, 1)
        self.assertEqual(w.values(), {"checkpoint/save_seconds": 12.3})

    def test_aggregate_excludes_warmup_and_token_weights_tps(self):
        rows = [{"phase": "warmup", "seconds": 999, "loss_tokens": 100, "grad_norm": 10},
                {"phase": "measurement", "seconds": 1, "loss_tokens": 100, "grad_norm": .5},
                {"phase": "measurement", "seconds": 3, "loss_tokens": 100, "grad_norm": 2}]
        values = m.step_statistics(rows, 1)
        self.assertEqual(values["loss_tokens_per_second"], 50)
        self.assertEqual(values["p95_step_seconds"], 3)
        self.assertEqual(values["clip_fraction"], .5)
        self.assertEqual(values["measured_steps"], 2)

    def test_missing_grad_no_fake_clip_fraction(self):
        values = m.step_statistics([{"phase": "measurement", "seconds": 1, "loss_tokens": 10}], 1)
        self.assertNotIn("clip_fraction", values)
        self.assertEqual(m.step_statistics([], 1), {})
        with self.assertRaises(ValueError):
            m.step_statistics([{"phase": "measurement", "seconds": 0, "loss_tokens": 10}], 1)

    def test_partial_jsonl_line_is_not_consumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "steps.jsonl"
            p.write_bytes(b'{"step": 1}\n{"step": 2')
            self.assertEqual(m.complete_jsonl(p), [{"step": 1}])
            with p.open("ab") as f: f.write(b'}\n')
            self.assertEqual(m.complete_jsonl(p), [{"step": 1}, {"step": 2}])

    def test_existing_output_rejected_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            source = p / "source.json"; source.write_text('{"seed": 3}')
            original = source.read_bytes()
            w = Writer()
            e = m.Exporter(p / "output", lambda _: w, 1)
            e.metadata("run", source)
            e.metadata("run", source)
            self.assertEqual(len(w.texts), 2)  # definitions + once-only config
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(FileExistsError): m.Exporter(p / "output", lambda _: Writer(), 1)

    def test_result_metrics_are_diagnostic_not_full_validation(self):
        w = Writer()
        result = {"initial_validation": {"loss": 2, "samples": 128, "predicted_tokens": 260000},
                  "final_validation": {"loss": 1.9, "perplexity": 6.7, "samples": 128},
                  "gradient_check": {"passed": True, "relative_l2": .001},
                  "validation_screen": {"passed": False, "post40_step_loss_difference": .02}}
        m.benchmark_result(w, result, 40)
        self.assertEqual(w.values()["benchmark/probe/samples"], 128)
        self.assertEqual(w.values()["benchmark/numerics/probe_gate_passed"], 0)
        self.assertFalse(any(tag.startswith("validation/") for tag, _, _ in w.scalars))

    def test_hardware_na_missing_not_zero_and_mib_to_gib(self):
        with patch.object(m.subprocess, "check_output", return_value="0, 1024, 46080, 97, 55, N/A, 1905\n"):
            rows = m.hardware_sample()
        w = Writer(); m.hardware_write(w, rows, 10)
        values = w.values()
        self.assertEqual(values["hardware/gpu_0/used_gib"], 1)
        self.assertEqual(values["hardware/gpu_0/utilization_percent"], 97)
        self.assertNotIn("hardware/gpu_0/power_w", values)
        self.assertFalse(any("mfu" in tag for tag in values))

    def test_nested_secrets_redacted_no_token_budget_redacted(self):
        value = m.safe_metadata({"seed": 1, "environment": {"DEEPSEEK_API_KEY": "x"},
                                 "nested": [{"password": "x", "total_tokens": 100}]})
        self.assertEqual(value["environment"], "[REDACTED]")
        self.assertEqual(value["nested"][0], {"password": "[REDACTED]", "total_tokens": 100})

    def test_poll_training_idempotent_reject_rewritten_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); run = p / "run"; run.mkdir()
            config = run / "config.json"; config.write_text('{"optimizer": {"gradient_clip": 1}}')
            model = run / "model.json"; model.write_text('{}')
            manifest = {key: {"path": str(path), "sha256": m.sha(path)} for key, path in
                        [("run_config", config), ("model_spec", model)]}
            (run / "run_manifest.json").write_text(json.dumps(manifest))
            metrics = run / "metrics.jsonl"
            metrics.write_text('{"event": "train", "step": 10, "loss": 2}\n')
            w = Writer(); e = m.Exporter(p / "out", lambda _: w, 1)
            e.training(run); count = len(w.scalars); e.training(run)
            self.assertEqual(count, len(w.scalars))
            metrics.write_text('{"event": "train", "step": 10, "loss": 3}\n')
            with self.assertRaisesRegex(ValueError, "rewritten"): e.training(run)

    def test_source_has_no_model_cuda_or_ddp_calls(self):
        source = (ROOT / "scripts/tensorboard_training_sidecar_v1.py").read_text()
        for forbidden in ("torch.cuda.", "register_forward_hook(", "dist.all_reduce(", "optimizer.step(", "model("):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__": unittest.main()
