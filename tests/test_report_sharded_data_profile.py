from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report_sharded_data_profile.py"
SPEC = importlib.util.spec_from_file_location("report_sharded_data_profile", SCRIPT)
assert SPEC and SPEC.loader
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def write_run(path: Path, *, throughput: float, mismatched: bool = False) -> None:
    path.mkdir()
    (path / "run_manifest.json").write_text(json.dumps({
        "source_tree_sha256": "same-source",
        "world_size": 8,
        "parameter_count": 100,
        "global_sequences_per_step": 32,
        "model_spec": {"sha256": "same-model"},
        "data_manifest": {"sha256": "same-data"},
        "train_sampler": {"type": "sharded_mixture_v1"},
    }))
    (path / "status.json").write_text(json.dumps({"state": "completed", "step": 5}))
    rows = []
    for step in range(1, 6):
        rows.append({
            "event": "train", "step": step,
            "loss": 3.0 - step / 10,
            "tokens_per_second": throughput,
            "data_wait_ms": 2.0,
            "h2d_copy_ms": 0.1,
            "data_wait_fraction": 0.01,
            "peak_allocated_gib": 20.0,
            "sample_refs_rank0": [step if not (mismatched and step == 4) else 999],
        })
    (path / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")


class DataProfileReportTests(unittest.TestCase):
    def test_compare_identical_trace_and_low_wait(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_run(root / "base", throughput=100.0)
            write_run(root / "workers", throughput=102.0)
            report = reporter.compare(root / "base", root / "workers")
            self.assertTrue(report["rank0_sample_trajectory_equal"])
            self.assertAlmostEqual(report["candidate_throughput_ratio"], 1.02)
            self.assertEqual(report["decision"], "keep_simple_loader_for_this_dataset")

    def test_trace_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_run(root / "base", throughput=100.0)
            write_run(root / "workers", throughput=102.0, mismatched=True)
            with self.assertRaisesRegex(ValueError, "trajectories differ"):
                reporter.compare(root / "base", root / "workers")


if __name__ == "__main__":
    unittest.main()
