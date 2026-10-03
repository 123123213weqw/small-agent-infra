from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import struct


SCRIPT = Path(__file__).parents[1] / "scripts" / "trainctl.py"
SPEC = importlib.util.spec_from_file_location("trainctl", SCRIPT)
assert SPEC and SPEC.loader
trainctl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainctl)


class TrainctlTest(unittest.TestCase):
    def test_parse_visible_devices(self) -> None:
        self.assertEqual(trainctl.parse_visible_devices("0, 2,7"), [0, 2, 7])
        with self.assertRaises(trainctl.PreflightError):
            trainctl.parse_visible_devices("0,0")
        with self.assertRaises(trainctl.PreflightError):
            trainctl.parse_visible_devices("GPU-uuid")

    def test_gpu_selection_fails_on_busy_device(self) -> None:
        gpus = [
            {
                "index": 0,
                "uuid": "GPU-a",
                "name": "NVIDIA L40",
                "memory_total_mib": 46068,
                "memory_used_mib": 0,
            },
            {
                "index": 1,
                "uuid": "GPU-b",
                "name": "NVIDIA L40",
                "memory_total_mib": 46068,
                "memory_used_mib": 10,
            },
        ]
        processes = [
            {"gpu_uuid": "GPU-b", "pid": 9, "process_name": "python", "used_memory_mib": 10}
        ]
        with self.assertRaisesRegex(trainctl.PreflightError, "active compute"):
            trainctl.validate_gpu_selection(gpus, processes, [0, 1], 2, "NVIDIA L40")

    def test_explicit_shared_gpu_requires_headroom(self) -> None:
        gpus = [{
            "index": 4, "uuid": "GPU-shared", "name": "NVIDIA L40",
            "memory_total_mib": 46068, "memory_used_mib": 2200,
        }]
        processes = [{
            "gpu_uuid": "GPU-shared", "pid": 9,
            "process_name": "other-user-python", "used_memory_mib": 2100,
        }]
        selected = trainctl.validate_gpu_selection(
            gpus, processes, [4], 1, "NVIDIA L40",
            allow_shared=True, minimum_free_memory_mib=24 * 1024,
            maximum_existing_memory_mib=8 * 1024,
        )
        self.assertEqual(selected[0]["index"], 4)
        with self.assertRaisesRegex(trainctl.PreflightError, "only"):
            trainctl.validate_gpu_selection(
                gpus, processes, [4], 1, "NVIDIA L40",
                allow_shared=True, minimum_free_memory_mib=44 * 1024,
                maximum_existing_memory_mib=8 * 1024,
            )
        with self.assertRaisesRegex(trainctl.PreflightError, "existing compute"):
            trainctl.validate_gpu_selection(
                gpus, processes, [4], 1, "NVIDIA L40",
                allow_shared=True, minimum_free_memory_mib=24 * 1024,
                maximum_existing_memory_mib=1024,
            )

    def test_build_launch_command(self) -> None:
        command = trainctl.build_launch_command(
            Path("/python"), 8, Path("/repo/run.json"), "run-a", 3, False, True
        )
        self.assertIn("--nproc_per_node=8", command)
        self.assertEqual(command[-3:], ["--max-steps", "3", "--save-final-checkpoint"])

    def test_validate_configuration_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src" / "small_agent").mkdir(parents=True)
            (root / "configs").mkdir()
            data = root / "data"
            data.mkdir()
            train = data / "train.npy"
            validation = data / "validation.npy"
            train.write_bytes(b"1234")
            validation.write_bytes(b"56")
            manifest = {
                "stage": "internal_smoke_token_stream",
                "dtype": "uint16",
                "vocab_size": 32768,
                "eos_token_id": 2,
                "license_gate": {"training_eligible": False},
                "splits": {
                    "train": {
                        "file": train.name,
                        "bytes": 4,
                        "sha256": trainctl.sha256_file(train),
                    },
                    "validation": {
                        "file": validation.name,
                        "bytes": 2,
                        "sha256": trainctl.sha256_file(validation),
                    },
                },
            }
            (data / "manifest.json").write_text(json.dumps(manifest))
            model = {
                "config": {
                    "vocab_size": 32768,
                    "eos_token_id": 2,
                    "max_position_embeddings": 8192,
                }
            }
            model_path = root / "configs" / "model.json"
            model_path.write_text(json.dumps(model))
            run = {
                "model_spec": "configs/model.json",
                "data_dir": "data",
                "output_root": "runs",
                "purpose": "test",
                "distributable": False,
                "context_length": 4096,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "max_steps": 1,
                "optimizer": {},
                "scheduler": {},
                "validation": {},
            }
            run_path = root / "configs" / "run.json"
            run_path.write_text(json.dumps(run))
            result = trainctl.validate_configuration(root, run_path)
            self.assertEqual(result[1], model_path.resolve())
            train.write_bytes(b"broken")
            with self.assertRaisesRegex(trainctl.PreflightError, "byte count mismatch"):
                trainctl.validate_configuration(root, run_path)

    def test_validate_sharded_manifest_and_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src" / "small_agent").mkdir(parents=True)
            (root / "configs").mkdir()
            data = root / "data"
            data.mkdir()
            shards = []
            for shard_id, split, tokens in ((0, "train", 10), (1, "validation", 7)):
                bin_path = data / f"{split}-{shard_id:05d}.bin"
                idx_path = data / f"{split}-{shard_id:05d}.idx"
                bin_path.write_bytes(struct.pack("<" + "H" * tokens, *range(10, 10 + tokens)))
                idx_path.write_bytes(struct.pack("<QII", 0, 5, 0) + struct.pack("<QII", 4, min(5, tokens - 4), 0 if tokens >= 9 else 1))
                shards.append({
                    "shard_id": shard_id, "split": split, "domain": "general",
                    "bin": bin_path.name, "idx": idx_path.name,
                    "tokens": tokens, "samples": 2,
                    "bin_bytes": bin_path.stat().st_size, "idx_bytes": idx_path.stat().st_size,
                    "bin_sha256": trainctl.sha256_file(bin_path),
                    "idx_sha256": trainctl.sha256_file(idx_path),
                })
            (data / "manifest.json").write_text(json.dumps({
                "version": "sharded_token_stream_v1",
                "stage": "internal_smoke_sharded_token_stream",
                "dtype": "uint16-le", "index_dtype": "<QII",
                "sequence_length": 5, "vocab_size": 64, "eos_token_id": 2,
                "tokenizer_sha256": "toy-tokenizer-hash",
                "license_gate": {"training_eligible": False},
                "shards": shards,
            }))
            (root / "configs" / "model.json").write_text(json.dumps({"config": {
                "max_position_embeddings": 16, "vocab_size": 64, "eos_token_id": 2,
            }}))
            run_path = root / "configs" / "run.json"
            run_path.write_text(json.dumps({
                "model_spec": "configs/model.json", "data_dir": "data", "output_root": "runs",
                "purpose": "test", "distributable": False, "data_reader": "sharded_v1",
                "context_length": 5, "micro_batch_size": 1,
                "gradient_accumulation_steps": 1, "max_steps": 1,
                "optimizer": {}, "scheduler": {}, "validation": {},
            }))
            trainctl.validate_configuration(root, run_path)
            run = json.loads(run_path.read_text())
            run["data_pipeline"] = {"num_workers": 2, "prefetch_factor": 0}
            run_path.write_text(json.dumps(run))
            with self.assertRaisesRegex(trainctl.PreflightError, "data_pipeline"):
                trainctl.validate_configuration(root, run_path)
            run.pop("data_pipeline")
            run_path.write_text(json.dumps(run))
            with (data / shards[0]["bin"]).open("r+b") as handle:
                handle.seek(0)
                handle.write(b"XX")
            with self.assertRaisesRegex(trainctl.PreflightError, "SHA-256 mismatch"):
                trainctl.validate_configuration(root, run_path)


if __name__ == "__main__":
    unittest.main()
