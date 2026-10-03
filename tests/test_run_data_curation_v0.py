"""Offline tests for the deterministic, non-destructive curation router."""

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_data_curation_v0.py"
spec = importlib.util.spec_from_file_location("run_data_curation_v0", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def task(source: str, row: int, text: str) -> dict:
    return {"data": {"source_file": source, "source_row_idx": row, "source_dataset": "test", "text": text}}


def score(source: str, row: int, text: str, decision: str) -> dict:
    return {
        "source": {"data.source_file": source, "data.source_row_idx": row},
        "source_id": json.dumps([source, row], separators=(",", ":")),
        "text_sha256": module.sha256_text(text),
        "status": "ok",
        "model": "test-model",
        "prompt_sha256": "prompt",
        "schema_sha256": "schema",
        "annotation": {"decision": decision, "quality": "中", "issues": [], "reason": "test"},
    }


class DataCurationV0Tests(unittest.TestCase):
    def setUp(self):
        self.policy = {
            "version": "test-v0",
            "source_review_markers": ["Sign up to view the full content"],
            "extraction_damage_patterns": [{"id": "broken_code", "regex": r"class\s+\w+:\s*def\s+"}],
        }
        self.patterns = [(x["id"], module.re.compile(x["regex"]))
                         for x in self.policy["extraction_damage_patterns"]]

    def classify(self, text: str, old: str = "保留", new: str = "保留", duplicate_of=None):
        key = ("a.jsonl", 1)
        x = task(*key, text)
        return module.classify(key, x, score(*key, text, old), score(*key, text, new),
                               self.policy, self.patterns, duplicate_of, "policy-hash")

    def test_route_priorities_and_no_raw_text(self):
        cases = [
            ("ordinary informative text", "保留", "保留", None, "train"),
            ("ordinary informative text", "剔除", "剔除", None, "drop"),
            ("ordinary informative text", "保留", "剔除", None, "review"),
            ("class Configs: def __init__(self): pass", "保留", "保留", None, "repair"),
            ("Sign up to view the full content", "保留", "保留", None, "source_check"),
            ("Sign up to view the full content", "保留", "保留", "a.jsonl#0", "drop"),
        ]
        for text, old, new, duplicate, expected in cases:
            with self.subTest(expected=expected):
                record = self.classify(text, old, new, duplicate)
                self.assertEqual(record["route"], expected)
                self.assertNotIn("\"text\"", module.canonical_json(record))

    def test_hash_mismatch_rejected(self):
        item = task("a.jsonl", 1, "original")
        prediction = score("a.jsonl", 1, "different", "保留")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "text SHA mismatch"):
                module.load_predictions(path, {module.key_from_task(item): item})

    def test_deterministic_re_run(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            items = [task("a.jsonl", 2, "hello"), task("a.jsonl", 1, "hello")]
            (base / "tasks.json").write_text(json.dumps(items), encoding="utf-8")
            (base / "policy.json").write_text(json.dumps(self.policy), encoding="utf-8")
            for name in ("v1", "v2"):
                with (base / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
                    for item in items:
                        data = item["data"]
                        handle.write(json.dumps(score(data["source_file"], data["source_row_idx"], data["text"], "保留")) + "\n")
            args = argparse.Namespace(tasks=base / "tasks.json", v1=base / "v1.jsonl",
                                      v2=base / "v2.jsonl", policy=base / "policy.json",
                                      run_dir=base / "run")
            module.run(args)
            before = {path.name: path.read_bytes() for path in args.run_dir.iterdir()}
            module.run(args)
            after = {path.name: path.read_bytes() for path in args.run_dir.iterdir()}
            self.assertEqual(before, after)
            records = [json.loads(line) for line in (args.run_dir / "records.jsonl").read_text().splitlines()]
            self.assertEqual([x["route"] for x in records], ["train", "drop"])
            self.assertEqual(records[1]["duplicate_of"], "a.jsonl#1")


if __name__ == "__main__":
    unittest.main()
