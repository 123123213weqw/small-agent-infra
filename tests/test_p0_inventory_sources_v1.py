import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/p0_inventory_sources_v1.py"
spec = importlib.util.spec_from_file_location("inventory", SCRIPT)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class InventoryTests(unittest.TestCase):
    def test_published_report_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "inventory.json"
            inventory.write_new(p, {"a": 1})
            with self.assertRaises(FileExistsError):
                inventory.write_new(p, {"a": 2})
            self.assertEqual(json.loads(p.read_text()), {"a": 1})
            self.assertEqual([x.name for x in p.parent.iterdir()], ["inventory.json"])

    def test_hash_mismatch_is_not_a_validated_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = root / "sample.jsonl"; p.write_text('{"text":"hello"}\n')
            registry = {("source", p.name): {"expected_bytes": p.stat().st_size,
                "expected_sha256": "0" * 64}}
            out = inventory.inspect_file(p, root, "source", registry, True)
            self.assertEqual(out["hash_status"], "mismatch")
            self.assertFalse(out["training_approved_by_inventory"])
            self.assertIsNone(out["own_tokenizer_tokens"])
            self.assertIsNone(out["rows"])

    def test_missing_pools_do_not_produce_ready_tokens(self):
        with tempfile.TemporaryDirectory() as d:
            pools, totals = inventory.frozen_inventory(Path(d))
            self.assertEqual(totals, {})
            self.assertEqual(len(pools), 3)
            self.assertTrue(all(p["status"] == "missing" for p in pools))

    def test_prior_manifest_binding_mismatch_excludes_counts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i, rel in enumerate(inventory.FROZEN):
                p = root / rel; p.parent.mkdir(parents=True)
                value = {"stage": "accepted_real", "license_gate": {"training_eligible": True},
                    "family_split_conflicts": 0, "splits": {"train": {"documents": 1, "text_tokens": 10}},
                    "inputs": {"prior_sources": []}}
                if i == 2:
                    value["inputs"]["prior_sources"] = [{"path": str((root / inventory.FROZEN[0]).parent),
                        "manifest_sha256": "0" * 64}]
                p.write_text(json.dumps(value))
            pools, totals = inventory.frozen_inventory(root)
            self.assertEqual(pools[0]["prior_manifest_binding"], "mismatch")
            self.assertFalse(pools[0]["included_in_declared_frozen_totals"])
            self.assertEqual(totals["train"]["body_tokens"], 20)
            self.assertEqual(totals["train"]["stream_tokens_with_eos"], 22)

    def test_raw_and_candidate_counts_do_not_enter_frozen_totals(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / inventory.RAW_ROOTS[0] / "raw/source/part.jsonl"
            p.parent.mkdir(parents=True); p.write_text('{"text":"data"}\n')
            c = root / inventory.CANDIDATES[0]; c.parent.mkdir(parents=True)
            c.write_text(json.dumps({"counts": {"candidate_tokens": 1000000}, "training_approved": False}))
            out = inventory.collect(root, [], set())
            self.assertEqual(out["declared_existing_frozen_totals"], {})
            self.assertEqual(out["new_approved_train_tokens"], 0)
            self.assertIsNone(out["raw_groups"][0]["own_tokenizer_train_tokens"])
            self.assertFalse(out["additional_candidate_reports"][0]["added_to_ready_totals"])


if __name__ == "__main__":
    unittest.main()
