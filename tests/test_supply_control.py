"""Ledger fixture tests, NOT real corpus or quality acceptance tests."""
import copy
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from small_agent.curation.supply_control import SupplyControl, STAGES, digest, file_sha

CONFIG = Path(__file__).resolve().parents[1] / "configs/curation/base_1b_supply_20b_v4.json"


class SupplyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.recipe = json.loads(CONFIG.read_text())
        self.ledger = SupplyControl(self.root / "state.sqlite", self.recipe)

    def tearDown(self):
        self.ledger.close()
        self.temp.cleanup()

    def fixture(self, name, body_sha=None):
        binding = {"fixture_only": name}
        batch = self.ledger.register("math", binding)
        folder = self.root / name; folder.mkdir()
        evidence = folder / "evidence.json"
        evidence.write_text(json.dumps({"fixture_only": True, "batch": batch}))
        for stage in STAGES:
            self.ledger.commit_stage(batch, stage, [evidence])
        index = folder / "docs.jsonl"
        index.write_text(json.dumps({"domain": "math", "split": "train", "body_tokens": 7,
            "body_sha256": body_sha or digest(name)}) + "\n" + json.dumps({"domain": "math", "split": "validation",
            "body_tokens": 3, "body_sha256": digest(name + '-val')}) + "\n")
        pack = folder / "manifest.json"
        pack.write_text(json.dumps({"tokenizer_sha256":self.recipe["tokenizer_sha256"], "eos_token_id":2,
            "shards":[{"split":"train","tokens":8,"documents":1},
                      {"split":"validation","tokens":4,"documents":1}]}))
        receipt = {"status":"READY_INTERNAL", "batch_id":batch, "recipe_sha256":self.ledger.recipe_sha,
            "input_binding_sha256":digest(binding), "tokenizer_sha256":self.recipe["tokenizer_sha256"],
            "unique_train_body_tokens":7, "document_index":{"path":str(index),"sha256":file_sha(index)},
            "pack_manifest":{"path":str(pack),"sha256":file_sha(pack)},
            "checks":[{"name":n,"status":"passed","checker_revision":"fixture_not_quality_certification",
                "batch_id":batch,"recipe_sha256":self.ledger.recipe_sha,"input_binding_sha256":digest(binding),
                "evidence_files":[{"path":str(evidence),"sha256":file_sha(evidence)}]}
                for n in self.recipe["required_release_checks"]]}
        out=folder/'READY.json';out.write_text(json.dumps(receipt))
        return batch,out,receipt

    def test_candidates_never_advance_counter(self):
        self.ledger.register('math', {'raw':'candidate'})
        self.assertEqual(self.ledger.status()['unique_ready_train_body_tokens'],0)

    def test_register_is_idempotent_and_backpressure_persists(self):
        first=self.ledger.register('math',{'raw':1})
        self.assertEqual(first,self.ledger.register('math',{'raw':1}))
        self.ledger.register('code',{'raw':2})
        with self.assertRaises(ValueError):self.ledger.register('general_zh',{'raw':3})
        self.ledger.hold(first,'bad_source',quarantine=True)
        self.ledger.register('general_zh',{'raw':3})

    def test_cannot_skip_stage(self):
        batch=self.ledger.register('code',{'raw':1})
        p=self.root/'out';p.write_text('fixture')
        with self.assertRaises(ValueError):self.ledger.commit_stage(batch,'pack',[p])

    def test_release_counter_excludes_validation_and_is_idempotent(self):
        batch,p,_=self.fixture('a')
        self.assertEqual(self.ledger.release(batch,p)['unique_ready_train_body_tokens'],7)
        self.assertEqual(self.ledger.release(batch,p)['unique_ready_train_body_tokens'],7)

    def test_cross_batch_duplicate_rolls_back_entire_release(self):
        batch,p,_=self.fixture('a');self.ledger.release(batch,p)
        batch,p,_=self.fixture('b',body_sha=digest('a'))
        with self.assertRaises(sqlite3.IntegrityError):self.ledger.release(batch,p)
        self.assertEqual(self.ledger.status()['unique_ready_train_body_tokens'],7)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM releases').fetchone()[0],1)

    def test_new_release_family_is_retained_and_cross_batch_split_conflict_rolls_back(self):
        batch,p,x=self.fixture('family_a')
        index=Path(x['document_index']['path'])
        docs=[json.loads(l) for l in index.read_text().splitlines()]
        docs[0]['family_id']='same-repo'
        index.write_text(''.join(json.dumps(d)+'\n' for d in docs))
        x['document_index']['sha256']=file_sha(index);p.write_text(json.dumps(x))
        self.ledger.release(batch,p)
        self.assertEqual(self.ledger.db.execute("SELECT split FROM legacy_families WHERE family_id='same-repo'").fetchone()[0],'train')
        batch,p,x=self.fixture('family_b');index=Path(x['document_index']['path'])
        docs=[json.loads(l) for l in index.read_text().splitlines()]
        docs[1]['family_id']='same-repo'
        index.write_text(''.join(json.dumps(d)+'\n' for d in docs))
        x['document_index']['sha256']=file_sha(index);p.write_text(json.dumps(x))
        with self.assertRaises(ValueError):self.ledger.release(batch,p)
        self.assertEqual(self.ledger.status()['unique_ready_train_body_tokens'],7)

    def test_missing_check_does_not_release(self):
        batch,p,x=self.fixture('a');x['checks'].pop();p.write_text(json.dumps(x))
        with self.assertRaises(ValueError):self.ledger.release(batch,p)
        self.assertEqual(self.ledger.status()['unique_ready_train_body_tokens'],0)

    def test_borrowed_check_rejected(self):
        batch,p,x=self.fixture('a');x['checks'][0]['batch_id']='other';p.write_text(json.dumps(x))
        with self.assertRaises(ValueError):self.ledger.release(batch,p)

    def test_changed_stage_evidence_rejected(self):
        batch,p,_=self.fixture('a');(p.parent/'evidence.json').write_text('changed')
        with self.assertRaises(ValueError):self.ledger.release(batch,p)

    def test_recipe_change_requires_migration(self):
        changed=copy.deepcopy(self.recipe);changed['review_policy']='changed_quality_semantics'
        with self.assertRaises(ValueError):SupplyControl(self.root/'state.sqlite',changed)

    def test_runtime_budget_change_preserves_counter(self):
        batch,p,_=self.fixture('a');self.ledger.release(batch,p)
        changed=copy.deepcopy(self.recipe);changed['api_budget_cny']=100
        other=SupplyControl(self.root/'state.sqlite',changed)
        try:self.assertEqual(other.status()['unique_ready_train_body_tokens'],7)
        finally:other.close()

    def test_choose_domain_only_from_available_sources(self):
        self.assertEqual(self.ledger.next_domain({'math'}),'math')
        self.assertIsNone(self.ledger.next_domain(set()))


if __name__ == '__main__':unittest.main()
