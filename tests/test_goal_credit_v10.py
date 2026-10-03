"""Legacy credit changes the goal, not the factual/quality status or body uniqueness."""
import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.goal_credit_v10 import goal_balance,GoalQuotaControl
from small_agent.curation.supply_control import SupplyControl
RECIPE=json.loads((ROOT/'configs/curation/base_1b_supply_20b_v4.json').read_text())
OLD={'general_en':824688270,'general_zh':210701974,'code':3443535,'math':597716}


class CreditTests(unittest.TestCase):
    def test_real_old_inventory_and_new_ready_sum_once(self):
        s=goal_balance(RECIPE,{'code':33463612},OLD,{'math':10064401415})
        self.assertEqual(sum(s['legacy_inherited_internal_by_domain'].values()),1039431495)
        self.assertEqual(s['unique_goal_credited_train_body_tokens'],1072895107)
        self.assertEqual(sum(s['remaining_goal_by_domain'].values()),18927104893)
        self.assertFalse(s['legacy_quality_recertified'])
    def test_domain_gaps_use_old_and_new_separately(self):
        s=goal_balance(RECIPE,{'code':33463612},OLD)
        self.assertEqual(s['remaining_goal_by_domain']['general_en'],6175311730)
        self.assertEqual(s['remaining_goal_by_domain']['general_zh'],4789298026)
        self.assertEqual(s['remaining_goal_by_domain']['math'],1999402284)
        self.assertEqual(s['target_weights']['math'],.1)
    def test_whole_goal_met_without_faking_new_ready(self):
        s=goal_balance(RECIPE,{},RECIPE['targets_by_domain'])
        self.assertTrue(s['target_met']);self.assertEqual(sum(s['ready_by_domain'].values()),0)
        self.assertFalse(s['legacy_quality_recertified'])
    def test_supply_quota_preserves_existing_shared_recipe_and_no_reimport(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'db';c=GoalQuotaControl(path,RECIPE)
            try:
                with c.db:
                    c.db.execute("INSERT INTO batches(batch_id,domain,input_binding,status,created) VALUES ('old','legacy','{}','legacy_internal',0)")
                    c.db.execute("INSERT INTO legacy_imports VALUES ('old','old')")
                    c.db.execute("INSERT INTO documents VALUES (?, 'old','train','math',597716)",('a'*64,))
                self.assertEqual(c.status()['remaining_by_domain']['math'],1999402284)
                self.assertEqual(c.status()['unique_ready_train_body_tokens'],0)
                identity=c.recipe_sha
            finally:c.close()
            old=SupplyControl(path,RECIPE)
            try:self.assertEqual(old.recipe_sha,identity);self.assertEqual(old.db.execute('SELECT COUNT(*) FROM documents').fetchone()[0],1)
            finally:old.close()

from unittest.mock import patch
from small_agent.curation.supply_control import file_sha,recipe_identity
from small_agent.curation.release_gates_v5 import write
from p0_legacy_goal_credit_v10 import verify
from p0_full_stock_status_v10 import report
import sqlite3

class BoundCreditTests(unittest.TestCase):
    def fixture(self,r):
        recipe=r/'recipe.json';write(recipe,RECIPE)
        shard=r/'train.bin';shard.write_bytes(b'physical-test-pack')
        manifest=r/'manifest.json';write(manifest,dict(fixture=True))
        physical=r/'physical.json'
        counts={'general_en/train':100,'general_en/validation':20,'math_en/train':10}
        write(physical,dict(stage='physical_pack_hashes_verified',manifest_path=str(manifest),manifest_sha256=file_sha(manifest),tokenizer_sha256=RECIPE['tokenizer_sha256'],body_tokens_by_raw_domain_split=counts,verified_files=[dict(path=str(shard),sha256=file_sha(shard),bytes=shard.stat().st_size)]))
        receipt=r/'legacy.json';write(receipt,dict(stage='legacy_inventory_audit_internal_only',recipe_sha256=recipe_identity(RECIPE),tokenizer_sha256=RECIPE['tokenizer_sha256'],pack_verification_path=str(physical),pack_manifest_sha256=file_sha(manifest),body_tokens_by_raw_domain_split=counts,evidence_files=[dict(path=str(physical),sha256=file_sha(physical))]))
        db=r/'supply.sqlite';c=SupplyControl(db,RECIPE)
        try:
            with c.db:
                c.db.execute("INSERT INTO batches(batch_id,domain,input_binding,status,created) VALUES ('old','legacy','{}','legacy_internal',0)")
                c.db.execute("INSERT INTO legacy_imports VALUES ('old',?)",(file_sha(receipt),))
                for h,split,domain,n in [('a'*64,'train','general_en',100),('b'*64,'train','math',10),('c'*64,'validation','general_en',20)]:
                    c.db.execute('INSERT INTO documents VALUES (?,?,?,?,?)',(h,'old',split,domain,n))
        finally:c.close()
        proof=r/'credit.json';verify(db,receipt,recipe,proof)
        original=r/'original';original.mkdir();d=sqlite3.connect(original/'stock.sqlite');d.execute('CREATE TABLE records(domain TEXT,n INTEGER,split TEXT)');d.execute("INSERT INTO records VALUES ('math',10000,'train')");d.commit();d.close()
        cfg=dict(recipe=str(recipe),database=str(db),stock=str(r),original_stock=str(original),legacy_goal_credit_receipt=str(proof),goal_credit_policy='inherited_internal_plus_new_READY',shared_campaign_budget_cny=1000,maximum_concurrency=2500,replenishment_preferences={d:[] for d in RECIPE['targets_by_domain']},missing_domain_action='WAIT')
        return cfg,db,receipt,recipe,proof,shard

    def test_physical_proof_and_report_credit_train_only_without_ready(self):
        with tempfile.TemporaryDirectory() as t:
            cfg,db,receipt,recipe,proof,_=self.fixture(Path(t))
            before=sqlite3.connect(db).execute('SELECT COUNT(*) FROM documents').fetchone()[0]
            self.assertEqual(verify(db,receipt,recipe,proof)['credited_train_body_tokens'],110)
            s=report(cfg);self.assertEqual(s['unique_goal_credited_train_body_tokens'],110)
            self.assertEqual(sum(s['ready_by_domain'].values()),0)
            self.assertEqual(s['remaining_goal_by_domain']['general_en'],6999999900)
            self.assertEqual(sqlite3.connect(db).execute('SELECT COUNT(*) FROM documents').fetchone()[0],before)
    def test_changed_physical_shard_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            cfg,db,receipt,recipe,proof,shard=self.fixture(Path(t));shard.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify(db,receipt,recipe,proof)
    def test_per_domain_ledger_mismatch_rejected_even_same_total(self):
        with tempfile.TemporaryDirectory() as t:
            cfg,db,receipt,recipe,proof,_=self.fixture(Path(t));d=sqlite3.connect(db);d.execute("UPDATE documents SET domain='code' WHERE domain='math'");d.commit();d.close()
            with self.assertRaises(ValueError):verify(db,receipt,recipe,proof)
            with self.assertRaises(ValueError):report(cfg)
    def test_ready_relabel_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            cfg,db,receipt,recipe,proof,_=self.fixture(Path(t));d=sqlite3.connect(db)
            d.execute("INSERT INTO releases VALUES ('old','bad',110)");d.commit();d.close()
            with self.assertRaises(ValueError):verify(db,receipt,recipe,proof)

class InheritedPhysicalScopeTests(unittest.TestCase):
    def test_explicit_inheritance_does_not_claim_physical_rehash(self):
        with tempfile.TemporaryDirectory() as t:
            cfg,db,receipt,recipe,proof,shard=BoundCreditTests().fixture(Path(t));shard.unlink()
            inherited=verify(db,receipt,recipe,Path(t)/'inherited-credit.json',True)
            self.assertFalse(inherited['physical_files_rehashed_this_run'])
            self.assertEqual(inherited['physical_verification_scope'],'inherited_hash_bound_L40_report')
            self.assertEqual(inherited['credited_train_body_tokens'],110)

if __name__=='__main__':unittest.main()
