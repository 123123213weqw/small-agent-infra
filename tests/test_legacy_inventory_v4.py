"""Synthetic migration tests; these do not certify corpus quality."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from small_agent.curation.supply_control import SupplyControl,digest,file_sha
CONFIG=Path(__file__).resolve().parents[1]/'configs/curation/base_1b_supply_20b_v4.json'

class LegacyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.recipe=json.loads(CONFIG.read_text());self.c=SupplyControl(self.root/'state.sqlite',self.recipe)
    def tearDown(self):self.c.close();self.tmp.cleanup()
    def fixture(self,rows=None,name='a'):
        rows=rows or [dict(body_sha256=digest('a'),family_id=digest('f'),raw_domain='math_en',domain='math',split='train',body_tokens=7),
                      dict(body_sha256=digest('b'),family_id=digest('g'),raw_domain='math_en',domain='math',split='validation',body_tokens=3)]
        index=self.root/(name+'.jsonl');index.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        tokens={};docs={}
        for r in rows:
            k=r['raw_domain']+'/'+r['split'];tokens[k]=tokens.get(k,0)+r['body_tokens'];docs[k]=docs.get(k,0)+1
        pack=self.root/(name+'.pack.json');pack.write_text(json.dumps(dict(stage='physical_pack_hashes_verified',
            tokenizer_sha256=self.recipe['tokenizer_sha256'],manifest_sha256=digest(name),
            body_tokens_by_raw_domain_split=tokens,documents_by_raw_domain_split=docs)))
        r=dict(stage='legacy_inventory_audit_internal_only',tokenizer_sha256=self.recipe['tokenizer_sha256'],
               recipe_sha256=self.c.recipe_sha,pack_manifest_sha256=digest(name),pack_verification_path=str(pack),
               document_index_path=str(index),input_identity={'fixture_only':name},
               body_tokens_by_raw_domain_split=tokens,documents_by_raw_domain_split=docs,
               evidence_files=[{'path':str(p),'sha256':file_sha(p)} for p in [index,pack]])
        p=self.root/(name+'.receipt.json');p.write_text(json.dumps(r));return p,rows
    def test_counts_train_only_and_does_not_claim_new_ready(self):
        p,_=self.fixture();s=self.c.import_legacy(p)
        self.assertEqual(s['accounted_unique_train_body_tokens'],7)
        self.assertEqual(s['unique_ready_train_body_tokens'],0)
        self.assertEqual(s['legacy_internal_by_domain']['math'],7)
        self.assertEqual(s['remaining_by_domain']['math'],2000000000-7)
    def test_repeat_import_is_idempotent(self):
        p,_=self.fixture();self.c.import_legacy(p);self.c.import_legacy(p)
        self.assertEqual(self.c.status()['legacy_imports'],1)
    def test_import_not_pending_batch(self):
        p,_=self.fixture();self.c.import_legacy(p)
        self.c.register('code',{'raw':1});self.c.register('math',{'raw':2})
    def test_changed_index_rejected(self):
        p,_=self.fixture();(self.root/'a.jsonl').write_text('changed')
        with self.assertRaises(ValueError):self.c.import_legacy(p)
        self.assertEqual(self.c.status()['accounted_unique_train_body_tokens'],0)
    def test_exact_duplicates_count_once(self):
        _,rows=self.fixture();p,_=self.fixture([rows[0],rows[0]],name='duplicates')
        self.c.import_legacy(p);self.assertEqual(self.c.status()['accounted_unique_train_body_tokens'],7)
    def test_cross_split_duplicate_rolls_back(self):
        _,rows=self.fixture();rows[1]['body_sha256']=rows[0]['body_sha256']
        p,_=self.fixture(rows,name='conflict')
        with self.assertRaises(ValueError):self.c.import_legacy(p)
        self.assertEqual(self.c.status()['legacy_imports'],0)
        self.assertEqual(self.c.db.execute('SELECT COUNT(*) FROM documents').fetchone()[0],0)
    def test_family_split_conflict_rolls_back(self):
        _,rows=self.fixture();rows[1]['family_id']=rows[0]['family_id'];p,_=self.fixture(rows,name='family')
        with self.assertRaises(ValueError):self.c.import_legacy(p)
        self.assertEqual(self.c.status()['accounted_unique_train_body_tokens'],0)
    def test_physical_pack_counter_mismatch_rejected(self):
        p,_=self.fixture();r=json.loads(p.read_text());r['body_tokens_by_raw_domain_split']['math_en/train']=8;p.write_text(json.dumps(r))
        with self.assertRaises(ValueError):self.c.import_legacy(p)

if __name__=='__main__':unittest.main()
