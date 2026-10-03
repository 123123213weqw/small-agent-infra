"""Observer fixtures only, never evidence that mock packs are train-approved."""
import json,sqlite3,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import p0_production_delivery_v12 as delivery

class DeliveryTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);r=Path(self.temp.name);root=r/'stock';root.mkdir()
  with sqlite3.connect(r/'ledger.sqlite') as d:
   d.executescript("CREATE TABLE documents(body_sha TEXT,batch_id TEXT,split TEXT,domain TEXT,tokens INTEGER);CREATE TABLE legacy_imports(batch_id TEXT,receipt_sha TEXT);")
   d.execute("INSERT INTO legacy_imports VALUES ('legacy','sha')");d.executemany('INSERT INTO documents VALUES (?,?,?,?,?)',[('old','legacy','train','general_en',1000),('held','legacy','validation','general_en',20),('new','new','train','code',100)])
  with sqlite3.connect(root/'stock.sqlite') as d:
   d.execute('CREATE TABLE files(state TEXT,next_pos INTEGER)');d.executemany('INSERT INTO files VALUES (?,?)',[('complete',30),('scanning',20)])
  self.cfg=dict(root=str(root),output=str(r/'delivery'),database=str(r/'ledger.sqlite'))
  self.cat=dict(packs=[dict(batch_id='fixture')],unique_train_body_tokens=100)
 def run_fixture(self,n=100):
  with patch.object(delivery,'catalog',return_value=dict(self.cat,unique_train_body_tokens=n)),patch.object(delivery,'check_documents',return_value=(dict(train=n,validation=5,test=3),dict(code=n),10)):
   return delivery.deliver(self.cfg)
 def test_only_verified_new_stock_and_legacy_separate(self):
  r=self.run_fixture();self.assertEqual(r['verified_ready_train_body_tokens'],100);self.assertEqual(r['inherited_legacy_train_body_tokens'],1000);self.assertEqual(r['inherited_plus_verified_ready_train_body_tokens'],1100);self.assertFalse(r['training_started']);self.assertEqual(r['cpu_rows_processed'],50)
 def test_baseline_is_immutable_and_delta_counts_train_only(self):
  a=self.run_fixture();p=Path(self.cfg['output'])/'BASELINE.json';before=p.read_bytes();b=self.run_fixture(150)
  self.assertEqual(p.read_bytes(),before);self.assertEqual(b['new_ready_since_baseline'],50);self.assertEqual(b['new_by_domain']['code'],50)
 def test_empty_candidates_never_replace_ready(self):
  with patch.object(delivery,'catalog',return_value=dict(packs=[],unique_train_body_tokens=0)):
   with self.assertRaises(ValueError):delivery.deliver(self.cfg)
  self.assertFalse((Path(self.cfg['output'])/'catalog.json').exists())
 def test_failed_physical_catalog_does_not_refresh_delivery(self):
  self.run_fixture();p=Path(self.cfg['output'])/'latest.json';before=p.read_bytes()
  with patch.object(delivery,'catalog',side_effect=ValueError('physical SHA mismatch')):
   with self.assertRaises(ValueError):delivery.deliver(self.cfg)
  self.assertEqual(p.read_bytes(),before)
 def test_counter_mismatch_is_rejected(self):
  with patch.object(delivery,'catalog',return_value=self.cat),patch.object(delivery,'check_documents',return_value=(dict(train=99),dict(code=99),10)):
   with self.assertRaises(ValueError):delivery.deliver(self.cfg)
 def test_old_catalog_generation_remains_bound_after_update(self):
  a=self.run_fixture();b=self.run_fixture(200)
  self.assertNotEqual(a['catalog_path'],b['catalog_path']);self.assertEqual(delivery.file_sha(Path(a['catalog_path'])),a['catalog_sha256'])
 def test_api_hold_and_cohort_holds_visible(self):
  root=Path(self.cfg['root']);(root/'API_HOLD.json').write_text('{}');(root/'holds').mkdir();(root/'holds/x').write_text(json.dumps(dict(cohort='code/c',failure_type='ValueError',reason='validation unavailable')))
  r=self.run_fixture();self.assertTrue(r['api_campaign_hold']);self.assertEqual(r['cohort_holds'][0]['cohort'],'code/c')
if __name__=='__main__':unittest.main()
