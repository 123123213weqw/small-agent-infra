"""Synthetic engineering only; no natural corpus or licensing certification."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import p0_release_math_batch_v6 as runner
from test_release_pipeline_v5 import PipelineTests
from small_agent.curation.supply_control import file_sha,SupplyControl
from small_agent.curation.stock_v6 import DiskNearIndex
from p0_stock_queue_v6 import completed_worker

class MathPipelineTests(unittest.TestCase):
 def fixture(self,r):
  import pyarrow as pa,pyarrow.parquet as pq
  argv,out,db,recipe,quality,rolling=PipelineTests().setup_inputs(r)
  rows=pq.read_table(r/'candidates.parquet').to_pylist()
  for row in rows:row.update(language='en',domain='math',policy_bucket='source_terms_hold',record_license='odc-by')
  pq.write_table(pa.Table.from_pylist(rows),r/'candidates.parquet');rolling['parquet_sha256']=file_sha(r/'candidates.parquet')
  quality.update(approved_language='en',approved_strata=['short/en'])
  terms=r/'terms.json';terms.write_text(json.dumps(dict(status='source_level_internal_scope_authorized',internal_training_eligible=True,distribution_eligible=False,source_id='fixture',revision='a'*40,scope='internal only, not individual content certification')))
  binding=r/'candidate_binding.json';binding.write_text(json.dumps(dict(candidate_sha256=file_sha(r/'candidates.parquet'))))
  empty=r/'legacy.json';empty.write_text(json.dumps(dict(evidence_files=[])));idx=DiskNearIndex(r/'diskidx')
  try:descriptor=idx.update(db,empty,r/'cache')
  finally:idx.close()
  argv[argv.index('--inventory-index')+1]=str(descriptor)
  argv+=['--source-terms',str(terms),'--candidate-binding',str(binding)]
  return argv,out,db,recipe,quality,rolling
 def test_real_math_pack_atomic_recovery_and_idempotence(self):
  with tempfile.TemporaryDirectory() as d:
   argv,out,db,recipe,quality,rolling=self.fixture(Path(d));realwrite=runner.write
   def interrupted(path,value):
    if Path(path).name=='READY_INTERNAL.json':raise RuntimeError('controlled after ledger commit')
    return realwrite(path,value)
   with patch.object(sys,'argv',argv),patch.object(runner,'validate_source_cohort',return_value=None),patch.object(runner,'quality_scope',return_value=quality),patch.object(runner,'rolling_audit',return_value=rolling),patch.object(runner,'write',side_effect=interrupted):
    with self.assertRaises(RuntimeError):runner.main()
   self.assertFalse((out/'READY_INTERNAL.json').exists());self.assertTrue((out/'RELEASE_INTENT.json').exists())
   with patch.object(sys,'argv',argv),patch.object(runner,'validate_source_cohort',return_value=None):runner.main();runner.main()
   c=SupplyControl(db,recipe)
   try:self.assertEqual(c.status()['unique_ready_train_body_tokens'],512);self.assertEqual(c.db.execute('SELECT COUNT(*) FROM releases').fetchone()[0],1)
   finally:c.close()
 def test_math_quality_hold_never_creates_pack(self):
  with tempfile.TemporaryDirectory() as d:
   argv,out,db,recipe,quality,rolling=self.fixture(Path(d))
   with patch.object(sys,'argv',argv),patch.object(runner,'validate_source_cohort',return_value=None),patch.object(runner,'quality_scope',return_value=dict(status='hold',holds=['controlled_arithmetic_false_keep'])),patch.object(runner,'rolling_audit',return_value=rolling):runner.main()
   self.assertTrue((out/'HOLD.json').exists());self.assertFalse((out/'tokenized').exists())
 def test_unapproved_source_terms_fail_closed(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);argv,out,*_=self.fixture(r);terms=json.loads((r/'terms.json').read_text());terms['distribution_eligible']=True;(r/'terms.json').write_text(json.dumps(terms))
   with patch.object(sys,'argv',argv),self.assertRaises(ValueError):runner.main()
   self.assertFalse((out/'tokenized').exists())
 def test_paid_audit_stop_not_automatically_retried(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);self.assertFalse(completed_worker(b));w=b/'worker/deepseek';w.mkdir(parents=True)
   (w/'ADAPTIVE_STATUS.json').write_text(json.dumps(dict(stop_reason='unknown_transport_or_fatal_response')))
   with self.assertRaises(ValueError):completed_worker(b)
if __name__=='__main__':unittest.main()
