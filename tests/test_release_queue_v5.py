import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from p0_release_queue_v5 import validate_job
from small_agent.curation.supply_control import file_sha
class QueueTests(unittest.TestCase):
 def fixture(self,r):
  source=r/'input.parquet';source.write_bytes(b'fixture');return dict(version=5,processor='python_internal_pretraining_v5',source_id='github_code_clean_supply_v4',candidates=str(source),candidate_sha256=file_sha(source),source_audit=str(r/'calibration'),output=str(r/'releases'/'job'))
 def test_binding_and_no_shell_execution_contract(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);j=self.fixture(r);self.assertIsNone(validate_job(j,r/'releases'));j['command']='echo unsafe'
   with self.assertRaises(ValueError):validate_job(j,r/'releases')
 def test_unknown_source_is_hold(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);j=self.fixture(r);j['source_id']='unknown';self.assertEqual(validate_job(j,r/'releases'),'unregistered_source_reference_hold')
 def test_outside_output_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);j=self.fixture(r);j['output']='/tmp/elsewhere'
   with self.assertRaises(ValueError):validate_job(j,r/'releases')
 def test_changed_candidate_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);j=self.fixture(r);Path(j['candidates']).write_bytes(b'changed')
   with self.assertRaises(ValueError):validate_job(j,r/'releases')
if __name__=='__main__':unittest.main()

class SourceCohortTests(unittest.TestCase):
 def test_same_source_name_new_revision_is_not_auto_approved(self):
  import pyarrow as pa,pyarrow.parquet as pq
  from p0_release_queue_v5 import validate_source_cohort
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);audit=r/'audit';audit.mkdir();inp=audit/'input.jsonl';inp.write_text(json.dumps(dict(source_revision='old'))+'\n')
   (audit/'SAMPLE_PLAN.json').write_text(json.dumps(dict(frozen_files={'input.jsonl':file_sha(inp)},request_identity={'source_id':'github_code_clean_supply_v4'},source_complete_in_plan=True,input_source_files=[{'source_file':'data/a.parquet'}])))
   p=r/'new.parquet';pq.write_table(pa.Table.from_pylist([dict(source_id='github_code_clean_supply_v4',source_revision='new',source_file='data/a.parquet')]),p)
   job=dict(source_audit=str(audit),source_id='github_code_clean_supply_v4',candidates=str(p));self.assertEqual(validate_source_cohort(job),'new_revision_or_uncalibrated_shard_hold')
 def test_new_shard_is_not_auto_approved(self):
  import pyarrow as pa,pyarrow.parquet as pq
  from p0_release_queue_v5 import validate_source_cohort
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);audit=r/'audit';audit.mkdir();inp=audit/'input.jsonl';inp.write_text(json.dumps(dict(source_revision='old'))+'\n')
   (audit/'SAMPLE_PLAN.json').write_text(json.dumps(dict(frozen_files={'input.jsonl':file_sha(inp)},request_identity={'source_id':'github_code_clean_supply_v4'},source_complete_in_plan=True,input_source_files=[{'source_file':'data/a.parquet'}])))
   p=r/'new.parquet';pq.write_table(pa.Table.from_pylist([dict(source_id='github_code_clean_supply_v4',source_revision='old',source_file='data/b.parquet')]),p)
   job=dict(source_audit=str(audit),source_id='github_code_clean_supply_v4',candidates=str(p));self.assertEqual(validate_source_cohort(job),'new_revision_or_uncalibrated_shard_hold')
