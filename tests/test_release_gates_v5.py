"""Synthetic gate tests: no corpus-quality conclusions from these fixtures."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.release_gates_v5 import *
from small_agent.curation.supply_control import SupplyControl
POLICY=json.loads((ROOT/'configs/curation/p0_release_gates_v5.json').read_text())

class ReleaseGateTests(unittest.TestCase):
 def fixture(self,root):
  import pyarrow as pa,pyarrow.parquet as pq
  recipe=json.loads((ROOT/'configs/curation/base_1b_supply_20b_v4.json').read_text());db=root/'supply.sqlite';ctl=SupplyControl(db,recipe)
  text='def reusable_calculation(values):\n    total = 0\n    for element in values:\n        total += element * element\n    return total\n';rows=[dict(text=text,normalized_sha256=sha(text),split='validation')]
  source=root/'reference.parquet';pq.write_table(pa.Table.from_pylist(rows),source)
  with ctl.db:
   ctl.db.execute('INSERT INTO batches(batch_id,domain,input_binding,status,created) VALUES (?,?,?,?,?)',('old','code','{}','legacy_internal',0))
   ctl.db.execute('INSERT INTO documents VALUES (?,?,?,?,?)',(sha(text),'old','validation','code',50))
  ctl.close();receipt=root/'legacy.json';receipt.write_text(json.dumps(dict(evidence_files=[dict(path=str(source),sha256=file_sha(source))])))
  idx=build_inventory_index(db,receipt,root/'index');return db,source,idx,text,receipt
 def test_full_coverage_and_index_resume(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);db,source,idx,text,receipt=self.fixture(root)
   self.assertEqual(json.loads(idx.read_text())['unique_documents'],1)
   self.assertEqual(build_inventory_index(db,receipt,root/'index'),idx)
 def test_reference_tamper_blocks(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);db,source,idx,text,receipt=self.fixture(root);source.write_bytes(b'changed')
   with self.assertRaises(ValueError):build_inventory_index(db,receipt,root/'index')
 def test_global_exact_validation_overlap_removed(self):
  with tempfile.TemporaryDirectory() as t:
   db,source,idx,text,_=self.fixture(Path(t));rows=[dict(text=text,normalized_sha256=sha(text),split='train')]
   kept,report=global_near(rows,idx,POLICY['near'],db);self.assertEqual(kept,[]);self.assertEqual(report['exclusions'][0]['reason'],'global_exact_duplicate')
 def test_global_near_whitespace_different_body_removed(self):
  with tempfile.TemporaryDirectory() as t:
   db,source,idx,text,_=self.fixture(Path(t));new=text.replace('    ','  ')
   kept,r=global_near([dict(text=new,normalized_sha256=sha(new),split='train')],idx,POLICY['near'],db)
   self.assertEqual(kept,[]);self.assertEqual(r['exclusions'][0]['reason'],'global_near_duplicate')
 def test_stale_ledger_blocks(self):
  with tempfile.TemporaryDirectory() as t:
   db,source,idx,text,_=self.fixture(Path(t));d=sqlite3.connect(db);d.execute('INSERT INTO documents VALUES (?,?,?,?,?)',(sha('other'),'old','train','code',10));d.commit();d.close()
   with self.assertRaises(ValueError):global_near([],idx,POLICY['near'],db)
 def test_missing_inventory_documents_blocks(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);db,source,idx,text,receipt=self.fixture(root);d=sqlite3.connect(db);d.execute('INSERT INTO documents VALUES (?,?,?,?,?)',(sha('missing'),'old','train','code',10));d.commit();d.close()
   with self.assertRaises(ValueError):build_inventory_index(db,receipt,root/'index')
 def test_casefold_only_affects_comparison(self):
  self.assertEqual(lexical('def Apple(): return 1'),lexical('def apple(): return 1'))
  self.assertNotEqual(sha('def Apple(): return 1'),sha('def apple(): return 1'))
 def test_short_shingle_nonempty(self):self.assertEqual(len(lexical('x+y')),1)
 def test_eval_full_and_ngram_hits_and_unrelated_kept(self):
  import pyarrow as pa,pyarrow.parquet as pq
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);source=root/'eval.parquet';question=' '.join('uniqueword'+str(i) for i in range(25));pq.write_table(pa.Table.from_pylist([dict(question=question,answer='answer')]),source)
   registry=root/'registry.json';registry.write_text(json.dumps(dict(training_eligible=False,files=[dict(task='gsm8k_test',path=str(source),sha256=file_sha(source))])))
   policy=copy.deepcopy(POLICY['evaluation']);policy['required_tasks']=['gsm8k_test'];screen=EvalScreen(registry,policy)
   self.assertEqual(screen.matches('preface '+question+' footer'),['gsm8k_test'])
   self.assertEqual(screen.matches(' '.join(normalized_words(question)[3:20])),['gsm8k_test'])
   rows=[dict(text='unrelated custom algorithm text',normalized_sha256=sha('u')),dict(text=question,normalized_sha256=sha(question))]
   kept,r=screen.filter(rows);self.assertEqual(len(kept),1);self.assertEqual(len(r['exclusions']),1)
 def test_eval_missing_registry_blocks(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'registry.json';p.write_text(json.dumps(dict(training_eligible=False,files=[])))
   with self.assertRaises(ValueError):EvalScreen(p,POLICY['evaluation'])
if __name__=='__main__':unittest.main()

class QualityGateTests(unittest.TestCase):
 def test_missing_reference_blocks_instead_of_agreement_approval(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'ref.json';p.write_text(json.dumps(dict(training_eligible=False,independently_verified_positive_code_tests=False)))
   # A fake reference carrying a model vote is not independently test-verified.
   (Path(t)/'labels.json').write_text('[]')
   with self.assertRaises(ValueError):quality_scope(Path(t),p,Path(t),POLICY['quality'])
 def test_no_execution_of_unadmitted_reference_code(self):
  import p0_quality_reference_v5 as refs
  for text in ['import os\nos.remove("x")','open("x")','x.__class__','getattr(x,"z")','class A: pass']:
   self.assertFalse(refs.admitted(text))
  self.assertTrue(refs.admitted('def square(x):\n    return x*x\nassert square(3)==9'))
 def test_external_reference_execution(self):
  import subprocess
  r=subprocess.run([sys.executable,str(ROOT/'scripts/p0_quality_reference_v5.py'),'--execute-one'],input=json.dumps(dict(code='def square(x):\n    return x*x\n',test_imports=[],test_list=['assert square(3)==9'])),text=True,capture_output=True)
  self.assertEqual(r.returncode,0);self.assertEqual(r.stdout.strip(),'REFERENCE_TESTS_PASSED')
 def test_failing_external_reference_tests_are_not_certified(self):
  import subprocess
  r=subprocess.run([sys.executable,str(ROOT/'scripts/p0_quality_reference_v5.py'),'--execute-one'],input=json.dumps(dict(code='def square(x):\n    return x\n',test_imports=[],test_list=['assert square(3)==9'])),text=True,capture_output=True)
  self.assertNotEqual(r.returncode,0)

class ScopedRollingTests(unittest.TestCase):
 def test_failed_stratum_is_not_silently_approved(self):
  from unittest.mock import patch
  class DB:
   def execute(self,_):return annotations
  class FakeBatch:
   def __init__(self,path):
    self.docs=docs;self.db=DB();self.policy=dict(minimum_confidence=.85,minimum_scores=dict(quality=3,format_integrity=3,completeness=1,educational_value=1));self.manifest_hash='manifest'
   def state_fingerprint(self):return 'state'
   def close(self):pass
  docs={};annotations=[]
  for h in ['short/python','medium/python','long/python']:
   for i in range(8):
    key=sha(h+str(i));docs[key]=dict(document_id=key,sampling_stratum=h)
    if h=='long/python' and i<2:continue
    annotations.append((key,json.dumps(dict(decision='keep',confidence=.99,quality=4,format_integrity=4,completeness=4,educational_value=4,findings=[]))))
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);source=r/'source.parquet';source.write_bytes(b'fixture');inp=r/'input.jsonl';inp.write_text('fixture')
   (r/'SAMPLE_PLAN.json').write_text(json.dumps(dict(parquet_path=str(source),parquet_sha256=file_sha(source),input_sha256=file_sha(inp),source_manifest_sha256='manifest',review_state_sha256='state',per_stratum=8,population={h:100 for h in ['short/python','medium/python','long/python']})))
   with patch('small_agent.curation.store.Batch',FakeBatch):report=rolling_audit(r,r/'source')
   self.assertEqual(report['approved_strata'],['medium/python','short/python']);self.assertEqual(report['held_strata'],['long/python']);self.assertEqual(report['status'],'passed');self.assertEqual(report['strata']['long/python']['valid'],6)
