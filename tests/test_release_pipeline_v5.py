"""Engineering fixture: mocked quality decisions, REAL pack/ledger/recovery tests.

Passing this test never means real-corpus calibration or natural quality passed.
"""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import p0_release_code_batch_v5 as runner
from small_agent.curation.release_gates_v5 import sha,ledger_snapshot
from small_agent.curation.supply_control import file_sha,SupplyControl

class PipelineTests(unittest.TestCase):
 def setup_inputs(self,r):
  import pyarrow as pa,pyarrow.parquet as pq
  from tokenizers import Tokenizer,models,pre_tokenizers
  t=Tokenizer(models.WordLevel({'<pad>':0,'<unk>':1,'<eos>':2,'a':3,'b':4,'c':5},unk_token='<unk>'));t.pre_tokenizer=pre_tokenizers.Whitespace();tok=r/'tokenizer.json';t.save(str(tok))
  rows=[]
  for split,w,n in [('train','a',512),('validation','b',150),('test','c',150)]:
   text=' '.join([w]*n);rows.append(dict(text=text,normalized_sha256=sha(text),source_id='fixture',source_locator=split,source_revision='a'*40,split=split,language='Python',own_text_tokens=n,family_id=split,record_license='mit',policy_bucket='record_license_whitelisted'))
  candidates=r/'candidates.parquet';pq.write_table(pa.Table.from_pylist(rows),candidates)
  recipe=json.loads((ROOT/'configs/curation/base_1b_supply_20b_v4.json').read_text());recipe['tokenizer_sha256']=file_sha(tok);rp=r/'recipe.json';rp.write_text(json.dumps(recipe));db=r/'supply.sqlite';c=SupplyControl(db,recipe);c.close()
  idx=r/'inventory.json';idx.write_text(json.dumps(dict(stage='complete_registered_inventory_index',identity=dict(ledger=ledger_snapshot(db)),files=[],unique_documents=0)))
  ef=r/'eval.parquet';pq.write_table(pa.Table.from_pylist([dict(question='completely unrelated frozen evaluation reference question',answer='answer')]),ef);er=r/'eval.json';er.write_text(json.dumps(dict(training_eligible=False,files=[dict(task='gsm8k_test',path=str(ef),sha256=file_sha(ef))])))
  policy=json.loads((ROOT/'configs/curation/p0_release_gates_v5.json').read_text());policy['sequence_length']=128;policy['target_shard_tokens']=1024;policy['evaluation']['required_tasks']=['gsm8k_test'];pp=r/'policy.json';pp.write_text(json.dumps(policy))
  out=r/'out';argv=['runner','--candidates',str(candidates),'--source-audit',str(r/'source'),'--reference',str(r/'ref'),'--reference-batch',str(r/'refbatch'),'--batch-audit',str(r/'batchaudit'),'--eval-registry',str(er),'--inventory-index',str(idx),'--database',str(db),'--tokenizer',str(tok),'--output',str(out),'--recipe',str(rp),'--policy',str(pp)]
  quality=dict(status='passed',source_id='fixture',approved_strata=['short/python'],known_audit_bad_body_sha256=[])
  rolling=dict(status='passed',parquet_sha256=file_sha(candidates),known_bad_body_sha256=[])
  return argv,out,db,recipe,quality,rolling
 def test_real_pack_idempotence_and_atomic_release_intent_recovery(self):
  with tempfile.TemporaryDirectory() as d:
   argv,out,db,recipe,quality,rolling=self.setup_inputs(Path(d));realwrite=runner.write
   def interrupted(path,value):
    if Path(path).name=='READY_INTERNAL.json':raise RuntimeError('controlled interruption after ledger commit')
    return realwrite(path,value)
   with patch.object(sys,'argv',argv),patch.object(runner,'quality_scope',return_value=quality),patch.object(runner,'rolling_audit',return_value=rolling),patch.object(runner,'write',side_effect=interrupted):
    with self.assertRaises(RuntimeError):runner.main()
   self.assertFalse((out/'READY_INTERNAL.json').exists());self.assertTrue((out/'RELEASE_INTENT.json').exists())
   c=SupplyControl(db,recipe)
   try:self.assertEqual(c.status()['unique_ready_train_body_tokens'],512)
   finally:c.close()
   with patch.object(sys,'argv',argv):runner.main();runner.main()
   self.assertTrue((out/'READY_INTERNAL.json').exists());c=SupplyControl(db,recipe)
   try:self.assertEqual(c.status()['unique_ready_train_body_tokens'],512);self.assertEqual(c.db.execute('SELECT COUNT(*) FROM releases').fetchone()[0],1)
   finally:c.close()
 def test_quality_hold_produces_no_pack_or_ready_tokens(self):
  with tempfile.TemporaryDirectory() as d:
   argv,out,db,recipe,quality,rolling=self.setup_inputs(Path(d));quality=dict(status='hold',holds=['missing_independent_reference'])
   with patch.object(sys,'argv',argv),patch.object(runner,'quality_scope',return_value=quality),patch.object(runner,'rolling_audit',return_value=rolling):runner.main()
   self.assertTrue((out/'HOLD.json').exists());self.assertFalse((out/'tokenized').exists());c=SupplyControl(db,recipe)
   try:self.assertEqual(c.status()['unique_ready_train_body_tokens'],0)
   finally:c.close()
if __name__=='__main__':unittest.main()
