#!/usr/bin/env python3
"""Automatic measured internal release; any unresolved gate stops READY publication.

Consumes a frozen candidate Parquet. API audits are prepared/scored separately so
unknown paid attempts cannot be hidden by pipeline retries. No training or HF push.
"""
import argparse,fcntl,json,subprocess,sys,time,os
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.release_gates_v5 import *
from small_agent.curation.supply_control import SupplyControl,digest,STAGES

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('candidates','source-audit','reference','reference-batch','batch-audit','eval-registry','inventory-index','database','tokenizer','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--recipe',type=Path,default=ROOT/'configs/curation/base_1b_supply_20b_v4.json');p.add_argument('--policy',type=Path,default=ROOT/'configs/curation/p0_release_gates_v5.json')
 a=p.parse_args();policy=json.loads(a.policy.read_text());recipe=json.loads(a.recipe.read_text());a.output.mkdir(parents=True,exist_ok=True)
 import pyarrow as pa,pyarrow.parquet as pq
 from tokenizers import Tokenizer
 from small_agent.data.sharded_pack import TokenizedDocument,pack_sharded
 from small_agent.data.sharded_format import load_sharded_manifest
 if file_sha(a.tokenizer)!=recipe['tokenizer_sha256']:raise ValueError('Tokenizer differs')
 # Serialize snapshot -> scan -> publish -> ledger commit; inventory writers must share this lock.
 with (a.database.parent/'production-release.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  ctl=SupplyControl(a.database,recipe)
  try:
   terminal=a.output/'READY_INTERNAL.json';intent=a.output/'RELEASE_INTENT.json'
   if terminal.exists() or intent.exists():
    receipt=terminal if terminal.exists() else intent
    ready=json.loads(receipt.read_text());bid=ready['batch_id']
    original=json.loads((a.output/'INPUT_BINDING.json').read_text())
    if original['candidate_sha256']!=file_sha(a.candidates) or original['policy_sha256']!=file_sha(a.policy) or original['tokenizer_sha256']!=file_sha(a.tokenizer):raise ValueError('Terminal release inputs differ')
    committed=ctl.db.execute('SELECT receipt_sha FROM releases WHERE batch_id=?',(bid,)).fetchone()
    if committed is None and ledger_snapshot(a.database)!=original['inventory_snapshot']:raise ValueError('Release intent stale after another inventory commit')
    for r in ctl.db.execute('SELECT files_json FROM stage_receipts WHERE batch_id=?',(bid,)):
     for name,h in json.loads(r[0]).items():
      if file_sha(Path(name))!=h:raise ValueError('Committed stage changed')
    load_sharded_manifest(a.output/'tokenized',verify_hashes=True)
    state=ctl.release(bid,receipt);write(terminal,ready)
    print(json.dumps(dict(stage='already_released_or_recovering_atomic_publish',ledger=state)));return
   ba=rolling_audit(a.batch_audit,a.source_audit);write(a.output/'ROLLING_AUDIT.json',ba)
   quality=quality_scope(a.source_audit,a.reference,a.reference_batch,policy['quality'],a.output/'ROLLING_AUDIT.json');write(a.output/'QUALITY.json',quality)
   if quality['status']!='passed':
    write(a.output/'HOLD.json',dict(stage='held_quality_scope',holds=quality['holds'],new_ready_train_tokens=0,training_started=False));print(json.dumps(dict(stage='held_quality_scope',holds=quality['holds'])));return
   if ba['parquet_sha256']!=file_sha(a.candidates):raise ValueError('Rolling audit is for another batch')
   ref=json.loads(a.inventory_index.read_text());snapshot=ledger_snapshot(a.database)
   if ref['identity']['ledger']!=snapshot:raise ValueError('Inventory index stale')
   engine=[Path(__file__),ROOT/'scripts/p0_verify_release_pack_v5.py',ROOT/'src/small_agent/curation/release_gates_v5.py',ROOT/'src/small_agent/curation/eval_index_v5.py',ROOT/'src/small_agent/curation/supply_control.py']
   binding=dict(candidate_sha256=file_sha(a.candidates),policy_sha256=file_sha(a.policy),tokenizer_sha256=file_sha(a.tokenizer),
     inventory_manifest_sha256=file_sha(a.inventory_index),inventory_snapshot=snapshot,eval_registry_sha256=file_sha(a.eval_registry),
     quality_sha256=file_sha(a.output/'QUALITY.json'),batch_audit_sha256=file_sha(a.output/'ROLLING_AUDIT.json'),
     engine_sha256={str(x):file_sha(x) for x in engine})
   write(a.output/'INPUT_BINDING.json',binding);bid=ctl.register('code',binding)
   def stage(name,fn):
    path=a.output/(name+'.receipt.json')
    if path.exists():
     r=bound_json(path)
     if r['input_binding_sha256']!=digest(binding):raise ValueError('Stage identity differs')
    else:
     files=fn();r=dict(stage=name,input_binding_sha256=digest(binding),evidence_files=[dict(path=str(x.resolve()),sha256=file_sha(x)) for x in files]);write(path,r)
    ctl.commit_stage(bid,name,[path,*[Path(e['path']) for e in r['evidence_files']]])
   def ingest():
    write(a.output/'ingest.json',dict(candidate_path=str(a.candidates.resolve()),candidate_sha256=binding['candidate_sha256'],inventory_snapshot=snapshot));return [a.output/'ingest.json']
   stage('ingest',ingest)
   def normalize():
    rows=[];excluded=[];families={};known=set(quality['known_audit_bad_body_sha256'])|set(ba['known_bad_body_sha256'])
    for b in pq.ParquetFile(a.candidates).iter_batches(batch_size=128):
     for r in b.to_pylist():
      reason=None;n=r['own_text_tokens'];h=('short' if n<=512 else 'medium' if n<=4096 else 'long')+'/python'
      if sha(r['text'])!=r['normalized_sha256']:raise ValueError('Frozen source body differs')
      if r['source_id']!=quality['source_id']:reason='uncalibrated_source_hold'
      elif r['language'].lower()!='python' or h not in quality['approved_strata']:reason='uncalibrated_language_or_stratum_hold'
      elif r['record_license'].lower() not in policy['licenses'] or r['policy_bucket']!='record_license_whitelisted':reason='record_license_hold'
      elif r['normalized_sha256'] in known:reason='known_audit_bad_document_hold'
      elif r['split'] not in ('train','validation','test'):reason='unassigned_split_hold'
      old=ctl.db.execute('SELECT split FROM legacy_families WHERE family_id=?',(r['family_id'],)).fetchone()
      if old and old[0]!=r['split']:reason='global_family_split_hold'
      if r['family_id'] in families and families[r['family_id']]!=r['split']:raise ValueError('Within batch family split conflict')
      if reason:excluded.append(dict(body_sha256=r['normalized_sha256'],reason=reason));continue
      families[r['family_id']]=r['split'];rows.append(r)
    if not rows:raise ValueError('All candidates held')
    pq.write_table(pa.Table.from_pylist(rows),a.output/'normalized.parquet',compression='zstd');write(a.output/'SOURCE_POLICY.json',dict(status='passed',source_id=quality['source_id'],language='python',retained_documents=len(rows),exclusions=excluded,scope='declared permissive licenses; internal-only, no rights or distribution certification'))
    return [a.output/'normalized.parquet',a.output/'SOURCE_POLICY.json']
   stage('normalize',normalize)
   def dedup():
    rows,near=global_near(pq.read_table(a.output/'normalized.parquet').to_pylist(),a.inventory_index,policy['near'],a.database)
    # All splits undergo benchmark screening before freeze.
    rows,ev=EvalScreen(a.eval_registry,policy['evaluation']).filter(rows)
    if not rows:raise ValueError('All candidates excluded by isolation gates')
    pq.write_table(pa.Table.from_pylist(rows),a.output/'deduplicated.parquet',compression='zstd');write(a.output/'GLOBAL_NEAR.json',near);write(a.output/'EVAL_SCREEN.json',ev)
    return [a.output/'deduplicated.parquet',a.output/'GLOBAL_NEAR.json',a.output/'EVAL_SCREEN.json']
   stage('dedup',dedup)
   stage('quality',lambda:[a.output/'QUALITY.json',a.output/'ROLLING_AUDIT.json'])
   def freeze():
    totals=Counter();docs=[]
    for b in pq.ParquetFile(a.output/'deduplicated.parquet').iter_batches(batch_size=128):
     for r in b.to_pylist():
      docs.append(dict(body_sha256=r['normalized_sha256'],document_id=r['normalized_sha256'],family_id=r['family_id'],domain='code',split=r['split'],body_tokens=r['own_text_tokens'],source_locator=r['source_locator'],source_revision=r['source_revision'],declared_license=r['record_license']));totals[r['split']]+=r['own_text_tokens']
    (a.output/'documents.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs))
    write(a.output/'FROZEN.json',dict(stage='source_audited_internal_pretraining_text',training_eligible=True,distribution_eligible=False,parquet_sha256=file_sha(a.output/'deduplicated.parquet'),index_sha256=file_sha(a.output/'documents.jsonl'),body_tokens=dict(totals),documents=len(docs),tokenizer_sha256=binding['tokenizer_sha256']))
    return [a.output/'FROZEN.json',a.output/'documents.jsonl',a.output/'deduplicated.parquet']
   stage('freeze',freeze)
   def pack():
    tok=Tokenizer.from_file(str(a.tokenizer))
    def docs():
     for b in pq.ParquetFile(a.output/'deduplicated.parquet').iter_batches(batch_size=64):
      for r in b.to_pylist():
       ids=tok.encode(r['text'],add_special_tokens=False).ids
       if len(ids)!=r['own_text_tokens'] or tok.decode(ids,skip_special_tokens=False)!=r['text']:raise ValueError('Token count/roundtrip differs')
       yield TokenizedDocument(r['normalized_sha256'],r['split'],'code',ids)
    out=a.output/'tokenized'
    if out.exists():
     m,_=load_sharded_manifest(out,verify_hashes=True)
     if m['source_manifest_sha256']!=file_sha(a.output/'FROZEN.json'):raise ValueError('Interrupted pack identity differs')
    else:pack_sharded(docs(),out,sequence_length=policy['sequence_length'],target_shard_tokens=policy['target_shard_tokens'],vocab_size=32768,eos_token_id=2,tokenizer_sha256=binding['tokenizer_sha256'],mixture={'code':1.0},license_gate=dict(training_eligible=True,distribution_eligible=False,scope=policy['scope']),source_manifest_sha256=file_sha(a.output/'FROZEN.json'))
    m=json.loads((out/'manifest.json').read_text());return [out/'manifest.json',*[out/s[k] for s in m['shards'] for k in ('bin','idx')]]
   stage('pack',pack)
   def verify():
    subprocess.run([sys.executable,str(ROOT/'scripts/p0_verify_release_pack_v5.py'),'--run',str(a.output),'--tokenizer',str(a.tokenizer),'--database',str(a.database)],check=True)
    return [a.output/'VERIFY.json']
   stage('verify',verify)
   if ledger_snapshot(a.database)!=snapshot:raise ValueError('Inventory changed: rescreen before READY')
   checks=[]
   for name,files in {'source_policy':['SOURCE_POLICY.json'],'quality_scope':['QUALITY.json','ROLLING_AUDIT.json'],'global_exact_near_dedup':['GLOBAL_NEAR.json'],'split_eval_isolation':['EVAL_SCREEN.json','VERIFY.json'],'tokenizer_pack_loader':['VERIFY.json']}.items():
    for n in files:
     r=json.loads((a.output/n).read_text())
     if 'status' in r and r['status']!='passed':raise ValueError('Check has not actually passed')
    checks.append(dict(name=name,status='passed',checker_revision=binding['engine_sha256'][str(Path(__file__))],batch_id=bid,recipe_sha256=ctl.recipe_sha,input_binding_sha256=digest(binding),evidence_files=[dict(path=str((a.output/n).resolve()),sha256=file_sha(a.output/n)) for n in files]))
   frozen=json.loads((a.output/'FROZEN.json').read_text());ready=dict(status='READY_INTERNAL',batch_id=bid,recipe_sha256=ctl.recipe_sha,input_binding_sha256=digest(binding),tokenizer_sha256=binding['tokenizer_sha256'],unique_train_body_tokens=frozen['body_tokens']['train'],checks=checks,document_index=dict(path=str((a.output/'documents.jsonl').resolve()),sha256=file_sha(a.output/'documents.jsonl')),pack_manifest=dict(path=str((a.output/'tokenized/manifest.json').resolve()),sha256=file_sha(a.output/'tokenized/manifest.json')),scope=policy['scope'],distribution_eligible=False)
   # Write intent outside the consumption filename, commit ledger, then publish READY.
   write(a.output/'RELEASE_INTENT.json',ready);state=ctl.release(bid,a.output/'RELEASE_INTENT.json');write(a.output/'READY_INTERNAL.json',ready)
   print(json.dumps(dict(stage='READY_INTERNAL',output=str(a.output),ledger=state)))
  finally:ctl.close()
if __name__=='__main__':main()
