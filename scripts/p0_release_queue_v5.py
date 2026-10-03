#!/usr/bin/env python3
"""Durable automatic queue: frozen candidate batches -> measured READY tokens.

New sources/languages need registered independent references. Unknown ones HOLD;
there is no universal classifier or bypass to create a fictitious READY counter.
Failed requests/failed terminal jobs are not automatically sent/billed again.
"""
import argparse,fcntl,json,os,subprocess,sys,shutil,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.release_gates_v5 import sha,ledger_snapshot
from small_agent.curation.supply_control import file_sha
from p0_curation_worker_v3 import atomic_status


def validate_job(job,release_root):
 required={'version','processor','source_id','candidates','candidate_sha256','source_audit','output'}
 if set(job)!=required or job['version']!=5 or job['processor']!='python_internal_pretraining_v5':raise ValueError('Unknown job contract')
 if job['source_id']!='github_code_clean_supply_v4':return 'unregistered_source_reference_hold'
 for k in ('candidates','source_audit','output'):
  if not Path(job[k]).is_absolute():raise ValueError('Absolute server paths required')
 if release_root.resolve() not in Path(job['output']).resolve().parents:raise ValueError('Output outside owned release root')
 if file_sha(Path(job['candidates']))!=job['candidate_sha256']:raise ValueError('Frozen candidate changed')
 return None


def validate_source_cohort(job):
    """A source ID alone is not permission to reuse calibration for new revisions/shards."""
    import pyarrow.parquet as pq
    audit=Path(job['source_audit']);plan=json.loads((audit/'SAMPLE_PLAN.json').read_text())
    for name,h in plan['frozen_files'].items():
        if file_sha(audit/name)!=h:raise ValueError('Frozen source calibration changed')
    if plan['request_identity']['source_id']!=job['source_id'] or not plan['source_complete_in_plan']:return 'source_calibration_scope_hold'
    revisions={json.loads(l)['source_revision'] for l in (audit/'input.jsonl').read_text().splitlines()}
    shards={x['source_file'] for x in plan['input_source_files']}
    for batch in pq.ParquetFile(job['candidates']).iter_batches(batch_size=4096,columns=['source_id','source_revision','source_file']):
        for r in batch.to_pylist():
            if r['source_id']!=job['source_id'] or r['source_revision'] not in revisions or r['source_file'] not in shards:return 'new_revision_or_uncalibrated_shard_hold'
    return None


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('queue','profile','release-root','database','tokenizer','legacy-receipt','inventory-root','eval-registry','quality-reference'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--recipe',type=Path,default=ROOT/'configs/curation/base_1b_supply_20b_v4.json');p.add_argument('--policy',type=Path,default=ROOT/'configs/curation/p0_release_gates_v5.json');p.add_argument('--allow-api',action='store_true')
 a=p.parse_args();a.queue.mkdir(parents=True,exist_ok=True);(a.queue/'state').mkdir(exist_ok=True)
 with (a.queue/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for path in sorted(a.queue.glob('*.json')):
   identity=file_sha(path);state=a.queue/'state'/(identity+'.json');job=json.loads(path.read_text());out=Path(job.get('output','/invalid'))
   if state.exists() and json.loads(state.read_text()).get('stage') in ('READY_INTERNAL','HOLD','FAILED_NO_AUTO_API_RETRY'):continue
   try:
    error=validate_job(job,a.release_root)
    if error is None:error=validate_source_cohort(job)
   except Exception as exc:
    atomic_status(state,dict(stage='FAILED_NO_AUTO_API_RETRY',failure_type=type(exc).__name__,reason=str(exc),job_sha256=identity,new_ready_tokens=0));continue
   if error:atomic_status(state,dict(stage='HOLD',reason=error,job_sha256=identity,new_ready_tokens=0));continue
   def status(stage,**kwargs):atomic_status(state,dict(stage=stage,job_sha256=identity,output=str(out),training_started=False,**kwargs))
   def run(script,args):
    status('running',component=script)
    result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)])
    if result.returncode:raise ValueError('Component failed: '+script+'; no automatic paid retry')
   try:
    # An already frozen/published release is reverified without any API request.
    terminal=out/'READY_INTERNAL.json'
    if not terminal.exists():
     recipe=json.loads(a.recipe.read_text())
     if shutil.disk_usage(a.release_root if a.release_root.exists() else a.release_root.parent).free<recipe['minimum_disk_free_bytes']:
      status('waiting_disk_backpressure',new_ready_tokens=0);continue
     db=sqlite3.connect('file:'+str(a.database.resolve())+'?mode=ro',uri=True)
     try:pending=db.execute("SELECT COUNT(*) FROM batches WHERE status NOT IN ('ready','quarantined','legacy_internal')").fetchone()[0]
     finally:db.close()
     if pending>=recipe['maximum_unreleased_batches'] and not (out/'INPUT_BINDING.json').exists():
      status('waiting_batch_backpressure',new_ready_tokens=0);continue
    if not terminal.exists() and not a.allow_api:
     status('waiting_explicit_api_authorization');continue
    if not (a.quality_reference/'REFERENCE.json').exists():raise ValueError('Missing independently verified quality reference; onboarding required')
    source=Path(job['source_audit']);batch_audit=out.parent/(out.name+'.batch-audit')
    if not terminal.exists():
     run('p0_prepare_batch_audit_v5.py',['--parquet',job['candidates'],'--profile',a.profile,'--output',batch_audit,'--source-audit',source])
     run('p0_deepseek_adaptive_worker_v5.py',['--profile',a.profile,'--batch',batch_audit/'batch','--allow-api','--initial-concurrency','8','--maximum-concurrency','2500','--limit','2000'])
    pointer=a.inventory_root/'CURRENT'
    if not terminal.exists() and (not pointer.exists() or json.loads(Path(pointer.read_text().strip()).read_text())['identity']['ledger']!=ledger_snapshot(a.database)):
     run('p0_build_inventory_index_v5.py',['--database',a.database,'--legacy-receipt',a.legacy_receipt,'--output',a.inventory_root])
    index=Path(pointer.read_text().strip())
    # If already READY the release runner takes its idempotent physical recheck path.
    run('p0_release_code_batch_v5.py',['--candidates',job['candidates'],'--source-audit',source,'--reference',a.quality_reference/'REFERENCE.json','--reference-batch',a.quality_reference/'batch',
      '--batch-audit',batch_audit,'--eval-registry',a.eval_registry,'--inventory-index',index,'--database',a.database,'--tokenizer',a.tokenizer,'--output',out,'--recipe',a.recipe,'--policy',a.policy])
    if terminal.exists():
     run('p0_ready_catalog_v5.py',['--database',a.database,'--output',a.release_root.parent/'ready_catalog_v5.json'])
     run('p0_training_bundle_v5.py',['--database',a.database,'--output',a.release_root.parent/'training_bundles_v5'])
     receipt=json.loads(terminal.read_text());status('READY_INTERNAL',ready_receipt_sha256=file_sha(terminal),new_ready_train_body_tokens=receipt['unique_train_body_tokens'])
    elif (out/'HOLD.json').exists():status('HOLD',holds=json.loads((out/'HOLD.json').read_text())['holds'],new_ready_tokens=0)
    else:raise ValueError('No terminal measured release/hold')
   except Exception as exc:
    status('FAILED_NO_AUTO_API_RETRY',failure_type=type(exc).__name__,reason=str(exc),new_ready_tokens=0)
    # Account-wide unknown paid attempts must not be bypassed by moving to another job.
    raise
if __name__=='__main__':main()
