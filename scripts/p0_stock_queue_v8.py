#!/usr/bin/env python3
"""Durable all-stock gate and bounded mathematical supply; never automatic training.

Re-entering after a terminal HOLD does not re-audit or re-bill. A new source
protocol/cohort requires an explicit versioned config, never overwriting evidence.
"""
import argparse,fcntl,json,os,shutil,sqlite3,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.math_quality_v8 import math_quality
from small_agent.curation.release_gates_v5 import write,ledger_snapshot
from small_agent.curation.supply_control import file_sha,digest
from p0_stock_plan_v6 import materialize
from p0_math_clean_jobs_v8 import open_shared_plan as open_plan
from p0_math_clean_jobs_v8 import allocate_clean
from p0_stock_audits_v8 import prepare_pool,link_audit
from p0_curation_worker_v3 import atomic_status

def completed_worker(batch):
    p=batch/'worker/deepseek/ADAPTIVE_STATUS.json'
    if not p.exists():return False
    s=json.loads(p.read_text())
    if s['stop_reason'] is not None:raise ValueError('Paid audit stopped: inspection required, never automatic retry')
    return s['stage']=='all_selected_attempted_with_holds' and s['considered']==s['selected']

def scope_gate(config):
    return math_quality(Path(config['source_audit']),Path(config['quality_reference'])/'REFERENCE.json',Path(config['quality_reference'])/'batch',json.loads(Path(config['policy']).read_text())['quality'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text())
    if cfg['maximum_concurrency']!=2500 or cfg['auto_training'] or cfg['distribution_eligible']:raise ValueError('Explicit frozen internal supply contract required')
    stock=Path(cfg['stock']);stock.mkdir(parents=True,exist_ok=True)
    filtered=Path(cfg['filtered'])
    if not (filtered/'QUALITY_SCOPE.json').exists():print('WAITING_FRESH_FILTERED_CALIBRATION_NO_API');return
    filtered_scope=json.loads((filtered/'QUALITY_SCOPE.json').read_text())
    if filtered_scope['status']!='passed':print('FILTERED_SOURCE_HOLD_NO_API');return
    with (stock/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (stock/'TERMINAL_HOLD.json').exists():print('EXISTING_TERMINAL_HOLD_NO_API');return
        allocate_clean(stock,Path(cfg['original_stock']),filtered,filtered_scope,cfg['batch_train_body_tokens'])
        def status(stage,**kw):atomic_status(stock/'QUEUE_STATUS.json',dict(version=6,stage=stage,config_sha256=file_sha(a.config),maximum_concurrency=2500,training_started=False,**kw))
        try:
            if not (stock/'STOCK_REGISTERED.json').exists():status('waiting_full_stock_registration');return
            source=Path(cfg['source_audit']);refs=Path(cfg['quality_reference'])
            if not completed_worker(source/'batch') or not completed_worker(refs/'batch'):status('waiting_calibration_not_queued_to_gpu');return
            scope=scope_gate(cfg);write(stock/'MATH_SCOPE.json',scope)
            if scope['status']!='passed':
                db=open_plan(stock)
                try:
                    with db:db.execute("UPDATE records SET gate='source_quality_calibration_hold' WHERE domain='math' AND gate='source_calibration_pending'")
                finally:db.close()
                write(stock/'TERMINAL_HOLD.json',dict(version=6,stage='math_source_calibration_hold',holds=scope['holds'],scope_sha256=file_sha(stock/'MATH_SCOPE.json'),new_math_ready_train_tokens=0,paid_retry=False,other_sources_continued_separately=True))
                status('HOLD_MATH_SOURCE',holds=scope['holds'],new_math_ready_train_tokens=0);print(json.dumps(dict(stage='HOLD_MATH_SOURCE',holds=scope['holds'],source_strata=scope['source_strata'],reference_counts=scope['reference_counts']),ensure_ascii=False));return
            # Allocation is restricted to frozen structural IDs above.
            if not (Path(cfg['inventory_root'])/'CURRENT').exists():status('waiting_initial_disk_index_no_duplicate_launch');return
            if not (stock/'AUDITS_PREPARED.json').exists():status('preparing_pointer_audit_controls');prepare_pool(stock,source,Path(cfg['profile']))
            def run(script,args):
                status('running',component=script)
                r=subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'})
                if r.returncode:raise ValueError('Component failed without automatic API retry: '+script)
            def worker(batch,initial):run('p0_deepseek_worker_v8.py',['--profile',cfg['profile'],'--batch',batch,'--allow-api','--initial-concurrency',initial,'--maximum-concurrency',2500,'--limit',10000])
            pool=stock/'audit_pool/batch'
            if not completed_worker(pool):worker(pool,128)
            idxroot=Path(cfg['inventory_root']);pointer=idxroot/'CURRENT'
            if not pointer.exists():status('waiting_initial_disk_index_no_duplicate_launch');return
            db=open_plan(stock)
            try:
                jobs=db.execute("SELECT id,name,state FROM jobs WHERE name LIKE 'math_clean_%_v8' AND state NOT IN ('READY_INTERNAL','HOLD','FAILED_NO_AUTO_API_RETRY') ORDER BY id").fetchall()
                releases=Path(cfg['release_root']);releases.mkdir(exist_ok=True)
                for jid,name,state in jobs:
                    if shutil.disk_usage(releases).free<json.loads(Path(cfg['recipe']).read_text())['minimum_disk_free_bytes']:status('waiting_disk_backpressure');return
                    out=releases/name;candidate=stock/'materialized'/name/'candidates.parquet';terminal=out/'READY_INTERNAL.json'
                    if not terminal.exists():
                        # Backpressure: at most two unfinished owned corpus batches.
                        s=sqlite3.connect(cfg['database'])
                        try:pending=s.execute("SELECT COUNT(*) FROM batches WHERE status NOT IN ('ready','quarantined','legacy_internal')").fetchone()[0]
                        finally:s.close()
                        if pending>=2 and not (out/'INPUT_BINDING.json').exists():status('waiting_batch_backpressure');return
                        status('materializing_current_pointer_batch',job_id=jid);materialize(Path(cfg['original_stock']),jid,candidate)
                        audit=link_audit(stock,jid,name,candidate);worker(audit/'batch',8) # imported paid pool receipts, no repeated billing
                        if json.loads(Path(pointer.read_text().strip()).read_text())['ledger']!=ledger_snapshot(Path(cfg['database'])):
                            run('p0_build_inventory_index_v6.py',['--database',cfg['database'],'--legacy-receipt',cfg['legacy_receipt'],'--sketch-root',cfg['sketch_root'],'--output',idxroot])
                    args=[]
                    bindings={'candidates':candidate,'candidate-binding':candidate.parent/'CANDIDATE_BINDING.json','source-audit':source,'reference':refs/'REFERENCE.json','reference-batch':refs/'batch','batch-audit':stock/'audits'/name,'eval-registry':cfg['eval_registry'],'inventory-index':Path(pointer.read_text().strip()),'database':cfg['database'],'tokenizer':cfg['tokenizer'],'output':out,'recipe':cfg['recipe'],'policy':cfg['policy'],'source-terms':cfg['source_terms'],'filter-receipt':filtered/'FILTER_COMPLETE.json','filter-config':cfg['filter_config']}
                    for key,value in bindings.items():args.extend(['--'+key,value])
                    run('p0_release_math_batch_v8.py',args)
                    if terminal.exists():end='READY_INTERNAL'
                    elif (out/'HOLD.json').exists():end='HOLD'
                    else:raise ValueError('Release lacks terminal evidence')
                    with db:db.execute('UPDATE jobs SET state=? WHERE id=?',(end,jid));db.execute('UPDATE records SET gate=? WHERE job=?',(end,jid))
                    ready=db.execute("SELECT COUNT(*) FROM jobs WHERE state='READY_INTERNAL'").fetchone()[0]
                    if end=='READY_INTERNAL' and (ready==1 or ready%10==0):
                        run('p0_ready_catalog_v5.py',['--database',cfg['database'],'--output',stock/'ready_catalog_v6.json'])
                        run('p0_training_bundle_v5.py',['--database',cfg['database'],'--output',stock/'training_bundles_v6'])
                run('p0_ready_catalog_v5.py',['--database',cfg['database'],'--output',stock/'ready_catalog_v6.json'])
                run('p0_training_bundle_v5.py',['--database',cfg['database'],'--output',stock/'training_bundles_v6'])
                status('ALL_ALLOCATED_MATH_BATCHES_TERMINAL',job_counts=dict(db.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state').fetchall()))
            finally:db.close()
        except Exception as exc:
            status('FAILED_NO_AUTO_API_RETRY',failure_type=type(exc).__name__,reason=str(exc))
            write(stock/'TERMINAL_HOLD.json',dict(stage='failed_fail_closed',failure_type=type(exc).__name__,reason=str(exc),new_ready_tokens=0,paid_retry=False));raise
if __name__=='__main__':main()
