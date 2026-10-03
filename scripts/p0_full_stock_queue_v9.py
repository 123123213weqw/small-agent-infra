#!/usr/bin/env python3
"""ALL-stock tier queue with quota/budget/backpressure; no training or automatic gate bypass."""
import argparse,fcntl,json,os,shutil,sqlite3,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.full_stock_v9 import MATH_BUCKETS,ReadyQuotaControl
from small_agent.curation.release_gates_v5 import write,ledger_snapshot
from small_agent.curation.supply_control import file_sha
from p0_full_stock_status_v9 import report
from p0_full_stock_math_v9 import prepare_source,source_scope,allocate_one,batch_audit
from p0_stock_plan_v6 import materialize,open_plan
from p0_curation_worker_v3 import atomic_status


def completed(batch):
    p=batch/'worker/deepseek/ADAPTIVE_STATUS.json'
    if not p.exists():return False
    s=json.loads(p.read_text())
    if s['stop_reason'] is not None:raise ValueError('Paid attempt stopped; explicit inspection, no automatic resend')
    return s['stage']=='all_selected_attempted_with_holds' and s['considered']==s['selected']


def run(config,allow_api):
    if config['maximum_concurrency']!=2500 or config['shared_campaign_budget_cny']!=1000 or config['auto_training'] is not False:
        raise ValueError('Explicit existing campaign and training boundary required')
    stock=Path(config['stock']);stock.mkdir(parents=True,exist_ok=True)
    with (stock/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def status(stage,**extra):
            atomic_status(stock/'QUEUE_STATUS.json',dict(version=9,stage=stage,training_started=False,maximum_concurrency=2500,shared_budget_cny=1000,**extra))
            atomic_status(stock/'BALANCE_STATUS.json',report(config))
        status('WAITING_ALL_STOCK_RULE_SCAN')
        if not (stock/'SCAN_COMPLETE.json').exists():return
        if not allow_api:status('WAITING_EXPLICIT_API_AUTHORIZATION_NO_PAID_ATTEMPTS');return
        receipt=json.loads((stock/'SCAN_COMPLETE.json').read_text())
        if receipt['identity']['config_sha256']!=file_sha(Path(config['filter_config'])) or receipt['identity']['router_sha256']!=file_sha(ROOT/'src/small_agent/curation/full_stock_v9.py'):
            raise ValueError('Frozen ALL-stock filter changed')
        reference=Path(config['quality_reference'])
        if not (reference/'REFERENCE_GATE.json').exists() or json.loads((reference/'REFERENCE_GATE.json').read_text())['status']!='passed':
            status('WAITING_INDEPENDENT_MATH_REFERENCE_NO_API');return
        def component(script,args):
            result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'})
            if result.returncode:raise ValueError('Component failed, no automatic API retry: '+script)
        def worker(batch):
            if completed(batch):return
            if not allow_api:raise ValueError('Explicit paid campaign authorization required')
            component('p0_deepseek_worker_v8.py',['--profile',config['profile'],'--batch',batch,'--allow-api','--initial-concurrency',64,'--maximum-concurrency',2500,'--limit',10000])
            if not completed(batch):raise ValueError('Paid worker lacks complete terminal status')
        approved=[]
        audits=stock/'source_audits';audits.mkdir(exist_ok=True)
        for bucket in MATH_BUCKETS:
            source=audits/bucket;terminal=source/'TERMINAL_HOLD.json'
            if terminal.exists():continue
            try:
                status('PREPARING_FRESH_ALL_HOST_TIER',bucket=bucket)
                if not prepare_source(config,bucket,source):continue
                scope_path=source/'QUALITY_SCOPE.json'
                if scope_path.exists() and json.loads(scope_path.read_text())['status']!='passed':continue # no resampling until pass
                status('SOURCE_CALIBRATION',bucket=bucket)
                worker(source/'batch');scope=source_scope(config,bucket,source)
                if scope['status']=='passed':approved.append((bucket,source,scope))
            except Exception as exc:
                source.mkdir(parents=True,exist_ok=True)
                write(terminal,dict(stage='FAILED_NO_AUTO_API_RETRY',failure_type=type(exc).__name__,reason=str(exc),training_started=False))
                print('TIER_TERMINAL_HOLD',bucket,type(exc).__name__,str(exc),flush=True)
        if not approved:
            status('ALL_NEW_MATH_TIERS_QUARANTINED_OTHER_PROCESSORS_CONTINUE',new_ready_tokens=0);return
        pointer=Path(config['inventory_root'])/'CURRENT'
        if not pointer.exists():status('WAITING_EXISTING_GLOBAL_NEAR_INDEX_NO_DUPLICATE_LAUNCH');return
        if shutil.disk_usage(stock).free<config['minimum_disk_free_bytes']:
            status('WAITING_DISK_BACKPRESSURE');return
        released=0
        for bucket,source,scope in approved:
            for _ in range(config['maximum_batches_per_tick']):
                state=report(config);gap=state['remaining_ready_by_domain']['math']
                if not gap:status('MATH_QUOTA_MET_RESERVE_UNCHANGED');return
                # Wait before spending batch-audit budget while another inventory writer runs.
                with (Path(config['database']).parent/'production-release.lock').open('a') as release_lock:
                    try:fcntl.flock(release_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:status('WAITING_EXISTING_INVENTORY_WRITER');return
                ctl=ReadyQuotaControl(Path(config['database']),json.loads(Path(config['recipe']).read_text()))
                try:pending=ctl.db.execute("SELECT COUNT(*) FROM batches WHERE status NOT IN ('ready','quarantined','legacy_internal')").fetchone()[0]
                finally:ctl.close()
                # A v9 interrupted own batch can resume at its already bound input.
                db=open_plan(stock)
                try:active=db.execute("SELECT name FROM jobs WHERE state='running' ORDER BY id LIMIT 1").fetchone()
                finally:db.close()
                if pending>=config['maximum_unreleased_batches'] and not active:status('WAITING_SHARED_BATCH_BACKPRESSURE');return
                job=allocate_one(config,bucket,scope,min(gap,config['batch_train_body_tokens']))
                if job is None:break
                jid,name=job;out=Path(config['release_root'])/name;out.mkdir(parents=True,exist_ok=True)
                if pending>=config['maximum_unreleased_batches'] and not (out/'INPUT_BINDING.json').exists():status('WAITING_SHARED_BATCH_BACKPRESSURE');return
                try:
                    db=open_plan(stock)
                    try:
                        with db:db.execute("UPDATE jobs SET state='running' WHERE id=?",(jid,))
                    finally:db.close()
                    status('MATERIALIZING_CURRENT_BOUNDED_BATCH',bucket=bucket,job_id=jid)
                    candidate=stock/'materialized'/name/'candidates.parquet';materialize(stock,jid,candidate)
                    audit=stock/'batch_audits'/name;batch_audit(config,source,candidate,audit)
                    status('ROLLING_BATCH_AUDIT',bucket=bucket,job_id=jid);worker(audit/'batch')
                    current=Path(pointer.read_text().strip())
                    if json.loads(current.read_text())['ledger']!=ledger_snapshot(Path(config['database'])):
                        component('p0_build_inventory_index_v6.py',['--database',config['database'],'--legacy-receipt',config['legacy_receipt'],'--sketch-root',config['sketch_root'],'--output',config['inventory_root']])
                        current=Path(pointer.read_text().strip())
                    bindings={'candidates':candidate,'candidate-binding':candidate.parent/'CANDIDATE_BINDING.json','source-audit':source,'reference':reference/'REFERENCE.json','reference-batch':reference/'batch','batch-audit':audit,
                        'eval-registry':config['eval_registry'],'inventory-index':current,'database':config['database'],'tokenizer':config['tokenizer'],'output':out,
                        'recipe':config['recipe'],'policy':config['policy'],'source-terms':config['source_terms'],'filter-receipt':stock/'SCAN_COMPLETE.json','filter-config':config['filter_config'],'filter-bucket':bucket}
                    args=[item for k,v in bindings.items() for item in ('--'+k,v)]
                    status('VERIFYING_AND_RELEASING',bucket=bucket,job_id=jid);component('p0_release_math_batch_v9.py',args)
                    end='READY_INTERNAL' if (out/'READY_INTERNAL.json').exists() else 'HOLD' if (out/'HOLD.json').exists() else None
                    if not end:raise ValueError('Batch lacks verified terminal receipt')
                    db=open_plan(stock)
                    try:
                        with db:db.execute('UPDATE jobs SET state=? WHERE id=?',(end,jid));db.execute('UPDATE records SET gate=? WHERE job=?',(end,jid))
                    finally:db.close()
                    released+=end=='READY_INTERNAL'
                except Exception as exc:
                    # Do not keep an abandoned incomplete ledger batch blocking unrelated domains forever.
                    if (out/'INPUT_BINDING.json').exists() and not (out/'RELEASE_INTENT.json').exists():
                        db=sqlite3.connect(config['database'])
                        try:
                            with db:db.execute("UPDATE batches SET status='quarantined',reason='v9 terminal component failure; explicit inspection required' WHERE input_binding=? AND status!='ready'",(json.dumps(json.loads((out/'INPUT_BINDING.json').read_text()),sort_keys=True),))
                        finally:db.close()
                    db=open_plan(stock)
                    try:
                        with db:db.execute("UPDATE jobs SET state='FAILED_NO_AUTO_API_RETRY',reason=? WHERE id=?",(str(exc),jid))
                    finally:db.close()
                    write(out/'TERMINAL_HOLD.json',dict(stage='FAILED_NO_AUTO_API_RETRY',reason=str(exc),failure_type=type(exc).__name__,training_started=False))
                    status('BATCH_QUARANTINED_INSPECTION_REQUIRED',job_id=jid);return
                if released>=config['maximum_batches_per_tick']:break
            if released>=config['maximum_batches_per_tick']:break
        if released:
            component('p0_ready_catalog_v5.py',['--database',config['database'],'--output',stock/'ready_catalog.json'])
            # Catalog = reserve bank. Never use the old observed-mixture one-pass
            # bundle as the 20B recipe. Missing domains block the balanced assembler.
            component('p0_balanced_training_bundle_v9.py',['--database',config['database'],'--output',stock/'balanced_training_bundles','--recipe',config['recipe']])
        status('TICK_COMPLETE',released_this_tick=released,approved_math_tiers=[b for b,_,_ in approved],
               code_nonpython_action='quarantined_until_independent_language_reference',
               other_domain_action='explicit_replenishment_queue_not_fictitious_ready')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--allow-api',action='store_true');a=p.parse_args()
    try:run(json.loads(a.config.read_text()),a.allow_api)
    except BlockingIOError:print('EXISTING_V9_QUEUE_RUNNING_NO_DUPLICATE_START')
if __name__=='__main__':main()
