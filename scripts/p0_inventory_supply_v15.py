#!/usr/bin/env python3
"""v15 ordered publisher rollout; unchanged frozen v11 gates, budget and sampling."""
import argparse,fcntl,json,os,sqlite3,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import p0_inventory_supply_v11 as original
from p0_inventory_supply_v11 import *
from small_agent.curation.release_gates_v5 import sha
from p0_arrow_reader_v15 import ArrowRawReader


def cpu_progress(cfg,phase,**extra):
    atomic_status(Path(cfg['root'])/'QUEUE_CPU_PROGRESS_V15.json',dict(version=15,phase=phase,updated_unix=time.time(),**extra))


def source_query(store,clauses):
    # Force a source-first join. Globally rank-ordered indexes otherwise fetch
    # millions of other sources before finding a small cohort's 64 samples.
    if store.execute("SELECT 1 FROM sqlite_master WHERE name='records_source_v15' AND type='index'").fetchone():
        return 'SELECT r.* FROM files f CROSS JOIN records r INDEXED BY records_source_v15 WHERE r.fid=f.id AND f.source=? AND r.domain=? AND r.lang=? AND '+clauses
    return 'SELECT r.* FROM records r JOIN files f ON f.id=r.fid WHERE f.source=? AND r.domain=? AND r.lang=? AND '+clauses


def source_cohorts(store,source):
    if store.execute("SELECT 1 FROM sqlite_master WHERE name='records_source_v15' AND type='index'").fetchone():
        query="SELECT DISTINCT r.domain,r.lang FROM files f CROSS JOIN records r INDEXED BY records_source_v15 WHERE r.fid=f.id AND f.source=? AND r.gate IN ('CANDIDATE','CANDIDATE_FALLBACK') ORDER BY r.domain,r.lang"
    else:query="SELECT DISTINCT domain,lang FROM records r JOIN files f ON f.id=r.fid WHERE f.source=? AND r.gate IN ('CANDIDATE','CANDIDATE_FALLBACK') ORDER BY domain,lang"
    return store.execute(query,(source,)).fetchall()


def sample(store,source,domain,language,k,gate_clause="r.gate='CANDIDATE' AND r.job IS NULL",job=None):
    params=[source,domain,language];extra='' if job is None else ' AND r.job=?';params+=[] if job is None else [job]
    query='SELECT r.* FROM records r JOIN files f ON f.id=r.fid WHERE f.source=? AND r.domain=? AND r.lang=? AND '+gate_clause+extra
    counts={x[0]:x[1] for x in store.execute('SELECT stratum,COUNT(*) FROM ('+query+") WHERE split='train' GROUP BY stratum",params)}
    selected=[]
    for s in sorted(counts):
        # Pure query routing, NOT an acceptance threshold or changed sample size.
        # Dense cohorts keep the covering rank path; sparse cohorts start at source.
        routed=query if counts[s]>=1000 else source_query(store,gate_clause+extra)
        selected+=store.execute('SELECT * FROM ('+routed+") WHERE split='train' AND stratum=? ORDER BY rank LIMIT ?",[*params,s,k]).fetchall()
    return selected,counts

def allocate(cfg,store,source,domain,language,scope,mode,smoke):
    clauses="r.gate='CANDIDATE' AND r.job IS NULL";params=[source,domain,language];approved=scope['approved_strata']
    if mode=='source_audit':
        if not approved:return None
        clauses+=' AND r.stratum IN ('+','.join('?' for _ in approved)+')';params+=approved
    else:
        if approved:
            clauses="(r.gate='CANDIDATE' AND r.stratum NOT IN ("+','.join('?' for _ in approved)+") OR r.gate='CANDIDATE_FALLBACK') AND r.job IS NULL";params+=approved
        else:clauses="r.gate IN ('CANDIDATE','CANDIDATE_FALLBACK') AND r.job IS NULL"
    q=source_query(store,clauses)
    selected=[];n=0;target=cfg['smoke_train_body_tokens'] if smoke else cfg['batch_train_body_tokens'];limit=508 if mode=='individual' else cfg['maximum_batch_documents']
    for r in store.execute(q+" AND r.split='train' ORDER BY r.rank LIMIT ?",[*params,limit]):
        selected.append(r);n+=r['n']
        if n>=target:break
    if not selected:return None
    for split in ('validation','test'):
        held=store.execute(q+' AND r.split=? ORDER BY r.rank LIMIT 2',[*params,split]).fetchall()
        if not held:return None # no invented heldout or moving train records
        selected+=held
    name='batch_'+digest([source,domain,language,mode,file_sha(Path(cfg['prompt'])),[r['h'] for r in selected]])[:24]
    with store:
        store.execute('INSERT OR IGNORE INTO jobs(name,source,mode) VALUES (?,?,?)',(name,source,mode))
        store.executemany('UPDATE records SET job=? WHERE h=?',[(name,r['h']) for r in selected])
    return name,selected

def frozen_rows(cfg,store,records,candidate,registry):
    import pyarrow.parquet as pq
    binding=json.loads((candidate.parent/'CANDIDATE_BINDING.json').read_text())
    if binding['candidate_sha256']!=file_sha(candidate) or binding['registry_sha256']!=file_sha(Path(cfg['root'])/'REGISTRY.json'):
        raise ValueError('Frozen candidate/registry changed')
    sources={s['id']:s for s in registry['sources']};files={f['path']:f for s in registry['sources'] for f in s['files']}
    selected={r['h']:r for r in records};checked=set();rows=pq.read_table(candidate).to_pylist()
    seen=set()
    for row in rows:
        h=row['normalized_sha256']
        if h in seen or h not in selected or sha(row['text'])!=h:raise ValueError('Frozen candidate body/selection changed')
        seen.add(h);rec=selected[h];f=store.execute('SELECT * FROM files WHERE id=?',(rec['fid'],)).fetchone()
        src=sources[f['source']];reg=files[f['path']]
        expected=dict(source_id=src['id'],source_revision=src['revision'],origin_file=f['path'],origin_sha256=f['sha'],origin_row=rec['pos'],origin_byte_offset=rec['byte'],domain=rec['domain'],language=rec['lang'],family_id=rec['family'],split=rec['split'],own_text_tokens=rec['n'])
        if any(row[k]!=v for k,v in expected.items()) or f['sha']!=reg['sha256']:
            raise ValueError('Frozen candidate original provenance differs')
        if f['path'] not in checked:
            if file_sha(Path(f['path']))!=f['sha']:raise ValueError('Original candidate file changed')
            checked.add(f['path'])
    return rows


def source_audit(cfg,store,source,domain,language,registry,reader):
    key=digest([source['id'],domain,language,file_sha(Path(cfg['root'])/'REGISTRY.json'),file_sha(Path(cfg['prompt']))])[:24]
    folder=Path(cfg['root'])/'source_audits'/key
    cpu_progress(cfg,'calibration_select',cohort=source['id']+'/'+domain+'/'+language)
    selected,pop=sample(store,source['id'],domain,language,64,gate_clause="r.initial_gate='CANDIDATE'")
    if not selected:raise ValueError('No train candidate calibration population')
    meta=dict(kind='source',source_id=source['id'],domain=domain,language=language,population=pop,
        strata={r['h']:r['stratum'] for r in selected},registry_sha256=file_sha(Path(cfg['root'])/'REGISTRY.json'))
    # An existing PLAN pins every frozen input byte and the exact selected IDs,
    # strata, registry and population. make_batch revalidates those bindings and
    # the Batch state. Re-reading raw text cannot add evidence to this paid plan.
    cpu_progress(cfg,'calibration_frozen_plan' if (folder/'PLAN.json').exists() else 'calibration_original_read',documents=len(selected))
    rows=[] if (folder/'PLAN.json').exists() else retrieve(cfg,store,selected,registry,reader)
    cpu_progress(cfg,'calibration_binding_check');make_batch(cfg,folder,rows,meta)
    cpu_progress(cfg,'calibration_teacher_terminal_check');score(cfg,folder)
    cpu_progress(cfg,'calibration_quality_gate')
    gate=source_gate(folder,json.loads(Path(cfg['release_policy']).read_text())['quality'])
    write(folder/'QUALITY_SCOPE.json',gate);cpu_progress(cfg,'calibration_complete',status=gate['status']);return folder,gate


def release_output(batch):
    migration=batch/'CPU_MIGRATION_V15.json'
    if migration.exists():
        m=json.loads(migration.read_text());intent=Path(m['intent_path'])
        if m['job']!=batch.name or file_sha(intent)!=m['intent_sha256'] or m['status']!='CPU_ONLY_SUPERSEDED_UNCOMMITTED':raise ValueError('Migration route changed')
        if Path(m['old_output'])!=batch/'release' or Path(m['new_output'])!=batch/'release_v15':raise ValueError('Migration path differs')
        if file_sha(batch/'release/INPUT_BINDING.json')!=m['old_binding_sha256'] or file_sha(batch/'candidates.parquet')!=m['candidate_sha256']:raise ValueError('Migration evidence changed')
        if (batch/'release/READY_INTERNAL.json').exists() or (batch/'release/RELEASE_INTENT.json').exists():raise ValueError('Old release published after migration')
        return batch/'release_v15'
    return batch/'release'


def publisher_for(args):
    output=Path(args[args.index('--output')+1]);binding=output/'INPUT_BINDING.json'
    if binding.exists():
        names={Path(p).name for p in json.loads(binding.read_text())['engine_sha256']}
        for version in (15,13,11):
            name='p0_release_inventory_batch_v%d.py'%version
            if name in names:return name
        raise ValueError('Unknown historical publisher')
    return 'p0_release_inventory_batch_v15.py'


def performance_for(args,default):
    output=Path(args[args.index('--output')+1]);path=output/'INPUT_BINDING.json'
    if path.exists():
        prior=json.loads(path.read_text())
        if 'performance_config_path' in prior:
            config=Path(prior['performance_config_path'])
            if file_sha(config)!=prior['performance_config_sha256']:raise ValueError('Historical performance config changed')
            return str(config)
    return default

def process(cfg,store,source,domain,language,source_folder,ref,scope,mode,registry,reader,smoke):
    import pyarrow as pa,pyarrow.parquet as pq
    root=Path(cfg['root']);old=store.execute("SELECT j.name FROM jobs j WHERE source=? AND state='planned' AND mode=? AND EXISTS(SELECT 1 FROM records r WHERE r.job=j.name AND r.domain=? AND r.lang=?) ORDER BY j.rowid LIMIT 1",(source['id'],mode,domain,language)).fetchone()
    if old:
        name=old[0];records=store.execute('SELECT * FROM records WHERE job=?',(name,)).fetchall()
    else:
        cpu_progress(cfg,'candidate_allocate',cohort=source['id']+'/'+domain+'/'+language,mode=mode)
        new=allocate(cfg,store,source['id'],domain,language,scope,mode,smoke)
        if not new:cpu_progress(cfg,'candidate_allocate_empty');return False
        name,records=new
    batch=root/'batches'/name;batch.mkdir(parents=True,exist_ok=True);candidate=batch/'candidates.parquet'
    cpu_progress(cfg,'candidate_frozen_reuse' if candidate.exists() else 'candidate_original_read',job=name,documents=len(records))
    rows=frozen_rows(cfg,store,records,candidate,registry) if candidate.exists() else retrieve(cfg,store,records,registry,reader)
    if not candidate.exists():
        # Cheap integrity/dedup/isolation before per-document billing.
        rows,screen=EvalScreen(Path(cfg['eval_registry']),json.loads(Path(cfg['release_policy']).read_text())['evaluation']).filter(rows)
        known=sqlite3.connect('file:'+str(Path(cfg['database']).resolve())+'?mode=ro',uri=True)
        try:rows=[r for r in rows if not known.execute('SELECT 1 FROM documents WHERE body_sha=?',(r['normalized_sha256'],)).fetchone()]
        finally:known.close()
        if mode=='individual' and not candidate.exists():
            selected=cheap_near(cfg,rows)
            if selected is None:
                component('p0_build_inventory_index_v6.py',['--database',cfg['database'],'--legacy-receipt',cfg['legacy_receipt'],'--sketch-root',cfg['sketch_root'],'--output',cfg['inventory_root']])
                selected=cheap_near(cfg,rows)
            if selected is None:raise BlockingIOError('Global inventory changed during cheap screen')
            rows=selected
        elif candidate.exists():
            previous=__import__('pyarrow.parquet',fromlist=['read_table']).read_table(candidate).to_pylist()
            old_ids={r['normalized_sha256'] for r in previous}
            # Recover the exact pre-billing shortlist, never reintroduce screened rows.
            rows=[r for r in rows if r['normalized_sha256'] in old_ids]
    if not rows or any(not any(r['split']==s for r in rows) for s in ('train','validation','test')):
        with store:store.execute("UPDATE jobs SET state='HOLD_EMPTY_OR_HELDOUT_UNAVAILABLE' WHERE name=?",(name,));store.execute("UPDATE records SET gate='HOLD_ISOLATION' WHERE job=?",(name,))
        return True
    if not candidate.exists():pq.write_table(pa.Table.from_pylist(rows),candidate,compression='zstd')
    else:
        original=pq.read_table(candidate).to_pylist()
        if {r['normalized_sha256'] for r in original}!={r['normalized_sha256'] for r in rows}:raise ValueError('Interrupted candidate isolation changed; explicit inspection required')
    write(batch/'CANDIDATE_BINDING.json',dict(candidate_sha256=file_sha(candidate),registry_sha256=file_sha(root/'REGISTRY.json'),source_id=source['id'],domain=domain,language=language))
    labels={r['normalized_sha256']:layer(r) for r in rows};population=Counter(labels.values())
    if mode=='individual':review=rows
    else:
        grouped=defaultdict(list)
        for r in rows:grouped[layer(r)].append(r)
        review=[r for s in sorted(grouped) for r in sorted(grouped[s],key=lambda r:digest(['rolling-v11',r['normalized_sha256']]))[:8]]
    audit=batch/'review';make_batch(cfg,audit,review,dict(kind='batch',mode=mode,candidate_sha256=file_sha(candidate),strata={r['normalized_sha256']:layer(r) for r in review},population=dict(population),source_manifest_sha256=file_sha(source_folder/'PLAN.json')));score(cfg,audit)
    # Wait for the existing global index writer; do not start a duplicate build.
    with (Path(cfg['database']).parent/'production-release.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pointer=Path(cfg['inventory_root'])/'CURRENT'
    if not pointer.exists():raise BlockingIOError('Existing global index not complete')
    current=Path(pointer.read_text().strip())
    if json.loads(current.read_text())['ledger']!=ledger_snapshot(Path(cfg['database'])):
        component('p0_build_inventory_index_v6.py',['--database',cfg['database'],'--legacy-receipt',cfg['legacy_receipt'],'--sketch-root',cfg['sketch_root'],'--output',cfg['inventory_root']]);current=Path(pointer.read_text().strip())
    out=release_output(batch)
    bindings={'candidates':candidate,'source-audit':source_folder,'reference':ref,'batch-audit':audit,'eval-registry':cfg['eval_registry'],'inventory-index':current,'database':cfg['database'],'tokenizer':cfg['tokenizer'],'output':out,'source-terms':root/'terms'/source['id']/'SOURCE_TERMS.json','candidate-binding':batch/'CANDIDATE_BINDING.json','config':cfg['config_path'],'recipe':cfg['recipe'],'policy':cfg['release_policy'],'domain':domain,'mode':mode}
    component('p0_release_inventory_batch_v11.py',[v for k,x in bindings.items() for v in ('--'+k,x)])
    state='READY_INTERNAL' if (out/'READY_INTERNAL.json').exists() else 'HOLD' if (out/'HOLD.json').exists() else None
    if not state:raise ValueError('Missing measured release terminal state')
    with store:store.execute('UPDATE jobs SET state=? WHERE name=?',(state,name));store.execute('UPDATE records SET gate=? WHERE job=?',(state,name))
    if state=='READY_INTERNAL':
        write(root/'SMOKE_ACCEPTED.json',dict(version=11,stage='first_real_full_chain_pack_verified_no_mock',first_batch=name,receipt_sha256=file_sha(out/'READY_INTERNAL.json'),training_started=False)) if not (root/'SMOKE_ACCEPTED.json').exists() else None
    if mode=='source_audit':
        q=json.loads((out/'QUALITY.json').read_text());kept=set(q['kept_ids'])
        # Accepted subset may have been further deduplicated. Only quality failures
        # enter individual fallback; exact/near/eval exclusions remain terminal.
        failed=set(q['excluded_ids']);ann_ids=set(json.loads((audit/'PLAN.json').read_text())['body_ids'])
        with store:
            store.executemany("UPDATE records SET gate='CANDIDATE_FALLBACK',job=NULL WHERE h=?",[(h,) for h in failed-ann_ids])
    return True


def run(cfg):
    if cfg['shared_campaign_budget_cny']!=1000 or cfg['maximum_concurrency']!=2500 or cfg['auto_training'] is not False or cfg['distribution_eligible'] is not False:raise ValueError('Frozen API/CPU-only campaign limits changed')
    root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    def status(stage,**extra):
        atomic_status(root/'QUEUE_STATUS.json',dict(version=11,stage=stage,training_started=False,maximum_concurrency=2500,shared_budget_cny=1000,**extra))
        from small_agent.curation.full_inventory_v11 import AllStockControl
        ctl=AllStockControl(Path(cfg['database']),json.loads(Path(cfg['recipe']).read_text()))
        try:atomic_status(root/'BALANCE_STATUS.json',ctl.status())
        finally:ctl.close()
    if (root/'API_HOLD.json').exists():status('WAITING_EXPLICIT_API_CAMPAIGN_INSPECTION_NO_AUTO_RETRY');return
    if not (root/'REGISTRY.json').exists():status('WAITING_FULL_INVENTORY_REGISTRATION');return
    registry=json.loads((root/'REGISTRY.json').read_text());policy=json.loads(Path(cfg['release_policy']).read_text())['quality']
    store=open_store(root);reader=ArrowRawReader();handled=0;blocked=[]
    try:
        # Reference preparation/scoring can overlap the CPU scan, source admission cannot.
        for source in registry['sources']:
            if store.execute("SELECT COUNT(*) FROM files WHERE source=? AND state!='complete'",(source['id'],)).fetchone()[0]:continue
            cpu_progress(cfg,'enumerate_source_cohorts',source_id=source['id'])
            cohorts=source_cohorts(store,source['id'])
            for domain,language in cohorts:
                key=source['id']+'/'+domain+'/'+language;terminal=root/'holds'/digest(key)
                if terminal.exists():continue
                try:
                    status('BINDING_INTERNAL_SOURCE_TERMS',cohort=key);terms(cfg,source)
                    status('VERIFYING_INDEPENDENT_REFERENCE',cohort=key);ref=reference(cfg,domain,language,policy)
                    status('SOURCE_CALIBRATION',cohort=key);audit,scope=source_audit(cfg,store,source,domain,language,registry,reader)
                    for mode in ('source_audit','individual'):
                        for _ in range(cfg['maximum_batches_per_tick']):
                            if shutil.disk_usage(root).free<cfg['minimum_disk_free_bytes']:status('WAITING_DISK_BACKPRESSURE');return
                            smoke=not (root/'SMOKE_ACCEPTED.json').exists()
                            status('PROCESSING_FULL_STOCK_BATCH',cohort=key,mode=mode,smoke=smoke)
                            if not process(cfg,store,source,domain,language,audit,ref,scope,mode,registry,reader,smoke):break
                            handled+=1
                            if handled>=cfg['maximum_batches_per_tick']:status('TICK_COMPLETE_CONTINUOUS_FULL_INVENTORY',processed_batches=handled);return
                except BlockingIOError:status('WAITING_SHARED_INVENTORY_OR_API_WRITER',cohort=key);return
                except CampaignHeld as exc:
                    write(root/'API_HOLD.json',dict(cohort=key,reason=str(exc),automatic_resend=False));status('WAITING_EXPLICIT_API_CAMPAIGN_INSPECTION_NO_AUTO_RETRY',cohort=key);return
                except Exception as exc:
                    # Unknown/budget/API state stays terminal for this cohort; never resend until human inspection.
                    write(terminal,dict(stage='EXPLICIT_COHORT_HOLD_NO_AUTO_API_RETRY',cohort=key,reason=str(exc),failure_type=type(exc).__name__,training_started=False));blocked.append(key)
                    print('COHORT_HOLD',key,type(exc).__name__,str(exc),flush=True)
        status('FULL_INVENTORY_TICK_COMPLETE_WITH_EXPLICIT_HOLDS',processed_batches=handled,newly_held_cohorts=blocked)
    finally:store.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--performance-config',type=Path,required=True);a=p.parse_args()
    perf=json.loads(a.performance_config.read_text());workers=perf['release_cpu_workers']
    if type(workers) is not int or not 1<=workers<=16:raise ValueError('Invalid CPU workers')
    os.environ['P0_PARALLEL_CONFIG']=str(a.performance_config.resolve());cfg=json.loads(a.config.read_text());cfg['config_path']=str(a.config.resolve())
    component_original=original.component
    def dispatch(script,args):
        if script=='p0_build_inventory_index_v6.py':script='p0_build_inventory_index_v13.py'
        prior=os.environ['P0_PARALLEL_CONFIG']
        if script=='p0_release_inventory_batch_v11.py':
            script=publisher_for(args);os.environ['P0_PARALLEL_CONFIG']=performance_for(args,prior)
        try:return component_original(script,args)
        finally:os.environ['P0_PARALLEL_CONFIG']=prior
    global component
    component=dispatch;original.component=dispatch;original.process=process;original.RawReader=ArrowRawReader;original.source_audit=source_audit
    root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('EXISTING_QUEUE_NO_DUPLICATE_START');return
        # Resume explicitly migrated paid batches first, before allocating any new work.
        store=open_store(root)
        try:
            registry=json.loads((root/'REGISTRY.json').read_text());sources={s['id']:s for s in registry['sources']};reader=ArrowRawReader()
            for receipt in sorted((root/'batches').glob('*/CPU_MIGRATION_V15.json')):
                m=json.loads(receipt.read_text());job=store.execute('SELECT * FROM jobs WHERE name=?',(m['job'],)).fetchone()
                if not job or job['state']!='planned':continue
                args=json.loads(Path(m['intent_path']).read_text())['arguments'];cb=json.loads(Path(args['candidate-binding']).read_text())
                scope=source_gate(Path(args['source-audit']),json.loads(Path(cfg['release_policy']).read_text())['quality'])
                process(cfg,store,sources[job['source']],cb['domain'],cb['language'],Path(args['source-audit']),Path(args['reference']),scope,job['mode'],registry,reader,False)
        finally:store.close()
        run(cfg)

if __name__=='__main__':main()
