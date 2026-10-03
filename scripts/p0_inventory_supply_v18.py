#!/usr/bin/env python3
"""v17 fair fast-lane scheduling and parallel pre-billing near checks. Frozen gates unchanged."""
import argparse,fcntl,json,os,sqlite3,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import p0_inventory_supply_v11 as original
from p0_inventory_supply_v11 import *
from small_agent.curation.release_gates_v5 import sha
from p0_arrow_reader_v15 import ArrowRawReader
from p0_parallel_near_v18 import ParallelNearIndex

_batch_timing = None



def cpu_progress(cfg,phase,**extra):
    now=time.monotonic()
    if _batch_timing is not None:
        previous=_batch_timing.get('current_phase')
        if previous:
            durations=_batch_timing['phase_seconds']
            durations[previous]=durations.get(previous,0)+now-_batch_timing['phase_started']
        _batch_timing.update(current_phase=phase,phase_started=now)
        if extra.get('job'):_batch_timing['job']=extra['job']
    if _batch_timing is not None and _batch_timing.get('job'):extra.setdefault('job',_batch_timing['job'])
    value=dict(version=18,phase=phase,updated_unix=time.time(),**extra)
    for name in ('QUEUE_CPU_PROGRESS_V18.json','QUEUE_CPU_PROGRESS_V15.json'):
        atomic_status(Path(cfg['root'])/name,value)


def cheap_near(cfg,rows):
    pointer=Path(cfg['inventory_root'])/'CURRENT'
    if not pointer.exists():raise BlockingIOError('Global index writer not complete')
    with (Path(cfg['database']).parent/'production-release.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=Path(pointer.read_text().strip());meta=json.loads(current.read_text())
        if meta['ledger']!=ledger_snapshot(Path(cfg['database'])):return None
        idx=ParallelNearIndex(Path(meta['database']).parent)
        try:return idx.filter(rows,current,json.loads(Path(cfg['release_policy']).read_text())['near'],Path(cfg['database']),cfg['_performance']['pre_billing_cpu_workers'])[0]
        finally:idx.close()


def score(cfg,folder):
    status=Path(folder)/'batch/worker/deepseek/ADAPTIVE_STATUS.json'
    if status.exists():
        s=json.loads(status.read_text())
        if s.get('stop_reason'):stopped_worker(s)
        if s['stage']=='all_selected_attempted_with_holds' and s['considered']==s['selected']:return
    if not cfg['allow_api']:raise ValueError('Explicit API authorization required')
    try:
        component('p0_deepseek_worker_v8.py',['--profile',cfg['profile'],'--batch',Path(folder)/'batch','--allow-api','--initial-concurrency',cfg['_performance']['initial_teacher_concurrency'],'--maximum-concurrency',2500,'--limit',10000])
    except subprocess.CalledProcessError as exc:
        if status.exists():stopped_worker(json.loads(status.read_text()))
        raise CampaignHeld('Paid worker nonzero exit; inspect shared account status before any new cohort, no automatic retry') from exc
    if not status.exists():raise ValueError('Worker has no terminal status')
    s=json.loads(status.read_text())
    if s.get('stop_reason'):stopped_worker(s)
    if s['considered']!=s['selected']:raise CampaignHeld('Paid worker incomplete or held by budget; no automatic retry')



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
    if mode=='source_audit' and not smoke:
        multiplier=cfg['_performance']['fast_batch_size_multiplier']
        target*=multiplier;limit*=multiplier
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
        for version in (18,17,15,13,11):
            name='p0_release_inventory_batch_v%d.py'%version
            if name in names:return name
        raise ValueError('Unknown historical publisher')
    return 'p0_release_inventory_batch_v18.py'


def performance_for(args,default):
    output=Path(args[args.index('--output')+1]);path=output/'INPUT_BINDING.json'
    if path.exists():
        prior=json.loads(path.read_text())
        if 'performance_config_path' in prior:
            config=Path(prior['performance_config_path'])
            if file_sha(config)!=prior['performance_config_sha256']:raise ValueError('Historical performance config changed')
            return str(config)
    return default

def _process_batch(cfg,store,source,domain,language,source_folder,ref,scope,mode,registry,reader,smoke):
    import pyarrow as pa,pyarrow.parquet as pq
    root=Path(cfg['root']);old=store.execute("SELECT j.name FROM jobs j WHERE source=? AND state='planned' AND mode=? AND EXISTS(SELECT 1 FROM records r WHERE r.job=j.name AND r.domain=? AND r.lang=?) ORDER BY j.rowid LIMIT 1",(source['id'],mode,domain,language)).fetchone()
    if old:
        name=old[0];records=store.execute('SELECT * FROM records WHERE job=?',(name,)).fetchall()
    else:
        cpu_progress(cfg,'candidate_allocate',cohort=source['id']+'/'+domain+'/'+language,mode=mode)
        new=allocate(cfg,store,source['id'],domain,language,scope,mode,smoke)
        if not new:cpu_progress(cfg,'candidate_allocate_empty');return False
        name,records=new
    cpu_progress(cfg,'batch_selected',job=name,mode=mode)
    if _batch_timing is not None:_batch_timing['frozen_candidate_reused']=(root/'batches'/name/'candidates.parquet').exists()
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
            cpu_progress(cfg,'pre_billing_near',job=name)
            selected=cheap_near(cfg,rows)
            if selected is None:
                component('p0_build_inventory_index_v6.py',['--database',cfg['database'],'--legacy-receipt',cfg['legacy_receipt'],'--sketch-root',cfg['sketch_root'],'--output',cfg['inventory_root']])
                cpu_progress(cfg,'pre_billing_near',job=name)
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
        # Preserve historical paid plans exactly. Larger new batches increase
        # samples proportionally (never reduce the original audit density).
        samples=8 if (batch/'review/PLAN.json').exists() else 8*cfg['_performance']['fast_batch_size_multiplier']
        review=[r for s in sorted(grouped) for r in sorted(grouped[s],key=lambda r:digest(['rolling-v11',r['normalized_sha256']]))[:samples]]
    cpu_progress(cfg,'teacher_binding',job=name)
    audit=batch/'review';make_batch(cfg,audit,review,dict(kind='batch',mode=mode,candidate_sha256=file_sha(candidate),strata={r['normalized_sha256']:layer(r) for r in review},population=dict(population),source_manifest_sha256=file_sha(source_folder/'PLAN.json')))
    cpu_progress(cfg,'teacher_requests',job=name);score(cfg,audit)
    cpu_progress(cfg,'pre_publish_snapshot',job=name)
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


def process(*args,**kwargs):
    global _batch_timing
    cfg=args[0];_batch_timing=dict(version=18,started_unix=time.time(),phase_seconds={},job=None)
    started=time.monotonic()
    try:return _process_batch(*args,**kwargs)
    finally:
        cpu_progress(cfg,'batch_attempt_finished',job=_batch_timing.get('job'))
        _batch_timing.update(finished_unix=time.time(),elapsed_seconds=time.monotonic()-started)
        _batch_timing.pop('phase_started',None)
        job=_batch_timing.get('job')
        if job:
            folder=Path(cfg['root'])/'batches'/job
            output=release_output(folder)
            ready=output/'READY_INTERNAL.json'
            _batch_timing['ready_internal']=ready.exists()
            if ready.exists():
                n=json.loads(ready.read_text())['unique_train_body_tokens']
                _batch_timing.update(ready_train_body_tokens=n,tokens_per_second=n/_batch_timing['elapsed_seconds'])
            path=folder/'PIPELINE_TIMING_V18.json'
            # Do not overwrite a successful measurement with a cheap terminal replay.
            prior=json.loads(path.read_text()) if path.exists() else {}
            if not prior.get('ready_internal'):atomic_status(path,_batch_timing)
        _batch_timing=None


def cohort_key(cohort):
    source,domain,language=cohort
    return source['id']+'/'+domain+'/'+language


def rotate(cohorts,last):
    keys=[cohort_key(x) for x in cohorts]
    offset=(keys.index(last)+1)%len(cohorts) if cohorts and last in keys else 0
    return cohorts[offset:]+cohorts[:offset]


def pending(store,source,domain,language,mode,approved=None):
    # Existing paid/planned work must always be recovered, never abandoned for speed.
    if store.execute("SELECT 1 FROM jobs j WHERE j.source=? AND j.mode=? AND j.state='planned' AND EXISTS(SELECT 1 FROM records r WHERE r.job=j.name AND r.domain=? AND r.lang=?) LIMIT 1",(source['id'],mode,domain,language)).fetchone():return True
    clause="r.gate='CANDIDATE' AND r.job IS NULL"
    params=[source['id'],domain,language]
    if mode=='source_audit' and approved is not None:
        if not approved:return False
        clause+=' AND r.stratum IN ('+','.join('?' for _ in approved)+')';params+=approved
    elif mode=='individual':
        clause="r.gate IN ('CANDIDATE','CANDIDATE_FALLBACK') AND r.job IS NULL"
        if approved:
            clause="(r.gate='CANDIDATE' AND r.stratum NOT IN ("+','.join('?' for _ in approved)+") OR r.gate='CANDIDATE_FALLBACK') AND r.job IS NULL";params+=approved
    query=source_query(store,clause)+" AND r.split='train' LIMIT 1"
    return store.execute('SELECT EXISTS('+query+')',params).fetchone()[0]==1


def cached_approved(cfg,source,domain,language):
    key=digest([source['id'],domain,language,file_sha(Path(cfg['root'])/'REGISTRY.json'),file_sha(Path(cfg['prompt']))])[:24]
    p=Path(cfg['root'])/'source_audits'/key/'QUALITY_SCOPE.json'
    # Scheduling hint only: reference/source gates are reverified before allocation
    # and the publisher independently rechecks every admission requirement.
    return json.loads(p.read_text())['approved_strata'] if p.exists() else None


def run(cfg):
    if cfg['shared_campaign_budget_cny']!=1000 or cfg['maximum_concurrency']!=2500 or cfg['auto_training'] is not False or cfg['distribution_eligible'] is not False:raise ValueError('Frozen API/CPU-only campaign limits changed')
    root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    def status(stage,**extra):
        atomic_status(root/'QUEUE_STATUS.json',dict(version=18,stage=stage,training_started=False,maximum_concurrency=2500,shared_budget_cny=1000,**extra))
        from small_agent.curation.full_inventory_v11 import AllStockControl
        ctl=AllStockControl(Path(cfg['database']),json.loads(Path(cfg['recipe']).read_text()))
        try:
            balance=ctl.status();atomic_status(root/'BALANCE_STATUS.json',balance)
            from p0_ready_throughput_v18 import record
            record(root,balance['unique_ready_train_body_tokens'])
        finally:ctl.close()
    if (root/'API_HOLD.json').exists():status('WAITING_EXPLICIT_API_CAMPAIGN_INSPECTION_NO_AUTO_RETRY');return
    if not (root/'REGISTRY.json').exists():status('WAITING_FULL_INVENTORY_REGISTRATION');return
    registry=json.loads((root/'REGISTRY.json').read_text());policy=json.loads(Path(cfg['release_policy']).read_text())['quality']
    state_path=root/'SCHEDULER_V18.json';registry_sha=file_sha(root/'REGISTRY.json')
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(registry_sha256=registry_sha,ticks=0,cursors={})
    if state['registry_sha256']!=registry_sha:raise ValueError('Scheduler registry changed; inspect explicitly')
    state['ticks']+=1;atomic_status(state_path,state)
    store=open_store(root);reader=ArrowRawReader();handled=0;blocked=[]
    try:
        cohorts=[]
        for source in registry['sources']:
            if store.execute("SELECT COUNT(*) FROM files WHERE source=? AND state!='complete'",(source['id'],)).fetchone()[0]:continue
            cohorts.extend((source,d,l) for d,l in source_cohorts(store,source['id']))
        priority={'math':0,'general_zh':1,'technical':2,'general_en':3,'code':4}
        cohorts.sort(key=lambda c:(priority[c[1]],c[0]['id'],c[2]))
        maximum=cfg['maximum_batches_per_tick']
        reserve=int(state['ticks']%cfg['_performance']['fallback_every_ticks']==0)
        fast_limit=maximum-reserve
        for mode in ('source_audit','individual'):
            if mode=='individual' and handled>=maximum:break
            for source,domain,language in rotate(cohorts,state['cursors'].get(mode)):
                key=source['id']+'/'+domain+'/'+language;terminal=root/'holds'/digest(key)
                if terminal.exists():continue
                approved=cached_approved(cfg,source,domain,language)
                if not pending(store,source,domain,language,mode,approved):continue
                state['cursors'][mode]=key;atomic_status(state_path,state)
                try:
                    status('BINDING_INTERNAL_SOURCE_TERMS',cohort=key,mode=mode);terms(cfg,source)
                    status('VERIFYING_INDEPENDENT_REFERENCE',cohort=key,mode=mode);ref=reference(cfg,domain,language,policy)
                    status('SOURCE_CALIBRATION',cohort=key,mode=mode);audit,scope=source_audit(cfg,store,source,domain,language,registry,reader)
                    status('PROCESSING_FULL_STOCK_BATCH',cohort=key,mode=mode,smoke=not (root/'SMOKE_ACCEPTED.json').exists())
                    if not process(cfg,store,source,domain,language,audit,ref,scope,mode,registry,reader,not (root/'SMOKE_ACCEPTED.json').exists()):continue
                    handled+=1
                    # One batch per cohort per pass; failed-stratum small jobs cannot
                    # occupy every tick before later approved sources are considered.
                    if mode=='source_audit' and handled>=fast_limit:break
                    if mode=='individual':break
                except BlockingIOError:status('WAITING_SHARED_INVENTORY_OR_API_WRITER',cohort=key);return
                except CampaignHeld as exc:
                    write(root/'API_HOLD.json',dict(cohort=key,reason=str(exc),automatic_resend=False));status('WAITING_EXPLICIT_API_CAMPAIGN_INSPECTION_NO_AUTO_RETRY',cohort=key);return
                except Exception as exc:
                    write(terminal,dict(stage='EXPLICIT_COHORT_HOLD_NO_AUTO_API_RETRY',cohort=key,reason=str(exc),failure_type=type(exc).__name__,training_started=False));blocked.append(key)
                    print('COHORT_HOLD',key,type(exc).__name__,str(exc),flush=True)
        status('TICK_COMPLETE_FAIR_FAST_LANE_AND_BOUNDED_FALLBACK',processed_batches=handled,newly_held_cohorts=blocked)
    finally:store.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--performance-config',type=Path,required=True);a=p.parse_args()
    perf=json.loads(a.performance_config.read_text());workers=perf['release_cpu_workers']
    if any(type(perf[k]) is not int for k in ('pre_billing_cpu_workers','initial_teacher_concurrency','fallback_every_ticks')) or not 1<=perf['pre_billing_cpu_workers']<=16 or not 1<=perf['initial_teacher_concurrency']<=2500 or not 1<=perf['fallback_every_ticks']<=16:raise ValueError('Invalid bounded v17 scheduling/CPU performance configuration')
    if any(perf.get(k) is not False for k in ('quality_thresholds_changed','skip_hashes','parallel_ledger_writers')):raise ValueError('v17 must preserve quality, full hashes and single ledger writer')
    if type(workers) is not int or not 1<=workers<=16:raise ValueError('Invalid CPU workers')
    if type(perf.get('fast_batch_size_multiplier')) is not int or not 1<=perf['fast_batch_size_multiplier']<=4:raise ValueError('Invalid proportional fast batch/audit multiplier')
    os.environ['P0_PARALLEL_CONFIG']=str(a.performance_config.resolve());cfg=json.loads(a.config.read_text());cfg['config_path']=str(a.config.resolve());cfg['_performance']=perf
    component_original=original.component
    def dispatch(script,args):
        if script=='p0_build_inventory_index_v6.py':script='p0_build_inventory_index_v18.py'
        prior=os.environ['P0_PARALLEL_CONFIG']
        if script=='p0_release_inventory_batch_v11.py':
            script=publisher_for(args);os.environ['P0_PARALLEL_CONFIG']=performance_for(args,prior)
        phase='inventory_update' if script=='p0_build_inventory_index_v18.py' else 'publish' if script.startswith('p0_release_inventory_batch_') else None
        if phase:cpu_progress(cfg,phase)
        try:return component_original(script,args)
        finally:
            os.environ['P0_PARALLEL_CONFIG']=prior
            if phase:cpu_progress(cfg,phase+'_complete')
    global component
    component=dispatch;original.component=dispatch;original.process=process;original.RawReader=ArrowRawReader;original.source_audit=source_audit;original.score=score;original.cheap_near=cheap_near
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
