#!/usr/bin/env python3
"""Persistent full-inventory QA and verified publication, same ledger/budget; no training."""
import argparse,fcntl,json,os,shutil,sqlite3,subprocess,sys,time,re,urllib.request
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.full_inventory_v11 import open_store,RawReader,adapter,layer
from small_agent.curation.quality_v11 import control_rows,make_batch,reference_gate,source_gate,CODE
from small_agent.curation.release_gates_v5 import write,bound_json,ledger_snapshot,EvalScreen
from small_agent.curation.supply_control import file_sha,digest
from small_agent.curation.stock_v6 import DiskNearIndex
from p0_curation_worker_v3 import atomic_status

class CampaignHeld(ValueError):pass

def stopped_worker(s):
    c=s.get('counts',{});ledger=s.get('ledger',{})
    if (s.get('stop_reason')=='pilot_not_valid' and c.get('invalid_outputs_quarantined',0)>0
        and not c.get('unknown_or_fatal',0) and not c.get('rejected_attempts_held',0)
        and not ledger.get('requests_by_state',{}).get('inflight',0) and ledger.get('cap_exceeded') is False):
        raise ValueError('Known returned invalid output: cohort HOLD, never reissue this protocol; other cohorts may continue')
    raise CampaignHeld('Paid worker stopped; account inspection required, no automatic retry')

def component(script,args):
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'})

def cheap_near(cfg,rows):
    # Release repeats this check against its own locked, current snapshot.
    pointer=Path(cfg['inventory_root'])/'CURRENT'
    if not pointer.exists():raise BlockingIOError('Global index writer not complete')
    with (Path(cfg['database']).parent/'production-release.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=Path(pointer.read_text().strip());meta=json.loads(current.read_text())
        if meta['ledger']!=ledger_snapshot(Path(cfg['database'])):return None
        idx=DiskNearIndex(Path(meta['database']).parent)
        try:return idx.filter(rows,current,json.loads(Path(cfg['release_policy']).read_text())['near'],Path(cfg['database']))[0]
        finally:idx.close()

def score(cfg,folder):
    status=Path(folder)/'batch/worker/deepseek/ADAPTIVE_STATUS.json'
    if status.exists():
        s=json.loads(status.read_text())
        if s.get('stop_reason'):stopped_worker(s)
        if s['stage']=='all_selected_attempted_with_holds' and s['considered']==s['selected']:return
    if not cfg['allow_api']:raise ValueError('Explicit API authorization required')
    try:
        component('p0_deepseek_worker_v8.py',['--profile',cfg['profile'],'--batch',Path(folder)/'batch','--allow-api','--initial-concurrency',64,'--maximum-concurrency',2500,'--limit',10000])
    except subprocess.CalledProcessError as exc:
        if status.exists():stopped_worker(json.loads(status.read_text()))
        raise CampaignHeld('Paid worker nonzero exit; inspect shared account status before any new cohort, no automatic retry') from exc
    if not status.exists():raise ValueError('Worker has no terminal status')
    s=json.loads(status.read_text())
    if s.get('stop_reason'):stopped_worker(s)
    if s['considered']!=s['selected']:raise CampaignHeld('Paid worker incomplete or held by budget; no automatic retry')

def terms(cfg,source):
    root=Path(cfg['root']);out=root/'terms'/source['id'];out.mkdir(parents=True,exist_ok=True);p=out/'SOURCE_TERMS.json'
    if p.exists():return bound_json(p)
    auth=bound_json(Path(cfg['terms_authorization']))
    if auth['internal_only_source_level_policy'] is not True:raise ValueError('Source-level internal authorization required')
    scope=source['terms_scope'];evidence=[Path(cfg['terms_authorization']),root/'REGISTRY.json'];license=''
    if scope=='finemath_authorized_internal':
        old=bound_json(Path(cfg['finemath_terms']));license=old['dataset_license'];evidence.append(Path(cfg['finemath_terms']))
    elif scope=='record_license_only':license='per_record_permissive_whitelist_not_dataset_license'
    elif scope.startswith('inherited_Fine'):
        old=bound_json(Path(cfg['legacy_receipt']));evidence.append(Path(cfg['legacy_receipt']));license='cc-by-sa-4.0 internal inherited source policy' if 'FineWiki' in scope else 'odc-by internal inherited source policy'
    elif source['id']=='mdn_content_en':
        # Pin the upstream LICENSE/README at the exact content revision; not an invented URL.
        dest=out/'source_license.txt'
        if not dest.exists():
            import tarfile
            manifest=Path(cfg['raw_root'])/'mdn_full_wave4_v1/audit/mdn_en_full_candidates_v1.manifest.json'
            m=json.loads(manifest.read_text());archive=Path(m['source_archive'])
            if m['source_revision']!=source['revision'] or file_sha(archive)!=m['source_archive_sha256']:raise ValueError('MDN pinned source archive differs')
            with tarfile.open(archive) as tar:
                member=tar.getmember('content-'+source['revision']+'/LICENSE.md')
                if not member.isfile() or member.size>=2000000:raise ValueError('MDN archive license invalid')
                body=tar.extractfile(member).read(2000000)
            if not body or len(body)>=2000000:raise ValueError('MDN source terms size invalid')
            dest.write_bytes(body)
            evidence.extend([manifest,archive])
        text=dest.read_text().lower()
        if 'creative commons' not in text or 'sharealike' not in text:raise ValueError('MDN terms need explicit review')
        license='mdn-content-CC-BY-SA-text/code-terms-source-level';evidence.append(dest)
    else:
        dataset=source.get('dataset');revision=source['revision']
        if not dataset or revision=='file-pinned-revision-unknown':raise ValueError('Pinned dataset terms absent')
        dest=out/'dataset_card.md'
        if not dest.exists():
            last=None
            for host in ('https://huggingface.co','https://hf-mirror.com'):
                try:
                    url=host+'/datasets/'+dataset+'/resolve/'+revision+'/README.md'
                    with urllib.request.urlopen(url,timeout=30) as response:body=response.read(2000000)
                    if not body or len(body)>=2000000:raise ValueError('Source card invalid/oversize')
                    dest.write_bytes(body);last=None;break
                except Exception as exc:last=exc
            if last:raise ValueError('Source terms unavailable; no API for unlicensed data') from last
        text=dest.read_text().lower();m=re.search(r'^license:\s*([^\n]+)',text,re.M)
        if not m:raise ValueError('Source license not unambiguously declared in pinned card')
        license=m[1].strip().strip('"\'')
        if license not in ('odc-by','apache-2.0','mit','cc-by-4.0','cc-by-sa-4.0','cc0-1.0'):raise ValueError('Special/unknown dataset terms need review:'+license)
        evidence.append(dest)
    obj=dict(version=11,status='source_level_internal_scope_authorized',source_id=source['id'],revision=source['revision'],dataset_license=license,
        internal_training_eligible=True,distribution_eligible=False,individual_contents_rights_certified=False,
        scope='explicit user source-level internal pretraining only; attribution retained; public data/weights separate review; code still per-record license gate',
        evidence_files=[dict(path=str(x.resolve()),sha256=file_sha(x)) for x in evidence])
    write(p,obj);return obj

def retrieve(cfg,store,records,registry,reader):
    sources={x['id']:x for x in registry['sources']};files={x['path']:x for s in registry['sources'] for x in s['files']};rows=[];checked=set()
    for rec in sorted(records,key=lambda x:(x['fid'],x['pos'])):
        f=store.execute('SELECT * FROM files WHERE id=?',(rec['fid'],)).fetchone();file=files[f['path']];source=sources[f['source']]
        if f['path'] not in checked:
            if file_sha(Path(f['path']))!=f['sha']:raise ValueError('Original candidate file changed')
            checked.add(f['path'])
        r=adapter(reader.row(file,rec['pos'],rec['byte']),source,file,rec['pos'],rec['byte']);r['own_text_tokens']=rec['n']
        if r['normalized_sha256']!=rec['h'] or r['split']!=rec['split'] or r['family_id']!=rec['family'] or r['domain']!=rec['domain'] or r['language']!=rec['lang']:raise ValueError('Frozen pointer/adapter/row differs')
        rows.append(r)
    return rows

def reference(cfg,domain,language,policy):
    root=Path(cfg['root'])/'references'/(domain+'_'+language+'_'+file_sha(Path(cfg['prompt']))[:12])
    if domain=='math' and language=='en':root=Path(cfg['existing_math_reference'])
    elif not (root/'PLAN.json').exists():
        rows,labels,verification=control_rows(domain,language)
        make_batch(cfg,root,rows,dict(kind='reference',domain=domain,language=language,checker_sha256=file_sha(ROOT/'src/small_agent/curation/quality_v11.py')))
        write(root/'labels.json',labels);write(root/'VERIFICATION.json',dict(controls=verification,training_eligible=False,scope='narrow syntax/format/ad/JSON controls only; no natural factual accuracy certification'))
        write(root/'REFERENCE_V11.json',dict(domain=domain,language=language,checker_sha256=file_sha(ROOT/'src/small_agent/curation/quality_v11.py'),training_eligible=False,evidence_files=[dict(path=str((root/n).resolve()),sha256=file_sha(root/n)) for n in ('labels.json','VERIFICATION.json','PLAN.json','input.jsonl')]))
    score(cfg,root);gate=reference_gate(root,json.loads((root/'labels.json').read_text()),policy);write(root/'REFERENCE_GATE_V11.json',gate)
    if gate['status']!='passed':raise ValueError('Independent reference held: '+domain+'/'+language)
    return root

def sample(store,source,domain,language,k,gate_clause="r.gate='CANDIDATE' AND r.job IS NULL",job=None):
    params=[source,domain,language];extra='' if job is None else ' AND r.job=?';params+=[] if job is None else [job]
    query='SELECT r.* FROM records r JOIN files f ON f.id=r.fid WHERE f.source=? AND r.domain=? AND r.lang=? AND '+gate_clause+extra
    counts={x[0]:x[1] for x in store.execute('SELECT stratum,COUNT(*) FROM ('+query+") WHERE split='train' GROUP BY stratum",params)}
    selected=[]
    for s in sorted(counts):selected+=store.execute('SELECT * FROM ('+query+") WHERE split='train' AND stratum=? ORDER BY rank LIMIT ?",[*params,s,k]).fetchall()
    return selected,counts

def source_audit(cfg,store,source,domain,language,registry,reader):
    key=digest([source['id'],domain,language,file_sha(Path(cfg['root'])/'REGISTRY.json'),file_sha(Path(cfg['prompt']))])[:24];root=Path(cfg['root'])/'source_audits'/key
    selected,pop=sample(store,source['id'],domain,language,64,gate_clause="r.initial_gate='CANDIDATE'")
    meta=dict(kind='source',source_id=source['id'],domain=domain,language=language,population=pop,strata={r['h']:r['stratum'] for r in selected},registry_sha256=file_sha(Path(cfg['root'])/'REGISTRY.json'))
    if not selected:raise ValueError('No train candidate calibration population')
    make_batch(cfg,root,retrieve(cfg,store,selected,registry,reader),meta);score(cfg,root)
    gate=source_gate(root,json.loads(Path(cfg['release_policy']).read_text())['quality']);write(root/'QUALITY_SCOPE.json',gate);return root,gate

def allocate(cfg,store,source,domain,language,scope,mode,smoke):
    clauses="r.gate='CANDIDATE' AND r.job IS NULL";params=[source,domain,language];approved=scope['approved_strata']
    if mode=='source_audit':
        if not approved:return None
        clauses+=' AND r.stratum IN ('+','.join('?' for _ in approved)+')';params+=approved
    else:
        if approved:
            clauses="(r.gate='CANDIDATE' AND r.stratum NOT IN ("+','.join('?' for _ in approved)+") OR r.gate='CANDIDATE_FALLBACK') AND r.job IS NULL";params+=approved
        else:clauses="r.gate IN ('CANDIDATE','CANDIDATE_FALLBACK') AND r.job IS NULL"
    q='SELECT r.* FROM records r JOIN files f ON f.id=r.fid WHERE f.source=? AND r.domain=? AND r.lang=? AND '+clauses
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

def process(cfg,store,source,domain,language,source_folder,ref,scope,mode,registry,reader,smoke):
    import pyarrow as pa,pyarrow.parquet as pq
    root=Path(cfg['root']);old=store.execute("SELECT j.name FROM jobs j WHERE source=? AND state='planned' AND mode=? AND EXISTS(SELECT 1 FROM records r WHERE r.job=j.name AND r.domain=? AND r.lang=?) ORDER BY j.rowid LIMIT 1",(source['id'],mode,domain,language)).fetchone()
    if old:
        name=old[0];records=store.execute('SELECT * FROM records WHERE job=?',(name,)).fetchall()
    else:
        new=allocate(cfg,store,source['id'],domain,language,scope,mode,smoke)
        if not new:return False
        name,records=new
    batch=root/'batches'/name;batch.mkdir(parents=True,exist_ok=True);candidate=batch/'candidates.parquet';rows=retrieve(cfg,store,records,registry,reader)
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
    out=batch/'release'
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
    store=open_store(root);reader=RawReader();handled=0;blocked=[]
    try:
        # Reference preparation/scoring can overlap the CPU scan, source admission cannot.
        for source in registry['sources']:
            if store.execute("SELECT COUNT(*) FROM files WHERE source=? AND state!='complete'",(source['id'],)).fetchone()[0]:continue
            cohorts=store.execute("SELECT DISTINCT domain,lang FROM records r JOIN files f ON f.id=r.fid WHERE f.source=? AND r.gate IN ('CANDIDATE','CANDIDATE_FALLBACK') ORDER BY domain,lang",(source['id'],)).fetchall()
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

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text());cfg['config_path']=str(a.config.resolve())
 Path(cfg['root']).mkdir(parents=True,exist_ok=True)
 with (Path(cfg['root'])/'queue.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(cfg)
  except BlockingIOError:print('EXISTING_V11_QUEUE_NO_DUPLICATE_START')
