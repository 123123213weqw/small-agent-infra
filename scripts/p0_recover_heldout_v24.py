#!/usr/bin/env python3
"""Explicit immutable heldout repair; CPU-only, no API or in-place batch rewrite.

Waits for the existing queue to finish. Its cached exact-protocol teacher receipts
are reused by the unchanged v8 worker when the derived job runs normally.
"""
import argparse,fcntl,json,os,sqlite3,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import p0_inventory_supply_v24 as q
from p0_heldout_reserve_v24 import select_reserve,check_parquet
from small_agent.curation.release_gates_v5 import bound_json,write
from small_agent.curation.quality_v11 import publication_quality
from small_agent.curation.supply_control import file_sha,digest,recipe_identity
from p0_migrate_release_v15 import terminal_review

def validate_derivation(original,extra):
    if not original or not extra:raise ValueError('Empty recovery')
    identity={(r['source_id'],r['domain'],r['language']) for r in original}
    if len(identity)!=1:raise ValueError('Mixed original cohort')
    seen={r['normalized_sha256'] for r in original};families={}
    for r in original+extra:
        if families.setdefault(r['family_id'],r['split'])!=r['split']:raise ValueError('Recovery family split conflict')
    for r in extra:
        if r['split'] not in ('validation','test'):raise ValueError('Recovery must not add training records')
        if (r['source_id'],r['domain'],r['language']) not in identity:raise ValueError('Recovery cohort changed')
        if r['normalized_sha256'] in seen:raise ValueError('Duplicate recovery body')
        seen.add(r['normalized_sha256'])
    return original+extra

def validate_failed(cfg,job):
    root=Path(cfg['root']);batch=root/'batches'/job;old=batch/'release'
    if not job.startswith('batch_') or Path(job).name!=job:raise ValueError('Invalid owned job name')
    binding=bound_json(old/'INPUT_BINDING.json')
    if any((old/n).exists() for n in ('READY_INTERNAL.json','RELEASE_INTENT.json')):raise ValueError('Published batches cannot be repaired')
    paths={'candidate_sha256':batch/'candidates.parquet','candidate_binding_sha256':batch/'CANDIDATE_BINDING.json',
           'config_sha256':Path(cfg['config_path']),'policy_sha256':Path(cfg['release_policy']),
           'tokenizer_sha256':Path(cfg['tokenizer']), 'quality_sha256':old/'QUALITY.json'}
    for field,p in paths.items():
        if file_sha(p)!=binding[field]:raise ValueError('Frozen repair input changed:'+field)
    for p,h in binding['engine_sha256'].items():
        if file_sha(Path(p))!=h:raise ValueError('Frozen historical engine changed')
    for receipt in old.glob('*.receipt.json'):bound_json(receipt)
    cb=json.loads((batch/'CANDIDATE_BINDING.json').read_text());reg=root/'REGISTRY.json'
    if cb['registry_sha256']!=file_sha(reg):raise ValueError('Repair registry changed')
    terms=root/'terms'/cb['source_id']/'SOURCE_TERMS.json'
    if file_sha(terms)!=binding['source_terms_sha256']:raise ValueError('Repair source terms changed')
    before=check_parquet(old/'deduplicated.parquet',json.loads(Path(cfg['release_policy']).read_text())['sequence_length'])
    if before['status']!='needs_heldout':raise ValueError('Not a heldout-shortage failure')
    with sqlite3.connect('file:'+cfg['database']+'?mode=ro',uri=True) as db:
        bid=digest(dict(domain=cb['domain'],input_binding=binding,recipe=recipe_identity(json.loads(Path(cfg['recipe']).read_text()))))
        state=db.execute('SELECT status,reason FROM batches WHERE batch_id=?',(bid,)).fetchone()
        if not state or state[0] not in ('quarantined','waiting_gate'):raise ValueError('Not an uncommitted failed ledger batch')
        if db.execute('SELECT 1 FROM releases WHERE batch_id=?',(bid,)).fetchone():raise ValueError('Old batch already committed')
        if not (str(state[1]).startswith('V24_NEEDS_HELDOUT:') or 'validation split needs at least one sample' in str(state[1])):
            raise ValueError('Unrelated failure must not be cleared')
    source_key=digest([cb['source_id'],cb['domain'],cb['language'],file_sha(reg),file_sha(Path(cfg['prompt']))])[:24]
    source=root/'source_audits'/source_key
    reference=Path(cfg['existing_math_reference']) if cb['domain']=='math' else root/'references'/('unused')
    # References may be configured through the generic domain/language registry.
    if cb['domain']!='math':raise ValueError('Explicit recovery currently scoped to the observed English math batch')
    if cb['language']!='en':raise ValueError('Wrong arithmetic reference language')
    paid={n:terminal_review(p) for n,p in [('source',source),('reference',reference),('batch',batch/'review')]}
    quality=publication_quality(cfg,batch/'candidates.parquet',source,reference,batch/'review',binding['mode'])
    if quality['status']!='passed' or digest(quality)!=digest(json.loads((old/'QUALITY.json').read_text())):
        raise ValueError('Prior quality changed; cannot reuse its scope')
    return batch,old,binding,cb,quality,paid,before

def apply(cfg,job):
    import pyarrow as pa,pyarrow.parquet as pq
    root=Path(cfg['root'])
    with (root/'queue.lock').open('a') as queue,(Path(cfg['database']).parent/'production-release.lock').open('a') as publication:
        fcntl.flock(queue,fcntl.LOCK_EX);fcntl.flock(publication,fcntl.LOCK_EX)
        if (root/'API_HOLD.json').exists():raise ValueError('Campaign HOLD requires explicit inspection, no recovery retry')
        store=q.open_store(root)
        try:
            old_job=store.execute('SELECT * FROM jobs WHERE name=?',(job,)).fetchone()
            if not old_job:raise ValueError('Unknown original stock job')
            if old_job['state']=='SUPERSEDED_HELDOUT_V24':
                p=Path(old_job['reason']);receipt=bound_json(p);new=root/'batches'/receipt['new_job']
                if file_sha(new/'candidates.parquet')!=receipt['new_candidate_sha256']:raise ValueError('Queued recovery candidate changed')
                if json.loads((new/'CANDIDATE_BINDING.json').read_text()).get('heldout_recovery_sha256')!=file_sha(p):raise ValueError('Queued recovery binding changed')
                finalize_hold(root,new,receipt)
                print(json.dumps(dict(stage='ALREADY_QUEUED_OR_COMPLETED',new_job=receipt['new_job'])));return
            batch,old,binding,cb,quality,paid,before=validate_failed(cfg,job)
            cohort=cb['source_id']+'/'+cb['domain']+'/'+cb['language'];hold=root/'holds'/digest(cohort)
            if hold.exists():
                h=json.loads(hold.read_text())
                if h.get('failure_type')!='CalledProcessError' or job not in h.get('reason',''):raise ValueError('Unrelated cohort HOLD')
            original=pq.read_table(batch/'candidates.parquet').to_pylist();kept=set(quality['kept_ids'])
            strata=sorted({q.layer(r) for r in original if r['normalized_sha256'] in kept})
            query=q.source_query(store,"r.gate='CANDIDATE' AND r.job IS NULL")
            extra=select_reserve(store,query,[cb['source_id'],cb['domain'],cb['language']],strata)
            if extra is None:raise ValueError('No original heldout in approved strata; do not move train documents')
            registry=json.loads((root/'REGISTRY.json').read_text());rows=q.retrieve(cfg,store,extra,registry,q.ArrowRawReader())
            derived_expected=validate_derivation(original,rows)
            name='batch_'+digest(['heldout-recovery-v24',job,file_sha(old/'INPUT_BINDING.json'),[r['h'] for r in extra]])[:24]
            new=root/'batches'/name;new.mkdir(exist_ok=True)
            candidate=new/'candidates.parquet'
            if not candidate.exists():
                tmp=new/'candidates.tmp';pq.write_table(pa.Table.from_pylist(derived_expected),tmp,compression='zstd')
                with tmp.open('rb') as f:os.fsync(f.fileno())
                tmp.replace(candidate)
            derived=pq.read_table(candidate).to_pylist()
            if derived!=derived_expected:raise ValueError('Immutable repair candidate differs')
            receipt=dict(version=24,old_job=job,new_job=name,old_binding_sha256=file_sha(old/'INPUT_BINDING.json'),
                         old_candidate_sha256=file_sha(batch/'candidates.parquet'),new_candidate_sha256=file_sha(candidate),
                         old_quality_sha256=file_sha(old/'QUALITY.json'),approved_recovery_strata=strata,
                         added_heldout_ids=[r['h'] for r in extra],old_hold_path=str(hold),old_hold_sha256=file_sha(hold) if hold.exists() else None,
                         prior_split_report=before,paid_reviews=paid,api_requests=0,quality_thresholds_changed=False,
                         original_splits_preserved=True,old_files_preserved=True,source_registry_sha256=file_sha(root/'REGISTRY.json'))
            intent=new/'HELDOUT_RECOVERY_V24.json'
            if intent.exists():
                if json.loads(intent.read_text())!=receipt:raise ValueError('Repair intent changed')
            else:write(intent,receipt)
            cb_new=dict(candidate_sha256=file_sha(candidate),registry_sha256=file_sha(root/'REGISTRY.json'),
                        source_id=cb['source_id'],domain=cb['domain'],language=cb['language'],heldout_recovery_sha256=file_sha(intent))
            write(new/'CANDIDATE_BINDING.json',cb_new)
            with store:
                store.execute('BEGIN IMMEDIATE')
                store.execute('INSERT INTO jobs(name,source,mode) VALUES(?,?,?)',(name,cb['source_id'],binding['mode']))
                old_ids=[r['normalized_sha256'] for r in original]
                for body in old_ids:
                    c=store.execute('UPDATE records SET job=? WHERE h=? AND job=?',(name,body,job))
                    if c.rowcount!=1:raise ValueError('Old stock claim changed')
                for r in extra:
                    c=store.execute("UPDATE records SET job=? WHERE h=? AND job IS NULL AND gate='CANDIDATE' AND split=?",(name,r['h'],r['split']))
                    if c.rowcount!=1:raise ValueError('Heldout stock claim changed')
                store.execute("UPDATE records SET gate='HOLD_ISOLATION' WHERE job=?",(job,))
                store.execute("UPDATE jobs SET state='SUPERSEDED_HELDOUT_V24',reason=? WHERE name=?",(str(intent),job))
            finalize_hold(root,new,receipt)
            print(json.dumps(dict(stage='HELDOUT_REPAIR_QUEUED_NOT_READY',new_job=name,added_heldout_documents=len(extra),api_requests=0)))
        finally:store.close()

def finalize_hold(root,new,receipt):
    hold=Path(receipt['old_hold_path']);archive=new/'OLD_COHORT_HOLD_V24.json'
    if receipt['old_hold_sha256']:
        if archive.exists():
            if file_sha(archive)!=receipt['old_hold_sha256']:raise ValueError('Archived hold changed')
            # A new independent HOLD may now occupy the old path; never clear it.
        elif hold.exists():
            if file_sha(hold)!=receipt['old_hold_sha256']:raise ValueError('Unrelated new hold must not be cleared')
            hold.rename(archive)
        else:raise ValueError('Old HOLD missing without archive')
    write(new/'RECOVERY_APPLIED_V24.json',dict(stage='QUEUED_NOT_READY',intent_sha256=file_sha(new/'HELDOUT_RECOVERY_V24.json'),api_requests=0))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--old-job',required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());cfg['config_path']=str(a.config.resolve());apply(cfg,a.old_job)

if __name__=='__main__':main()
