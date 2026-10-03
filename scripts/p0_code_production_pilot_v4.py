#!/usr/bin/env python3
"""Real CPU production pilot. Held diagnostic packs NEVER advance READY."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.supply_control import SupplyControl, file_sha, digest
from p0_source_audit_v4 import snapshot_files
from p0_count_supply_candidates_v4 import LICENSES


def body_sha(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()


def publish(path,value):
    data=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
    if path.exists():
        if path.read_text()!=data:raise ValueError('Immutable output changed: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.tmp')
    with tmp.open('w') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def lexical_shingles(text):
    import xxhash
    tokens=re.findall(r'[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\s]',text)
    return {xxhash.xxh64_intdigest('\x1f'.join(tokens[i:i+5])) for i in range(max(1,len(tokens)-4))}


def near_prune(rows,cfg):
    """Case-sensitive lexical approximate retrieval + exact candidate Jaccard.

    Scope is this selected pilot, NOT a global or exhaustive near-dup claim.
    """
    parent=list(range(len(rows)));sets=[];inverted=defaultdict(list);counts=Counter();dense=set()
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):parent[find(a)]=find(b)
    for i,r in enumerate(rows):
        values=lexical_shingles(r['text']);sets.append(values)
        sketch=heapq.nsmallest(cfg['near_sketch_size'],values)
        candidates=Counter(j for h in sketch for j in inverted[h])
        candidates=[j for j,n in candidates.items() if n>=min(cfg['near_min_shared_sketch'],len(sketch),len(sets[j]))]
        if len(candidates)>cfg['maximum_near_candidates']:
            dense.add(i);counts['dense_retrieval_hold']+=1
        else:
            for j in candidates:
                counts['exact_jaccard_pairs_checked']+=1
                score=len(values & sets[j])/max(1,len(values | sets[j]))
                if score>=cfg['near_threshold']:union(i,j);counts['near_edges']+=1
        for h in sketch:inverted[h].append(i)
    groups=defaultdict(list)
    for i in range(len(rows)):groups[find(i)].append(i)
    kept=[];excluded=[]
    for ids in groups.values():
        eligible=[i for i in ids if i not in dense]
        if not eligible:
            excluded.extend({'body_sha256':rows[i]['normalized_sha256'],'reason':'dense_near_retrieval_hold'} for i in ids)
            continue
        # Preserve held-out membership; never move a held-out record into train.
        winner=min(eligible,key=lambda i:(rows[i]['split']=='train',-rows[i]['own_text_tokens'],rows[i]['rank']))
        kept.append(rows[winner])
        for i in ids:
            if i!=winner:excluded.append({'body_sha256':rows[i]['normalized_sha256'],'reason':'pilot_near_duplicate_or_dense_hold','representative':rows[winner]['normalized_sha256']})
    counts.update(input_documents=len(rows),retained_documents=len(kept),excluded_documents=len(excluded))
    return kept,excluded,dict(counts)


def pick_quota(rows,targets):
    totals=Counter();selected=[]
    for r in sorted(rows,key=lambda x:x['rank']):
        split=r['split']
        if totals[split]>=targets[split]:continue
        selected.append(r);totals[split]+=r['own_text_tokens']
    return selected,dict(totals)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['census-root','download-plan','tokenizer','recipe','database','source-audit','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/curation/p0_code_production_pilot_v4.json')
    p.add_argument('--allow-held-diagnostic-pack',action='store_true')
    p.add_argument('--stop-after',choices=['normalize','freeze'])
    a=p.parse_args();cfg=json.loads(a.config.read_text());recipe=json.loads(a.recipe.read_text())
    if not a.allow_held_diagnostic_pack or any(cfg[k] for k in ('allow_api','auto_training','auto_publish')):raise ValueError('Explicit non-training diagnostic pilot required')
    if file_sha(a.tokenizer)!=recipe['tokenizer_sha256']:raise ValueError('Frozen tokenizer changed')
    if shutil.disk_usage(a.output.parent if a.output.parent.exists() else a.output.parent.parent).free<recipe['minimum_disk_free_bytes']:raise ValueError('Low disk backpressure')
    files,planned=snapshot_files(a.census_root,json.loads(a.download_plan.read_text()),cfg['source_id'],False)
    a.output.mkdir(parents=True,exist_ok=True);lock=(a.output/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    binding={'config_sha256':file_sha(a.config),'plan_sha256':file_sha(a.download_plan),'tokenizer_sha256':file_sha(a.tokenizer),
        'source_audit_sha256':file_sha(a.source_audit),'sources':[{k:r[k] for k in ('source_file','source_sha256','output_relative_path','output_sha256')} for r in files],
        'producer_sha256':file_sha(Path(__file__)),'checker_sha256':file_sha(ROOT/'scripts/p0_verify_code_production_pilot_v4.py')}
    publish(a.output/'INPUT_BINDING.json',binding)
    control=SupplyControl(a.database,recipe);batch_id=control.register('code',binding)
    try:
        def stage(name,fn):
            receipt=a.output/(name+'.receipt.json')
            if receipt.exists():
                r=json.loads(receipt.read_text())
                if r['binding_sha256']!=digest(binding):raise ValueError('Wrong stage identity')
                for n,h in r['files'].items():
                    if file_sha(a.output/n)!=h:raise ValueError('Changed stage output')
                print('RESUMED_VERIFIED_STAGE',name,flush=True)
            else:
                names=fn();r={'stage':name,'binding_sha256':digest(binding),'files':{n:file_sha(a.output/n) for n in names},'training_approved':False}
                publish(receipt,r);print('COMMITTED_STAGE',name,flush=True)
            control.commit_stage(batch_id,name,[receipt,*[a.output/n for n in r['files']]])
            if a.stop_after==name:print('CONTROLLED_RECOVERY_STOP',name,flush=True);raise SystemExit(75)
        def ingest():
            publish(a.output/'ingest.json',{'sources':binding['sources'],'completed_source_shards':len(files),'planned_source_shards':planned,'raw_corpus_modified':False})
            return ['ingest.json']
        stage('ingest',ingest)
        def normalize():
            import pyarrow as pa
            import pyarrow.parquet as pq
            db=sqlite3.connect('file:'+str(a.database)+'?mode=ro',uri=True);db.execute('BEGIN')
            old={r[0] for r in db.execute('SELECT body_sha FROM documents')}
            old_families=dict(db.execute('SELECT family_id,split FROM legacy_families'))
            rejected=Counter();candidate_db=a.output/'.selection.work.sqlite';candidate_db.unlink(missing_ok=True)
            c=sqlite3.connect(candidate_db);c.execute('CREATE TABLE candidates(rank TEXT PRIMARY KEY,file INTEGER,rowno INTEGER,split TEXT,tokens INTEGER)')
            for fi,rec in enumerate(files):
                n=0;pf=pq.ParquetFile(a.census_root/rec['output_relative_path'])
                for batch in pf.iter_batches(batch_size=256):
                    for r in batch.to_pylist():
                        rowno=n;n+=1;reason=None
                        if r['policy_bucket']!='record_license_whitelisted' or r['record_license'].lower() not in LICENSES:reason='license_hold'
                        elif r['split'] not in cfg['target_body_tokens']:reason='unassigned_split_hold'
                        elif r['normalized_sha256'] in old:reason='exact_overlap_released_ledger'
                        elif r['family_id'] in old_families and old_families[r['family_id']]!=r['split']:reason='legacy_family_split_hold'
                        elif len(r['text'].encode('utf-8'))>cfg['maximum_body_bytes'] or r['own_text_tokens']>cfg['maximum_document_tokens']:reason='pilot_size_scope_hold'
                        elif not r['text'].strip() or '\x00' in r['text'] or body_sha(r['text'])!=r['normalized_sha256']:raise ValueError('Normalized body invalid')
                        if reason:rejected[reason]+=1;continue
                        rank=body_sha(str(cfg['seed'])+'|'+r['normalized_sha256'])
                        c.execute('INSERT INTO candidates VALUES (?,?,?,?,?)',(rank,fi,rowno,r['split'],r['own_text_tokens']))
                c.commit();print('INDEXED_SOURCE',rec['source_file'],n,flush=True)
            selected={};totals=Counter()
            for rank,fi,rowno,split,tokens in c.execute('SELECT * FROM candidates ORDER BY rank'):
                if totals[split]>=cfg['target_body_tokens'][split]*cfg['proposal_multiplier']:continue
                selected[(fi,rowno)]=rank;totals[split]+=tokens
            rows=[]
            for fi,rec in enumerate(files):
                n=0;path=a.census_root/rec['output_relative_path']
                for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                    for r in batch.to_pylist():
                        if (fi,n) in selected:r['rank']=selected[(fi,n)];rows.append(r)
                        n+=1
                if file_sha(path)!=rec['output_sha256']:raise ValueError('Input changed during scan')
            rows.sort(key=lambda r:r['rank'])
            pq.write_table(pa.Table.from_pylist(rows),a.output/'.proposals.tmp',compression='zstd');os.replace(a.output/'.proposals.tmp',a.output/'proposals.parquet')
            publish(a.output/'normalize.json',{'rejected':dict(rejected),'proposal_body_tokens':dict(totals),'proposal_documents':len(rows),'legacy_exact_reference_documents':len(old),'scope':'license-declared code candidates; original formatting preserved'})
            c.close();candidate_db.unlink();db.close()
            return ['proposals.parquet','normalize.json']
        stage('normalize',normalize)
        def dedup():
            import pyarrow as pa
            import pyarrow.parquet as pq
            rows=pq.read_table(a.output/'proposals.parquet').to_pylist()
            rows,excluded,counts=near_prune(rows,cfg);rows,totals=pick_quota(rows,cfg['target_body_tokens'])
            families={}
            for r in rows:
                prior=families.setdefault(r['family_id'],r['split'])
                if prior!=r['split']:raise ValueError('Selected family split conflict')
            pq.write_table(pa.Table.from_pylist(rows),a.output/'.deduplicated.tmp',compression='zstd');os.replace(a.output/'.deduplicated.tmp',a.output/'deduplicated.parquet')
            publish(a.output/'dedup.json',{'counts':counts,'body_tokens':totals,'scope':'selected-pilot case-sensitive token-5-shingles; bottom-k retrieval then exact Jaccard','global_near_reference_complete':False,'limits':'Legacy and unselected source near matches are not covered; approximate retrieval can miss pairs.'})
            publish(a.output/'dedup_exclusions.json',excluded)
            return ['deduplicated.parquet','dedup.json','dedup_exclusions.json']
        stage('dedup',dedup)
        def quality():
            audit=json.loads(a.source_audit.read_text())
            if audit.get('source_approved') is not False or audit['sample_documents']!=96:raise ValueError('Unexpected calibration evidence')
            publish(a.output/'quality.json',{'source_audit_path':str(a.source_audit),'source_audit_sha256':file_sha(a.source_audit),'signals':{k:audit[k] for k in ('valid_annotations','invalid_annotations','model_keep_signals','model_review_signals')},'independent_reference_complete':False,'source_quality_approved':False,'new_paid_api_calls':0})
            return ['quality.json']
        stage('quality',quality)
        def freeze():
            import pyarrow.parquet as pq
            rows=pq.read_table(a.output/'deduplicated.parquet').to_pylist();totals=Counter()
            path=a.output/'documents.jsonl'
            with (a.output/'.documents.tmp').open('w') as f:
                for r in rows:
                    d={'body_sha256':r['normalized_sha256'],'document_id':r['normalized_sha256'],'candidate_document_id':r['document_id'],'family_id':r['family_id'],'domain':'code','split':r['split'],'body_tokens':r['own_text_tokens'],'source_locator':r['source_locator'],'source_revision':r['source_revision'],'declared_license':r['record_license']}
                    f.write(json.dumps(d,ensure_ascii=False)+'\n');totals[r['split']]+=r['own_text_tokens']
                f.flush();os.fsync(f.fileno())
            os.replace(a.output/'.documents.tmp',path)
            publish(a.output/'FROZEN.json',{'stage':'frozen_internal_diagnostic_candidate_not_training_eligible','parquet_sha256':file_sha(a.output/'deduplicated.parquet'),'index_sha256':file_sha(path),'body_tokens':dict(totals),'documents':len(rows),'tokenizer_sha256':binding['tokenizer_sha256'],'training_eligible':False})
            return ['documents.jsonl','FROZEN.json']
        stage('freeze',freeze)
        def pack():
            import pyarrow.parquet as pq
            from tokenizers import Tokenizer
            from small_agent.data.sharded_pack import TokenizedDocument,pack_sharded
            tok=Tokenizer.from_file(str(a.tokenizer))
            def docs():
                for batch in pq.ParquetFile(a.output/'deduplicated.parquet').iter_batches(batch_size=64):
                    for r in batch.to_pylist():
                        ids=tok.encode(r['text'],add_special_tokens=False).ids
                        if len(ids)!=r['own_text_tokens'] or tok.decode(ids,skip_special_tokens=False)!=r['text']:raise ValueError('Token count/roundtrip differs')
                        yield TokenizedDocument(r['normalized_sha256'],r['split'],'code',ids)
            out=a.output/'tokenized'
            if out.exists():
                # Interrupted after atomic pack publication: independently verify before reuse.
                from small_agent.data.sharded_format import load_sharded_manifest
                m,_=load_sharded_manifest(out)
                if m['source_manifest_sha256']!=file_sha(a.output/'FROZEN.json'):raise ValueError('Existing pack identity differs')
            else:
                pack_sharded(docs(),out,sequence_length=cfg['sequence_length'],target_shard_tokens=cfg['target_shard_tokens'],vocab_size=32768,eos_token_id=2,tokenizer_sha256=binding['tokenizer_sha256'],mixture={'code':1.0},license_gate={'training_eligible':False,'quality_approved':False,'restriction':'held_non_distributable_diagnostic_only'},source_manifest_sha256=file_sha(a.output/'FROZEN.json'))
            m=json.loads((out/'manifest.json').read_text())
            return ['tokenized/manifest.json',*[f'tokenized/{s[k]}' for s in m['shards'] for k in ('bin','idx')]]
        stage('pack',pack)
        def verify():
            subprocess.run([sys.executable,str(ROOT/'scripts/p0_verify_code_production_pilot_v4.py'),'--run',str(a.output),'--tokenizer',str(a.tokenizer),'--database',str(a.database)],check=True)
            return ['VERIFY.json']
        stage('verify',verify)
        holds=['independent_source_quality_reference_missing','global_near_reference_missing','registered_benchmark_contamination_screen_missing']
        control.hold(batch_id,';'.join(holds))
        publish(a.output/'RELEASE_HOLD.json',{'batch_id':batch_id,'stage':'engineering_pipeline_verified_release_held','holds':holds,'training_approved':False,'source_approved':False,'ready_published':False,'new_ready_train_tokens':0})
        print(json.dumps({'batch_id':batch_id,'output':str(a.output),'stage':'engineering_pipeline_verified_release_held','ledger':control.status()},ensure_ascii=False,indent=2))
    finally:control.close();lock.close()

if __name__=='__main__':main()
