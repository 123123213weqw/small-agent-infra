#!/usr/bin/env python3
"""Pool independent future-batch spotchecks without materializing corpus batches.

Only small model-control samples are collected. Original text and SHA/revision
remain bound. After materialization, resampling is independently verified before
the paid sample can be used as that batch's rolling evidence.
"""
import heapq,json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p0_stock_plan_v6 import open_plan
from small_agent.curation.release_gates_v5 import sha,write,ParquetPointer
from small_agent.curation.stock_v6 import layer
from small_agent.curation.supply_control import file_sha,digest
from small_agent.curation.store import Batch,prepare
from p0_source_audit_v4 import registry_for
from p0_curation_deepseek_env_v3 import load_profile

def selected_metadata(db,jid):
    heaps=defaultdict(list);population=Counter()
    for h,pos,n in db.execute("SELECT h,pos,n FROM records WHERE job=? AND split='train'",(jid,)):
        name=('short' if n<=512 else 'medium' if n<=4096 else 'long')+'/en';population[name]+=1
        key=-int(sha('rolling-v6|'+h),16);item=(key,h,pos,n)
        if len(heaps[name])<8:heapq.heappush(heaps[name],item)
        elif item[:2]>heaps[name][0][:2]:heapq.heapreplace(heaps[name],item)
    return [(name,h,pos,n) for name,heap in sorted(heaps.items()) for _,h,pos,n in sorted(heap,key=lambda x:(-x[0],x[1]))],dict(population)

def prepare_pool(stock,source_audit,profile_path):
    import pyarrow.parquet as pq
    root=stock/'audits';root.mkdir(exist_ok=True);db=open_plan(stock)
    source=Batch(source_audit/'batch')
    try:ss=source.state_fingerprint();sm=source.manifest_hash;registry=source.registry
    finally:source.close()
    profile,_,_=load_profile(profile_path)
    pool_docs={};job_receipts=[]
    try:
        for sid,path,source_sha in db.execute('SELECT id,path,sha FROM sources ORDER BY path').fetchall():
            jobs=db.execute('SELECT id,name FROM jobs WHERE sid=? ORDER BY id',(sid,)).fetchall()
            if not jobs:continue
            path=Path(path)
            if file_sha(path)!=source_sha:raise ValueError('Audit pointer source changed')
            needed={};designs={}
            for jid,name in jobs:
                selected,pop=selected_metadata(db,jid);designs[jid]=(name,selected,pop)
                for label,h,pos,n in selected:needed[pos]=(jid,label,h,n)
            # Read each source once, and retain ONLY selected complete sample bodies.
            samples={};pos=0
            for b in pq.ParquetFile(path).iter_batches(batch_size=128):
                for r in b.to_pylist():
                    if pos in needed:
                        jid,label,h,n=needed[pos]
                        if sha(r['text'])!=h or r['own_text_tokens']!=n:raise ValueError('Audit row pointer changed')
                        samples[pos]=dict(document_id=h,document_text=r['text'],source_id=r['source_id'],source_locator=r['source_locator'],source_revision=r['source_revision'],family_id=r['family_id'],language='en',content_kind='text',token_count=n,tokenizer_sha256='9d0b99ab9cf20d337217c96a5b39e9ff87590eb462abcbcc8f1119c3d43911a7',license_status='source_level_internal_scope_authorized',sampling_stratum=label,current_bucket='math_en')
                    pos+=1
            for jid,(name,selected,pop) in designs.items():
                out=root/name;out.mkdir(exist_ok=True);docs=[samples[pos] for _,h,pos,n in selected]
                inp=out/'input.jsonl';data=''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs)
                if inp.exists() and inp.read_text()!=data:raise ValueError('Immutable stock audit differs')
                if not inp.exists():inp.write_text(data)
                write(out/'judges.json',registry)
                if not (out/'batch').exists():prepare(inp,out/'batch',ROOT/'configs/curation/p0_source_audit_policy_parallel_v5.json',ROOT/'schemas/p0_teacher_quality_v3.json',source_audit/'batch/prompt.txt',out/'judges.json')
                pointer=dict(version=6,job_id=jid,name=name,source_path=str(path),source_sha256=source_sha,selected=selected,population=pop,input_sha256=file_sha(inp),source_manifest_sha256=sm,review_state_sha256=ss,per_stratum=8)
                write(out/'AUDIT_POINTER_PLAN.json',pointer);job_receipts.append(dict(job_id=jid,path=str(out.resolve()),sha256=file_sha(out/'AUDIT_POINTER_PLAN.json')))
                for d in docs:
                    if d['document_id'] in pool_docs and pool_docs[d['document_id']]!=d:raise ValueError('Conflicting pooled source identity')
                    pool_docs[d['document_id']]=d
            print('PREPARED_STOCK_SOURCE_AUDITS',sid,len(jobs),len(samples),flush=True)
    finally:db.close()
    pool=stock/'audit_pool';pool.mkdir(exist_ok=True)
    if len(pool_docs)>10000:raise ValueError('Split audit controls into bounded pools before sending')
    inp=pool/'input.jsonl';data=''.join(json.dumps(pool_docs[h],ensure_ascii=False)+'\n' for h in sorted(pool_docs))
    if inp.exists() and inp.read_text()!=data:raise ValueError('Frozen audit pool differs')
    if not inp.exists():inp.write_text(data)
    policy=json.loads((ROOT/'configs/curation/p0_source_audit_policy_parallel_v5.json').read_text());policy.update(maximum_batch_documents=10000,maximum_batch_body_bytes=536870912)
    write(pool/'policy.json',policy);write(pool/'judges.json',registry)
    if not (pool/'batch').exists():prepare(inp,pool/'batch',pool/'policy.json',ROOT/'schemas/p0_teacher_quality_v3.json',source_audit/'batch/prompt.txt',pool/'judges.json')
    write(stock/'AUDITS_PREPARED.json',dict(version=6,jobs=job_receipts,pooled_documents=len(pool_docs),maximum_concurrency=2500,scope='control samples only; no unreviewed corpus publication',paid_calls=0))
    return pool/'batch'

def link_audit(stock,jid,name,parquet):
    import pyarrow.parquet as pq
    out=stock/'audits'/name;pointer=json.loads((out/'AUDIT_POINTER_PLAN.json').read_text());db=open_plan(stock)
    try:selected,pop=selected_metadata(db,jid)
    finally:db.close()
    if json.loads(json.dumps(selected))!=pointer['selected'] or pop!=pointer['population'] or file_sha(out/'input.jsonl')!=pointer['input_sha256']:raise ValueError('Stock audit design changed')
    actual={};heaps=defaultdict(list);population=Counter()
    for b in pq.ParquetFile(parquet).iter_batches(batch_size=128):
        for r in b.to_pylist():
            if sha(r['text'])!=r['normalized_sha256']:raise ValueError('Candidate body differs')
            if r['split']!='train':continue
            h=r['normalized_sha256'];actual[h]=r;name_layer=layer(r);population[name_layer]+=1
            item=(-int(sha('rolling-v6|'+h),16),h)
            if len(heaps[name_layer])<8:heapq.heappush(heaps[name_layer],item)
            elif item>heaps[name_layer][0]:heapq.heapreplace(heaps[name_layer],item)
    wanted={h for _,h,_,_ in selected};picked={h for heap in heaps.values() for _,h in heap}
    if wanted!=picked or dict(population)!=pop:raise ValueError('Materialized batch does not match independently planned audit')
    for line in (out/'input.jsonl').read_text().splitlines():
        d=json.loads(line)
        if d['document_text']!=actual[d['document_id']]['text']:raise ValueError('Paid audit source text changed')
    write(out/'SAMPLE_PLAN.json',dict(parquet_path=str(parquet.resolve()),parquet_sha256=file_sha(parquet),input_sha256=file_sha(out/'input.jsonl'),source_manifest_sha256=pointer['source_manifest_sha256'],review_state_sha256=pointer['review_state_sha256'],selected=len(selected),population=pop,per_stratum=8,pointer_plan_sha256=file_sha(out/'AUDIT_POINTER_PLAN.json'),scope='independently recomputed full-body stock batch hash sample'))
    return out
