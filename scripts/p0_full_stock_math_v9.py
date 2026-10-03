"""Versioned all-host math calibration and per-batch sampling, no self-approval."""
from collections import Counter, defaultdict
import heapq
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.release_gates_v5 import sha,write
from small_agent.curation.supply_control import file_sha,digest
from small_agent.curation.store_v8 import Batch,prepare
from small_agent.curation.math_quality_v8 import math_quality
from small_agent.curation.full_stock_v9 import length_layer,route,MATH_BUCKETS
from p0_stock_plan_v6 import open_plan


def offer(heaps,label,value,k):
    heap=heaps[label]
    if len(heap)<k:heapq.heappush(heap,value)
    elif value>heap[0]:heapq.heapreplace(heap,value)


def selection(db,bucket,excluded,seed,k=64):
    heaps=defaultdict(list);population=Counter()
    for h,sid,pos,n,label in db.execute("SELECT r.h,r.sid,r.pos,r.n,x.stratum FROM records r JOIN routing x USING(h) WHERE r.split='train' AND x.bucket=? AND x.state='PENDING_CALIBRATION'",(bucket,)):
        population[label]+=1
        if h in excluded:continue
        offer(heaps,label,(-int(sha(str(seed)+'|'+h),16),h,sid,pos,n),k)
    return [(label,h,sid,pos,n) for label,heap in sorted(heaps.items()) for _,h,sid,pos,n in sorted(heap,reverse=True)],dict(population)


def excluded_ids(config):
    excluded=set()
    for p in config['excluded_cohort_files']:
        for line in Path(p).read_text().splitlines():excluded.add(json.loads(line)['document_id'])
    return excluded


def document(row,label,tokenizer):
    return dict(document_id=row['normalized_sha256'],document_text=row['text'],source_id=row['source_id'],
                source_locator=row['source_locator'],source_revision=row['source_revision'],family_id=row['family_id'],
                language='en',content_kind='text',token_count=row['own_text_tokens'],tokenizer_sha256=tokenizer,
                license_status='source_level_internal_scope_authorized',sampling_stratum=label,current_bucket='math_en')


def read_controls(db,selected,tokenizer):
    import pyarrow.parquet as pq
    wanted=defaultdict(dict)
    for label,h,sid,pos,n in selected:wanted[sid][pos]=(label,h,n)
    docs=[];sources=[]
    for sid,positions in wanted.items():
        path,h=db.execute('SELECT path,sha FROM sources WHERE id=?',(sid,)).fetchone();path=Path(path)
        if file_sha(path)!=h:raise ValueError('Full-stock source bytes changed')
        # Selected controls only; streaming prevents loading source bodies into RAM.
        i=0
        for b in pq.ParquetFile(path).iter_batches(batch_size=256):
            for r in b.to_pylist():
                if i in positions:
                    label,body,n=positions[i]
                    if sha(r['text'])!=body or r['own_text_tokens']!=n:raise ValueError('Control pointer changed')
                    docs.append(document(r,label,tokenizer))
                i+=1
        if file_sha(path)!=h:raise ValueError('Source changed during control read')
        sources.append(dict(path=str(path),sha256=h))
    if len(docs)!=len(selected):raise ValueError('Missing full-body controls')
    return sorted(docs,key=lambda d:d['document_id']),sources


def prepare_source(config,bucket,output):
    if bucket not in MATH_BUCKETS:raise ValueError('Unknown all-host source tier')
    stock=Path(config['stock']);frozen=stock/'SCAN_COMPLETE.json'
    # The scan is immutable. Actual selected population is recomputed on resume.
    receipt=json.loads(frozen.read_text());db=open_plan(stock)
    seed=config['source_sampling_seed'];excluded=excluded_ids(config)
    selected,pop=selection(db,bucket,excluded,seed)
    identity=dict(source_id='finemath_4plus_supply_v4',full_stock_scan_sha256=file_sha(frozen),bucket=bucket,sampling_seed=seed,
                  excluded_cohort_sha256={p:file_sha(Path(p)) for p in config['excluded_cohort_files']})
    try:
        if (output/'SAMPLE_PLAN.json').exists():
            plan=json.loads((output/'SAMPLE_PLAN.json').read_text())
            if plan['request_identity']!=identity or plan['population']!=pop:raise ValueError('Immutable full-stock calibration changed')
            actual=[json.loads(l) for l in (output/'selection.jsonl').read_text().splitlines()]
            expected=[dict(stratum=l,document_id=h,source_id=sid,source_row=p,body_tokens=n) for l,h,sid,p,n in selected]
            if actual!=expected:raise ValueError('Paid cohort differs from recomputed full population')
            for n,h in plan['frozen_files'].items():
                if file_sha(output/n)!=h:raise ValueError('Frozen paid source files changed')
            return sum(pop.values())
        if not selected:return 0
        docs,files=read_controls(db,selected,config['tokenizer_sha256']);output.mkdir(parents=True,exist_ok=True)
        ref=Path(config['quality_reference']);registry=json.loads((ref/'batch/judges.json').read_text())
        inp=output/'input.jsonl';inp.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs))
        selected_file=output/'selection.jsonl'
        selected_file.write_text(''.join(json.dumps(dict(stratum=l,document_id=h,source_id=sid,source_row=p,body_tokens=n))+'\n' for l,h,sid,p,n in selected))
        write(output/'judges.json',registry)
        if not (output/'batch/MANIFEST.json').exists():prepare(inp,output/'batch',ref/'batch/policy.json',ref/'batch/schema.json',ref/'batch/prompt.txt',output/'judges.json')
        # Bind ALL math shards, not just the ones from which the small sample happened to draw.
        baseline=json.loads(Path(config['baseline_source_plan']).read_text())
        write(output/'SAMPLE_PLAN.json',dict(version=9,request_identity=identity,source_complete_in_plan=True,
              input_source_files=baseline['input_source_files'],population=pop,frozen_files={n:file_sha(output/n) for n in ('input.jsonl','selection.jsonl','judges.json')},
              pointer_source_bindings=files,selected_documents=len(docs),original_cohort_overlap=len(excluded&{d['document_id'] for d in docs}),
              scope='ALL original math shards, one deterministic structural tier; not a site whitelist or per-document factual certification'))
        return sum(pop.values())
    finally:db.close()


def source_scope(config,bucket,output):
    prepare_source(config,bucket,output) # recompute and check immutable cohort before approval
    reference=Path(config['quality_reference']);stock=Path(config['stock']);policy=json.loads(Path(config['policy']).read_text())
    s=math_quality(output,reference/'REFERENCE.json',reference/'batch',policy['quality'])
    s.update(bucket=bucket,scan_receipt_sha256=file_sha(stock/'SCAN_COMPLETE.json'),whole_inventory_approved=False,
             training_started=False,source_level_internal_only=True)
    write(output/'QUALITY_SCOPE.json',s);return s


def allocate_one(config,bucket,scope,limit):
    """Only one bounded batch is allocated on demand; no up-front audit of 10B bodies."""
    if scope['status']!='passed' or limit<=0:return None
    stock=Path(config['stock']);db=open_plan(stock);key=digest(dict(bucket=bucket,scope=scope))
    try:
        prefix='math_full_'+bucket+'_%_v9'
        old=db.execute("SELECT id,name FROM jobs WHERE name LIKE ? AND state NOT IN ('READY_INTERNAL','HOLD','FAILED_NO_AUTO_API_RETRY') ORDER BY id LIMIT 1",(prefix,)).fetchone()
        if old:return old
        bad=set(scope['known_audit_bad_body_sha256']);approved=set(scope['approved_strata'])
        # Sources are one frozen corpus; deterministic ordering follows original body hash rank.
        for sid, in db.execute("SELECT DISTINCT r.sid FROM records r JOIN routing x USING(h) WHERE x.bucket=? AND x.state='PENDING_CALIBRATION' AND r.job IS NULL ORDER BY r.sid",(bucket,)).fetchall():
            chosen=[];tokens=0
            for h,n,label in db.execute("SELECT r.h,r.n,x.stratum FROM records r JOIN routing x USING(h) WHERE r.sid=? AND r.split='train' AND r.job IS NULL AND x.bucket=? AND x.state='PENDING_CALIBRATION' ORDER BY r.rank",(sid,bucket)):
                if h in bad or label not in approved:continue
                if tokens+n>limit:continue # strict READY quota cap; leave oversized documents in reserve
                chosen.append(h);tokens+=n
                if tokens>=min(config['batch_train_body_tokens'],limit)*.95:break
            if not chosen:continue
            held={split:[h for h,label in db.execute("SELECT r.h,x.stratum FROM records r JOIN routing x USING(h) WHERE r.sid=? AND r.split=? AND r.job IS NULL AND x.bucket=? AND x.state='PENDING_CALIBRATION' ORDER BY r.rank LIMIT 512",(sid,split,bucket)) if label in approved and h not in bad] for split in ('validation','test')}
            if any(not v for v in held.values()):continue
            for values in held.values():chosen.extend(values[:64])
            name='math_full_'+bucket+'_'+digest(dict(scope=key,records=chosen))[:20]+'_v9'
            with db:
                cur=db.execute('INSERT INTO jobs(sid,name,scope_sha) VALUES (?,?,?)',(sid,name,key));jid=cur.lastrowid
                db.executemany("UPDATE records SET job=?,gate='FULL_STOCK_PLANNED_NOT_READY' WHERE h=?",[(jid,h) for h in chosen])
            return jid,name
        return None
    finally:db.close()


def batch_audit(config,source,candidates,output):
    import pyarrow.parquet as pq
    s=Batch(source/'batch')
    try:manifest,state=s.manifest_hash,s.state_fingerprint()
    finally:s.close()
    heaps=defaultdict(list);pop=Counter();by_id={}
    for b in pq.ParquetFile(candidates).iter_batches(batch_size=128):
        for r in b.to_pylist():
            if r['split']!='train':continue
            layer=length_layer(r['own_text_tokens'],r['language']);pop[layer]+=1
            h=r['normalized_sha256'];offer(heaps,layer,(-int(sha('rolling-v9|'+h),16),h),8)
    wanted={h:layer for layer,heap in heaps.items() for _,h in heap}
    for b in pq.ParquetFile(candidates).iter_batches(batch_size=128):
        for r in b.to_pylist():
            if r['normalized_sha256'] in wanted:by_id[r['normalized_sha256']]=document(r,wanted[r['normalized_sha256']],config['tokenizer_sha256'])
    if len(by_id)!=len(wanted):raise ValueError('Missing rolling controls')
    plan=dict(parquet_path=str(candidates.resolve()),parquet_sha256=file_sha(candidates),source_manifest_sha256=manifest,review_state_sha256=state,
              selected=len(wanted),population=dict(pop),per_stratum=8,source_sampling_seed='rolling-v9')
    output.mkdir(parents=True,exist_ok=True);inp=output/'input.jsonl'
    data=''.join(json.dumps(by_id[h],ensure_ascii=False)+'\n' for h in sorted(by_id))
    if inp.exists() and inp.read_text()!=data:raise ValueError('Frozen rolling sample changed')
    if not inp.exists():inp.write_text(data)
    plan['input_sha256']=file_sha(inp)
    if (output/'SAMPLE_PLAN.json').exists():
        if json.loads((output/'SAMPLE_PLAN.json').read_text())!=plan:raise ValueError('Frozen rolling evidence changed')
        return
    write(output/'judges.json',json.loads((source/'batch/judges.json').read_text()))
    if not (output/'batch/MANIFEST.json').exists():prepare(inp,output/'batch',source/'batch/policy.json',source/'batch/schema.json',source/'batch/prompt.txt',output/'judges.json')
    write(output/'SAMPLE_PLAN.json',plan)
