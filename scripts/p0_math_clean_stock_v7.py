#!/usr/bin/env python3
"""Incremental full-source cheap filter and FRESH disjoint full-body calibration.

One SQLite metadata sidecar, no full corpus duplicate. Original 192 protocol-test
bodies excluded from new calibration; same source family/split/body unchanged.
"""
import argparse,json,sqlite3,sys,heapq,time
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.math_filter_v7 import structural_filter
from small_agent.curation.release_gates_v5 import sha,write
from small_agent.curation.supply_control import file_sha,digest
from small_agent.curation.evidence_ids_v8 import extension
from small_agent.curation.store_v8 import prepare,Batch
from p0_curation_worker_v3 import atomic_status

def build(stock,output,config_path):
    import pyarrow.parquet as pq
    config=json.loads(config_path.read_text());old=json.loads((stock/'STOCK_REGISTERED.json').read_text())
    output.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(output/'eligible.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA cache_size=-65536')
    db.executescript('CREATE TABLE IF NOT EXISTS eligible(h TEXT PRIMARY KEY,sid INTEGER,pos INTEGER,split TEXT,n INTEGER,stratum TEXT); CREATE INDEX IF NOT EXISTS eligible_source ON eligible(sid,pos); CREATE TABLE IF NOT EXISTS sources(sid INTEGER PRIMARY KEY,path TEXT,sha TEXT,complete INTEGER,stats TEXT); CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT);')
    code_sha=file_sha(ROOT/'src/small_agent/curation/math_filter_v7.py');identity=dict(filter_sha256=code_sha,config_sha256=file_sha(config_path),stock_binding=old['binding'])
    bound=db.execute("SELECT v FROM meta WHERE k='identity'").fetchone()
    if bound and json.loads(bound[0])!=identity:raise ValueError('Frozen structural filter changed; explicit new version required')
    with db:db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)',('identity',json.dumps(identity,sort_keys=True)))
    src=sqlite3.connect('file:'+str((stock/'stock.sqlite').resolve())+'?mode=ro',uri=True)
    files=src.execute("SELECT id,path,sha FROM sources WHERE source_id='finemath_4plus_supply_v4' AND complete=1 ORDER BY path").fetchall();src.close()
    if len(files)!=64:raise ValueError('Full mathematical source required')
    try:
        for sid,path,h in files:
            done=db.execute('SELECT sha,complete FROM sources WHERE sid=?',(sid,)).fetchone()
            if done and done[0]!=h:raise ValueError('Source identity changed')
            if done and done[1]:continue
            path=Path(path)
            if file_sha(path)!=h:raise ValueError('Frozen mathematical source changed')
            c=Counter();pos=0;rows=[]
            # Host check is first, so regex calculations never run for excluded hosts.
            for b in pq.ParquetFile(path).iter_batches(batch_size=256,columns=['text','normalized_sha256','source_locator','language','own_text_tokens','split']):
                rows=[]
                for r in b.to_pylist():
                    gate=structural_filter(r,config);c[gate]+=1;c['seen_documents']+=1
                    if gate=='structural_candidate_pass_not_quality_approval' and r['split'] in ('train','validation','test'):
                        if sha(r['text'])!=r['normalized_sha256']:raise ValueError('Source body hash differs')
                        n=r['own_text_tokens'];layer='short/en' if n<=512 else 'medium/en' if n<=4096 else 'long/en';rows.append((r['normalized_sha256'],sid,pos,r['split'],n,layer));c['passed_'+r['split']+'_documents']+=1;c['passed_'+r['split']+'_body_tokens']+=n
                    pos+=1
                with db:db.executemany('INSERT OR IGNORE INTO eligible VALUES (?,?,?,?,?,?)',rows)
                if pos%8192<256:atomic_status(output/'STATUS.json',dict(stage='filtering_original_stock_no_quality_approval',source_id=sid,source_position=pos,source_stats=dict(c),training_started=False))
            if file_sha(path)!=h:raise ValueError('Source changed during filter')
            with db:db.execute('INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?)',(sid,str(path),h,1,json.dumps(dict(c),sort_keys=True)))
            print('MATH_STRUCTURAL_FILTERED_SOURCE',sid,dict(c),flush=True)
        total=Counter()
        for s, in db.execute('SELECT stats FROM sources'):total.update(json.loads(s))
        receipt=dict(version=7,stage='complete_structural_candidate_subset_not_quality_approval',identity=identity,source_count=64,stats=dict(total),eligible_database=str((output/'eligible.sqlite').resolve()),selected_documents=db.execute('SELECT COUNT(*) FROM eligible').fetchone()[0],evidence_files=[dict(path=str(config_path.resolve()),sha256=file_sha(config_path)),dict(path=str((ROOT/'src/small_agent/curation/math_filter_v7.py').resolve()),sha256=code_sha)],training_eligible=False,distribution_eligible=False)
        write(output/'FILTER_COMPLETE.json',receipt);return receipt
    finally:db.close()

def audit(filtered,baseline,reference,output,seed=2026100207):
    import pyarrow.parquet as pq
    if (output/'SAMPLE_PLAN.json').exists():return
    frozen=json.loads((filtered/'FILTER_COMPLETE.json').read_text());db=sqlite3.connect(filtered/'eligible.sqlite');old=json.loads((baseline/'SAMPLE_PLAN.json').read_text())
    excluded={json.loads(l)['document_id'] for l in (baseline/'selection.jsonl').read_text().splitlines()};excluded.update(json.loads(l)['document_id'] for l in (reference/'input.jsonl').read_text().splitlines())
    selected=defaultdict(list);population=Counter()
    for h,sid,pos,n,stratum in db.execute("SELECT h,sid,pos,n,stratum FROM eligible WHERE split='train'"):
        population[stratum]+=1
        if h in excluded:continue
        v=(-int(sha(str(seed)+'|'+h),16),h,sid,pos,n)
        heap=selected[stratum]
        if len(heap)<64:heapq.heappush(heap,v)
        elif v>heap[0]:heapq.heapreplace(heap,v)
    wanted=defaultdict(dict);selection=[]
    for label,heap in sorted(selected.items()):
        for _,h,sid,pos,n in sorted(heap,reverse=True):wanted[sid][pos]=(label,h,n)
    docs=[]
    for sid,positions in wanted.items():
        path,h=db.execute('SELECT path,sha FROM sources WHERE sid=?',(sid,)).fetchone();path=Path(path)
        if file_sha(path)!=h:raise ValueError('Calibration pointer source changed')
        i=0
        for b in pq.ParquetFile(path).iter_batches(batch_size=128):
            for r in b.to_pylist():
                if i in positions:
                    label,body,n=positions[i]
                    if sha(r['text'])!=body:raise ValueError('Calibration pointer text changed')
                    docs.append(dict(document_id=body,document_text=r['text'],source_id=r['source_id'],source_locator=r['source_locator'],source_revision=r['source_revision'],family_id=r['family_id'],language='en',content_kind='text',token_count=n,tokenizer_sha256='9d0b99ab9cf20d337217c96a5b39e9ff87590eb462abcbcc8f1119c3d43911a7',license_status='source_level_internal_scope_authorized',sampling_stratum=label,current_bucket='math_en'))
                    selection.append(dict(document_id=body,stratum=label,source_path=str(path),source_row=i))
                i+=1
    db.close()
    if len(docs)!=sum(len(x) for x in wanted.values()):raise ValueError('Missing calibration pointers')
    output.mkdir(parents=True,exist_ok=True);inp=output/'input.jsonl';inp.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in sorted(docs,key=lambda d:d['document_id'])))
    sel=output/'selection.jsonl';sel.write_text(''.join(json.dumps(d)+'\n' for d in sorted(selection,key=lambda d:d['document_id'])))
    registry=json.loads((reference/'batch/judges.json').read_text());write(output/'judges.json',registry)
    prepare(inp,output/'batch',reference/'batch/policy.json',reference/'batch/schema.json',reference/'batch/prompt.txt',output/'judges.json')
    plan=dict(version=7,request_identity=dict(source_id='finemath_4plus_supply_v4',filter_identity=frozen['identity'],sampling_seed=seed),source_complete_in_plan=True,input_source_files=old['input_source_files'],frozen_files={'input.jsonl':file_sha(inp),'selection.jsonl':file_sha(sel),'judges.json':file_sha(output/'judges.json')},population=dict(population),fresh_disjoint_from_protocol_debug_samples=True,original_cohort_overlap=len(excluded&{d['document_id'] for d in docs}),structural_filter_receipt=str((filtered/'FILTER_COMPLETE.json').resolve()),structural_filter_receipt_sha256=file_sha(filtered/'FILTER_COMPLETE.json'),scope='ONLY deterministic structural subset, not all original FineMath candidates')
    write(output/'SAMPLE_PLAN.json',plan);print('FRESH_FILTERED_CALIBRATION_PREPARED',len(docs),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('stock','output','config','baseline','reference','audit-output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();build(a.stock,a.output,a.config);audit(a.output,a.baseline,a.reference,a.audit_output)
if __name__=='__main__':main()
