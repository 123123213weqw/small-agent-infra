#!/usr/bin/env python3
"""CPU-only immutable range worker. No ledger, teacher, model, or code execution."""
import argparse,fcntl,json,os,sys,time,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from small_agent.curation.full_inventory_v11 import adapter,rule,layer,FIELDS
from small_agent.curation.supply_control import file_sha,digest
from small_agent.curation.release_gates_v5 import write

def identity(task):
    return digest({k:v for k,v in task.items() if k not in ('local_file','local_tokenizer')})

def records(path,start,end,offset=0):
    if path.suffix=='.jsonl':
        with path.open('rb') as f:
            f.seek(offset);pos=start
            while end is None or pos<end:
                byte=f.tell();line=f.readline()
                if not line:break
                try:r=json.loads(line)
                except (ValueError,UnicodeDecodeError):r={'_decode_error':True}
                yield pos,byte,f.tell(),r;pos+=1
    else:
        import pyarrow.parquet as pq
        pf=pq.ParquetFile(path);base=0
        for group in range(pf.num_row_groups):
            n=pf.metadata.row_group(group).num_rows
            if base+n<=start:base+=n;continue
            if end is not None and base>=end:break
            pos=base
            for b in pf.iter_batches(batch_size=128,row_groups=[group],columns=[x for x in FIELDS if x in pf.schema_arrow.names]):
                if pos+b.num_rows<=start:pos+=b.num_rows;continue
                if end is not None and pos>=end:return
                for r in b.to_pylist():
                    if end is not None and pos>=end:return
                    if pos>=start and (end is None or pos<end):yield pos,0,0,r
                    pos+=1
            base+=n

def classify(row,task,tok,pos,byte):
    r=None
    try:
        if row.get('_decode_error'):raise ValueError('invalid_jsonl')
        r=adapter(row,task['source'],task['file'],pos,byte);gate,reason=rule(r)
        if gate=='CANDIDATE':
            ids=None
            if r['own_text_tokens'] is None:
                ids=tok.encode(r['text'],add_special_tokens=False).ids;r['own_text_tokens']=len(ids)
            if not r['own_text_tokens']:gate='DROP';reason='empty_encoded_body'
            if ids and any(x<15 for x in ids):gate='HOLD';reason='reserved_token_in_body'
            if task['source'].get('normalized') and any(x in r['text'] for x in task['reserved_tokens']):gate='HOLD';reason='reserved_token_literal'
    except (ValueError,TypeError,KeyError) as exc:gate='HOLD';reason='adapter_'+str(exc)[:90]
    result=dict(pos=pos,byte=byte,gate=gate,reason=reason,h=None,domain=None,lang=None,family=None,split=None,n=None,stratum=None,rank=None)
    if r is not None:
        n=r['own_text_tokens'];h=r['normalized_sha256']
        result.update(h=h,domain=r['domain'],lang=r['language'],family=r['family_id'],split=r['split'],n=n,stratum=layer(r) if n else 'unencoded/'+r['domain']+'/'+r['language'],rank=digest(['candidate-order-v11',h]))
    return result

def run(task,out):
    import pyarrow as pa,pyarrow.parquet as pq
    from tokenizers import Tokenizer
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    out.mkdir(parents=True,exist_ok=True);key=identity(task)
    if task['worker_sha256']!=file_sha(Path(__file__)) or task['adapter_sha256']!=file_sha(ROOT/'src/small_agent/curation/full_inventory_v11.py'):raise ValueError('Worker/adapter differs')
    for name,expected in task.get('helper_sha256',{}).items():
        if file_sha(ROOT/name)!=expected:raise ValueError('Frozen helper differs')
    path=Path(task['local_file']);tokenizer=Path(task['local_tokenizer'])
    if file_sha(path)!=task['file']['sha256'] or file_sha(tokenizer)!=task['tokenizer_sha256']:raise ValueError('Input/tokenizer SHA differs')
    versions={p:importlib.metadata.version(p) for p in ('tokenizers','pyarrow')}
    if versions!=task['versions']:raise ValueError('Runtime versions differ')
    if (out/'COMPLETE.json').exists():
        receipt=json.loads((out/'COMPLETE.json').read_text())
        if receipt['task_id']!=key:raise ValueError('Immutable task changed')
        for p in receipt['parts']:
            if file_sha(out/p['name'])!=p['sha256']:raise ValueError('Completed result changed')
        return receipt
    tok=Tokenizer.from_file(str(tokenizer));parts=[];start=task['start'];byte=task['byte'];count=0;t=time.monotonic()
    # Atomic part receipts are cursor commits; interrupted uncommitted output is rebuilt.
    for marker in sorted(out.glob('part-*.json')):
        m=json.loads(marker.read_text());p=out/m['name']
        if m['task_id']!=key or m['start']!=start or file_sha(p)!=m['sha256']:raise ValueError('Part cursor/identity differs')
        parts.append(m);start=m['end'];byte=m['next_byte'];count+=m['rows']
    pending=[];schema=pa.schema([(k,pa.int64() if k in ('pos','byte','n') else pa.string()) for k in ('pos','byte','gate','reason','h','domain','lang','family','split','n','stratum','rank')])
    def commit(next_pos,next_byte):
        nonlocal pending,start,byte,count
        name='part-'+str(len(parts)).zfill(6)+'.parquet';p=out/name;tmp=out/(name+'.tmp')
        pq.write_table(pa.Table.from_pylist(pending,schema=schema),tmp,compression='zstd')
        with tmp.open('rb') as f:os.fsync(f.fileno())
        os.replace(tmp,p)
        m=dict(task_id=key,name=name,sha256=file_sha(p),start=start,end=next_pos,next_byte=next_byte,rows=len(pending));write(p.with_suffix('.json'),m)
        parts.append(m);count+=len(pending);start=next_pos;byte=next_byte;pending=[]
    next_pos=start;next_byte=byte
    for pos,b,end,row in records(path,start,task['end'],byte):
        pending.append(classify(row,task,tok,pos,b));next_pos=pos+1;next_byte=end
        if len(pending)>=4096:commit(next_pos,next_byte)
    if pending:commit(next_pos,next_byte)
    if task['end'] is not None and next_pos!=task['end']:raise ValueError('Range truncated')
    if file_sha(path)!=task['file']['sha256']:raise ValueError('Input changed while computing')
    receipt=dict(version=12,stage='CPU_RANGE_COMPLETE_NOT_TRAINING_APPROVAL',task_id=key,start=task['start'],end=next_pos,next_byte=next_byte,rows=count,parts=parts,input_sha256=task['file']['sha256'],tokenizer_sha256=task['tokenizer_sha256'],worker_sha256=task['worker_sha256'],versions=versions,elapsed_seconds=round(time.monotonic()-t,3),training_approved=False)
    write(out/'COMPLETE.json',receipt);return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);print(json.dumps(run(json.loads(a.task.read_text()),a.output)))
