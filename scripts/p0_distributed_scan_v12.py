#!/usr/bin/env python3
"""Distributed CPU extraction; one primary importer, unchanged v11 release gates.

Range tasks are immutable and restartable. Network completion order never changes
registry/file/row import order. Remote nodes never receive credentials or databases.
"""
import argparse,concurrent.futures,fcntl,json,os,shlex,shutil,sqlite3,subprocess,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.full_inventory_v11 import open_store,RawReader
from small_agent.curation.supply_control import digest,file_sha
from small_agent.curation.release_gates_v5 import write
import p0_distributed_worker_v12 as worker

SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3']
CODE=['scripts/p0_distributed_worker_v12.py','src/small_agent/curation/full_inventory_v11.py','src/small_agent/curation/supply_control.py','src/small_agent/curation/release_gates_v5.py','src/small_agent/__init__.py','src/small_agent/curation/__init__.py']

def command(args,timeout=7200):
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError('command failed: '+p.stderr[-1500:])
    return p.stdout

def shell(node,line,timeout=7200):
    return command(SSH+[node['ssh'],line],timeout) if node.get('ssh') else command(['bash','-c',line],timeout)

def sync(node,src,dst,download=False):
    if node.get('ssh'):
        remote=node['ssh']+':'+str(src if download else dst)
        args=['rsync','-a','--partial','--timeout=300','-e',shlex.join(SSH)]
        command(args+[remote,str(dst)] if download else args+[str(src),remote])
    else:
        a,b=Path(src),Path(dst)
        if a.resolve()!=b.resolve():
            b.parent.mkdir(parents=True,exist_ok=True)
            if a.is_dir():shutil.copytree(a,b,dirs_exist_ok=True)
            else:shutil.copy2(a,b)

def prepare(node,cfg,file):
    r=Path(node['root']);q=shlex.quote
    shell(node,'mkdir -p '+q(str(r/'code/scripts'))+' '+q(str(r/'code/src/small_agent/curation'))+' '+q(str(r/'cache'))+' '+q(str(r/'tasks')))
    # Only public worker/package source modules. No .env, profile, ledger, or SSH material.
    for path in CODE:sync(node,ROOT/path,r/'code'/path)
    sync(node,Path(cfg['tokenizer']),r/'tokenizer.json')
    if node.get('ssh'):
        free=int(shell(node,'df -PB1 '+q(str(r))+" | awk 'NR==2 {print $4}'").strip())
        if free<2*Path(file['path']).stat().st_size+cfg['remote_free_floor_bytes']:raise ValueError('remote disk reserve')
        name=file['sha256']+Path(file['path']).suffix
        shell(node,'find '+q(str(r/'cache'))+' -maxdepth 1 -type f ! -name '+q(name)+' ! -name '+q('.'+name+'.*')+' -delete')
        sync(node,Path(file['path']),r/'cache'/name)
    return node

def task_for(cfg,source,file,start,end,byte):
    return dict(version=12,source={k:v for k,v in source.items() if k!='files'},file=file,start=start,end=end,byte=byte,
        registry_sha256=cfg['registry_sha256'],tokenizer_sha256=cfg['tokenizer_sha256'],reserved_tokens=cfg['reserved_tokens'],
        worker_sha256=file_sha(ROOT/CODE[0]),adapter_sha256=file_sha(ROOT/CODE[1]),
        helper_sha256={x:file_sha(ROOT/x) for x in CODE[2:]},versions=dict(pyarrow='24.0.0',tokenizers='0.22.2'))

def execute(node,cfg,task):
    key=worker.identity(task);r=Path(node['root']);dest=Path(cfg['distributed_root'])/'results'/key;dest.mkdir(parents=True,exist_ok=True)
    actual=dict(task,local_file=str(r/'cache'/(task['file']['sha256']+Path(task['file']['path']).suffix)) if node.get('ssh') else task['file']['path'],local_tokenizer=str(r/'tokenizer.json'))
    taskpath=dest/('task-'+node['id']+'.json');write(taskpath,actual)
    remote=r/'tasks'/key;shell(node,'mkdir -p '+shlex.quote(str(remote)))
    sync(node,taskpath,remote/'task.json')
    env='CUDA_VISIBLE_DEVICES= TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONPATH='+shlex.quote(str(r/'runtime')+':'+str(r/'code/src'))
    line=env+' nice -n 15 '+shlex.quote(node['python'])+' -u '+shlex.quote(str(r/'code'/CODE[0]))+' --task '+shlex.quote(str(remote/'task.json'))+' --output '+shlex.quote(str(remote/'output'))+' > '+shlex.quote(str(remote/'worker.log'))+' 2>&1'
    try:shell(node,line)
    except Exception as exc:
        tail=shell(node,'tail -c 2000 '+shlex.quote(str(remote/'worker.log')),30)
        raise RuntimeError(str(exc)+' '+tail) from exc
    sync(node,str(remote/'output')+'/',dest/'output',download=True)
    receipt=json.loads((dest/'output/COMPLETE.json').read_text())
    return dict(node=node['id'],task_id=key,rows=receipt['rows'],elapsed_seconds=receipt['elapsed_seconds'])

def verify_result(task,out):
    """Verify an entire range before opening any SQLite write transaction."""
    import pyarrow.parquet as pq
    out=Path(out);r=json.loads((out/'COMPLETE.json').read_text());key=worker.identity(task)
    if r['task_id']!=key or r['start']!=task['start'] or (task['end'] is not None and r['end']!=task['end']):raise ValueError('Range/identity mismatch')
    for name,wanted in [('input_sha256',task['file']['sha256']),('tokenizer_sha256',task['tokenizer_sha256']),('worker_sha256',task['worker_sha256']),('versions',task['versions'])]:
        if r[name]!=wanted:raise ValueError('Receipt binding mismatch')
    if r['training_approved'] is not False:raise ValueError('CPU result cannot approve training')
    pos=task['start'];rows=[];next_byte=task['byte'];is_jsonl=Path(task['file']['path']).suffix=='.jsonl'
    for part_index,p in enumerate(r['parts']):
        if p['name']!=f'part-{part_index:06d}.parquet':raise ValueError('Part numbering mismatch')
        name=Path(p['name'])
        if name.name!=p['name'] or name.suffix!='.parquet':raise ValueError('Unsafe result path')
        if p['task_id']!=key or p['start']!=pos or file_sha(out/name)!=p['sha256']:raise ValueError('Part hash/cursor mismatch')
        chunk=pq.read_table(out/name,use_threads=False).to_pylist()
        if p['rows']!=len(chunk) or p['end']!=pos+len(chunk):raise ValueError('Part rows mismatch')
        if chunk and chunk[0]['byte']!=next_byte:raise ValueError('Part byte start differs')
        previous_byte=-1
        for x in chunk:
            if is_jsonl and x['byte']<=previous_byte:raise ValueError('JSONL byte order differs')
            if not is_jsonl and x['byte']!=0:raise ValueError('Parquet byte pointer differs')
            previous_byte=x['byte']
            if x['pos']!=pos or type(x['byte']) is not int or x['byte']<0:raise ValueError('Position mismatch')
            if x['gate'] not in ('CANDIDATE','DROP','HOLD'):raise ValueError('Unknown gate')
            if x['h'] is not None:
                if len(x['h'])!=64 or any(c not in '0123456789abcdef' for c in x['h']):raise ValueError('Bad body hash')
                if x['rank']!=digest(['candidate-order-v11',x['h']]) or x['split'] not in ('train','validation','test'):raise ValueError('Bad rank/split')
                if x['domain'] not in ('general_en','general_zh','technical','code','math'):raise ValueError('Bad domain')
                if x['n'] is not None and (type(x['n']) is not int or x['n']<0):raise ValueError('Bad count')
                if x['gate']=='CANDIDATE' and not x['n']:raise ValueError('Unencoded candidate')
            rows.append(x);pos+=1
        if is_jsonl and chunk and p['next_byte']<=previous_byte:raise ValueError('JSONL byte end differs')
        if not is_jsonl and p['next_byte']!=0:raise ValueError('Parquet byte end differs')
        next_byte=p['next_byte']
    if r['next_byte']!=next_byte:raise ValueError('Complete byte cursor mismatch')
    if r['rows']!=len(rows) or r['end']!=pos:raise ValueError('Complete rows mismatch')
    return r,rows

def import_result(db,cfg,task,reader=None,tok=None):
    key=worker.identity(task);out=Path(cfg['distributed_root'])/'results'/key/'output'
    receipt,rows=verify_result(task,out)
    rec=db.execute('SELECT * FROM files WHERE path=?',(task['file']['path'],)).fetchone()
    if db.execute('SELECT 1 FROM meta WHERE k=?',('v12_import:'+key,)).fetchone():return False
    if rec['next_pos']!=task['start'] or rec['next_byte']!=task['byte']:raise ValueError('Import must follow committed cursor')
    # Deterministic spot recomputation in primary; downstream release re-encodes every approved document.
    from tokenizers import Tokenizer
    tok=tok or Tokenizer.from_file(cfg['tokenizer']);reader=reader or RawReader()
    for x in rows[::256]:
        try:raw=reader.row(task['file'],x['pos'],x['byte'])
        except (json.JSONDecodeError,UnicodeDecodeError):raw={'_decode_error':True}
        expected=worker.classify(raw,task,tok,x['pos'],x['byte'])
        if expected!=x:raise ValueError('Primary recomputation differs')
    stats=Counter(json.loads(rec['stats']))
    # One atomic range: records + cursor + idempotency marker. A crash is all or nothing.
    with db:
        for x in rows:
            stats['rows']+=1;stats[x['gate']+':'+x['reason']]+=1
            if x['h'] is None:continue
            n=x['n'];h=x['h'];gate=x['gate'];stats['body_tokens:'+x['split']]+=n or 0
            if db.execute('SELECT split FROM supply.documents WHERE body_sha=?',(h,)).fetchone():stats['inherited_or_READY_exact_reuse']+=1;continue
            family=db.execute('SELECT split FROM supply.legacy_families WHERE family_id=?',(x['family'],)).fetchone()
            if family and family[0]!=x['split']:gate='HOLD';stats['legacy_family_split_hold']+=1
            old=db.execute('SELECT split,n FROM records WHERE h=?',(h,)).fetchone()
            if old:
                stats['candidate_exact_duplicates']+=1
                if old[0]!=x['split']:db.execute("UPDATE records SET gate='HOLD_SPLIT_CONFLICT' WHERE h=?",(h,))
                if old[1] is not None and n is not None and old[1]!=n:raise ValueError('Same body/tokenizer different counts')
                continue
            db.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL,?)',(h,rec['id'],x['pos'],x['byte'],x['domain'],x['lang'],x['family'],x['split'],n,x['rank'],x['stratum'],gate,gate))
        db.execute("UPDATE files SET next_pos=?,next_byte=?,stats=?,state='scanning' WHERE id=?",(receipt['end'],receipt['next_byte'],json.dumps(dict(stats),sort_keys=True),rec['id']))
        db.execute('INSERT INTO meta VALUES (?,?)',('v12_import:'+key,json.dumps(dict(rows=len(rows),task_id=key))))
    return True

def status(cfg,**value):
    p=Path(cfg['distributed_root'])/'STATUS.json';p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    state=dict(version=12,updated_unix=time.time(),training_started=False,**value)
    tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2));os.replace(tmp,p)
    if not cfg.get('_probe'):
        with sqlite3.connect(Path(cfg['root'])/'stock.sqlite') as db:
            state['completed_files']=db.execute("SELECT COUNT(*) FROM files WHERE state='complete'").fetchone()[0]
            state['registered_files']=db.execute('SELECT COUNT(*) FROM files').fetchone()[0]
        state['next_pos']=state.get('committed_row')
        q=Path(cfg['root'])/'SCAN_STATUS.json';t=q.with_suffix('.v12.tmp');t.write_text(json.dumps(state,ensure_ascii=False,indent=2));os.replace(t,q)

def run(cfg,probe=False):
    root=Path(cfg['root']);reg=json.loads((root/'REGISTRY.json').read_text());cfg=dict(cfg,registry_sha256=file_sha(root/'REGISTRY.json'),_probe=probe)
    if file_sha(Path(cfg['tokenizer']))!=cfg['tokenizer_sha256']:raise ValueError('Frozen tokenizer differs')
    db=open_store(root);db.execute('ATTACH DATABASE ? AS supply',(cfg['database'],))
    binding=digest(dict(registry=cfg['registry_sha256'],adapter=file_sha(ROOT/CODE[1]),scanner=file_sha(ROOT/'scripts/p0_full_inventory_scan_v11.py')))
    if db.execute("SELECT v FROM meta WHERE k='scan_identity'").fetchone()[0]!=binding:raise ValueError('Original v11 scan identity differs')
    complete=Path(cfg['distributed_root'])/'COMPLETE.json'
    if not probe and complete.exists():
        previous=json.loads(complete.read_text())
        if previous['registry_sha256']!=cfg['registry_sha256'] or db.execute("SELECT COUNT(*) FROM files WHERE state NOT IN ('complete','hold')").fetchone()[0]:raise ValueError('Terminal receipt no longer matches inventory')
        if not (root/'SCAN_COMPLETE.json').exists():
            counts=[dict(domain=x[0],language=x[1],split=x[2],gate=x[3],documents=x[4],body_tokens=x[5]) for x in db.execute('SELECT domain,lang,split,gate,COUNT(*),SUM(n) FROM records GROUP BY domain,lang,split,gate')]
            write(root/'SCAN_COMPLETE.json',dict(version=12,scan_identity=binding,counts=counts,training_started=False,**previous))
        db.close();status(cfg,**previous);return previous
    if not probe:
        code_sha=file_sha(Path(__file__))
        write(Path(cfg['distributed_root'])/('START_'+digest(dict(code=code_sha,config={k:v for k,v in cfg.items() if not k.startswith('_')}))+'.json'),dict(version=12,coordinator_sha256=code_sha,config_sha256=digest({k:v for k,v in cfg.items() if not k.startswith('_')}),original_scan_identity=binding,registry_sha256=cfg['registry_sha256'],nodes=cfg['nodes'],training_started=False))
    counters={n['id']:dict(completed_tasks=0,completed_rows=0,errors=0,worker_seconds=0) for n in cfg['nodes']}
    from tokenizers import Tokenizer
    primary_reader=RawReader();primary_tokenizer=Tokenizer.from_file(cfg['tokenizer'])
    try:
        for source in reg['sources']:
            for file in source['files']:
                rec=db.execute('SELECT * FROM files WHERE path=?',(file['path'],)).fetchone()
                if rec['state'] in ('complete','hold'):continue
                if file_sha(Path(file['path']))!=file['sha256']:raise ValueError('Registered source changed')
                status(cfg,stage='PREPARING_HASHED_NODE_CACHES',source=source['id'],file=Path(file['path']).name,committed_row=rec['next_pos'],nodes=counters)
                nodes=[]
                for node in cfg['nodes']:
                    try:
                        for attempt in range(3):
                            try:ready=prepare(node,cfg,file);break
                            except (RuntimeError,subprocess.TimeoutExpired):
                                if attempt==2:raise
                                time.sleep(2**(attempt+1))
                        nodes.append(ready)
                    except Exception as e:
                        counters[node['id']]['errors']+=1;print('NODE_UNAVAILABLE',node['id'],str(e),flush=True)
                        if probe:raise
                if not nodes:raise RuntimeError('No CPU node available')
                import pyarrow.parquet as pq
                total=pq.ParquetFile(file['path']).metadata.num_rows if Path(file['path']).suffix=='.parquet' else None
                jsonl_ranges=[]
                if total is None:
                    # Boundary indexing is cheap I/O, not text parsing/tokenization. No line is split.
                    with open(file['path'],'rb') as stream:
                        stream.seek(rec['next_byte']);pos=rec['next_pos']
                        while True:
                            a=pos;b=stream.tell()
                            for _ in range(cfg['range_rows']):
                                if not stream.readline():break
                                pos+=1
                            if pos==a:break
                            jsonl_ranges.append((a,pos,b))
                        total=pos
                if probe:
                    task=task_for(cfg,source,file,rec['next_pos'],min(rec['next_pos']+256,total) if total is not None else rec['next_pos']+256,rec['next_byte'])
                    results=[];hashes=[]
                    # Identical content identity, physical location is not part of identity.
                    for node in nodes:
                        result=execute(node,cfg,task);r,rows=verify_result(task,Path(cfg['distributed_root'])/'results'/worker.identity(task)/'output')
                        hashes.append(digest(rows));results.append(result)
                        # Preserve receipts separately; canonical range can be reused by production.
                        shutil.copytree(Path(cfg['distributed_root'])/'results'/worker.identity(task)/'output',Path(cfg['distributed_root'])/'probes'/node['id'],dirs_exist_ok=True)
                    if len(set(hashes))!=1:raise ValueError('Cross-node classification differs')
                    result=dict(stage='THREE_NODE_PROBE_PASS_NOT_TRAINING_APPROVAL',results=results,metadata_sha256=hashes[0],rows=len(rows))
                    write(Path(cfg['distributed_root'])/('PROBE_'+worker.identity(task)+'.json'),result);print(json.dumps(result));return result
                slots=[n for index in range(max(n['workers'] for n in nodes)) for n in nodes if index<n['workers']];start=rec['next_pos'];byte=rec['next_byte']
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as pool:
                    while total is None or start<total:
                        tasks=[]
                        for k,node in enumerate(slots):
                            if jsonl_ranges:
                                pending=[x for x in jsonl_ranges if x[0]>=start]
                                if k>=len(pending):break
                                a,end,task_byte=pending[k]
                            else:
                                a=start+k*cfg['range_rows']
                                if a>=total:break
                                end=min(total,a+cfg['range_rows']);task_byte=0
                            task=task_for(cfg,source,file,a,end,task_byte);tasks.append((node,task,pool.submit(execute,node,cfg,task)))
                        status(cfg,stage='DISTRIBUTED_CPU_RUNNING_NOT_READY',source=source['id'],file=Path(file['path']).name,committed_row=start,inflight_tasks=len(tasks),nodes=counters)
                        for node,task,future in tasks:
                            try:result=future.result()
                            except Exception as e:
                                counters[node['id']]['errors']+=1;print('RANGE_RETRY_ON_PRIMARY',node['id'],worker.identity(task),str(e),flush=True)
                                # Reuse identical task; CPU re-extraction only, no paid API retries.
                                primary=next(n for n in nodes if not n.get('ssh'));result=execute(primary,cfg,task);node=primary
                            import_result(db,cfg,task,primary_reader,primary_tokenizer);r=json.loads((Path(cfg['distributed_root'])/'results'/worker.identity(task)/'output/COMPLETE.json').read_text());start=r['end'];byte=r['next_byte']
                            c=counters[node['id']];c['completed_tasks']+=1;c['completed_rows']+=r['rows'];c['worker_seconds']+=result['elapsed_seconds']
                            print('CPU_RANGE_IMPORTED',node['id'],source['id'],task['start'],start,flush=True)
                            status(cfg,stage='DISTRIBUTED_CPU_RUNNING_NOT_READY',source=source['id'],file=Path(file['path']).name,committed_row=start,nodes=counters)
                        if total is None:break
                if file_sha(Path(file['path']))!=file['sha256']:raise ValueError('Registered source changed during extraction')
                with db:db.execute("UPDATE files SET state='complete' WHERE id=?",(rec['id'],))
                for node in nodes:
                    if node.get('ssh'):
                        try:shell(node,'rm -f '+shlex.quote(str(Path(node['root'])/'cache'/(file['sha256']+Path(file['path']).suffix))))
                        except Exception:pass
                print('CPU_FILE_COMPLETE',source['id'],Path(file['path']).name,flush=True)
        state=dict(stage='ALL_REGISTERED_CPU_FILES_PROCESSED_WITH_EXPLICIT_HOLDS_NOT_QUALITY_APPROVAL',nodes=counters,registry_sha256=cfg['registry_sha256'],files=reg['files'],held_files=db.execute("SELECT COUNT(*) FROM files WHERE state='hold'").fetchone()[0])
        write(Path(cfg['distributed_root'])/'COMPLETE.json',state)
        counts=[dict(domain=x[0],language=x[1],split=x[2],gate=x[3],documents=x[4],body_tokens=x[5]) for x in db.execute('SELECT domain,lang,split,gate,COUNT(*),SUM(n) FROM records GROUP BY domain,lang,split,gate')]
        write(root/'SCAN_COMPLETE.json',dict(version=12,scan_identity=binding,counts=counts,training_started=False,**state));status(cfg,**state);return state
    finally:db.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--probe',action='store_true');a=p.parse_args();cfg=json.loads(a.config.read_text());Path(cfg['distributed_root']).mkdir(parents=True,exist_ok=True)
    control=Path.home()/'.ssh/curation-v12';control.mkdir(mode=0o700,exist_ok=True);os.chmod(control,0o700)
    if len(os.fsencode(str(control/('a'*40))))>100:raise ValueError('SSH Unix control socket path too long')
    SSH+=['-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath='+str(control/'%C')]
    with (Path(cfg['distributed_root'])/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.probe:run(cfg,True)
        else:
            with (Path(cfg['root'])/'scan.lock').open('a') as scanlock:
                fcntl.flock(scanlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                try:print(json.dumps(run(cfg)))
                except Exception as exc:
                    status(cfg,stage='DISTRIBUTED_CPU_HOLD_NOT_READY',reason=type(exc).__name__+':'+str(exc)[-1500:]);raise
