#!/usr/bin/env python3
"""Three bounded file-affinity CPU lanes; ordered primary import, unchanged worker and gates."""
import argparse,concurrent.futures,fcntl,json,os,shlex,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import p0_distributed_scan_v18 as base
from small_agent.curation.full_inventory_v11 import open_store
from small_agent.curation.supply_control import digest,file_sha
from small_agent.curation.release_gates_v5 import write
from p0_arrow_reader_v15 import ArrowRawReader
from p0_parallel_cpu_v13 import ordered_map


def process_file(node,cfg,source,file,cursor):
    # Exactly one file per node at a time: the old bounded cache deletion remains safe.
    base.prepare(node,cfg,file)
    import pyarrow.parquet as pq
    if Path(file['path']).suffix!='.parquet':raise ValueError('v19 affinity supports registered Parquet only; use original coordinator for JSONL')
    total=pq.ParquetFile(file['path']).metadata.num_rows
    tasks=[base.task_for(cfg,source,file,start,min(start+cfg['range_rows'],total),0) for start in range(cursor,total,cfg['range_rows'])]
    def compute(task):
        result=base.execute(node,cfg,task)
        # Verify hashes/identity before marking the file compute lane complete.
        out=Path(cfg['distributed_root'])/'results'/base.worker.identity(task)/'output'
        base.verify_result(task,out)
        return task,result
    with concurrent.futures.ThreadPoolExecutor(max_workers=node['workers']) as pool:
        results=list(ordered_map(pool,compute,tasks,max(1,node['workers'])))
    return dict(node=node['id'],total=total,results=results)


def run(cfg):
    root=Path(cfg['root']);regpath=root/'REGISTRY.json';reg=json.loads(regpath.read_text())
    cfg=dict(cfg,registry_sha256=file_sha(regpath))
    db=open_store(root);db.execute('ATTACH DATABASE ? AS supply',(cfg['database'],))
    binding=digest(dict(registry=cfg['registry_sha256'],adapter=file_sha(base.ROOT/base.CODE[1]),scanner=file_sha(base.ROOT/'scripts/p0_full_inventory_scan_v11.py')))
    stored=db.execute("SELECT v FROM meta WHERE k='scan_identity'").fetchone()
    if not stored or stored[0]!=binding:raise ValueError('Frozen scan identity changed')
    nodes=cfg['nodes']
    if not nodes or any(type(n['workers']) is not int or not 1<=n['workers']<=32 for n in nodes):raise ValueError('Invalid bounded worker count')
    if len({n['id'] for n in nodes})!=len(nodes):raise ValueError('Duplicate node')
    status=lambda **v:base.status(cfg,coordinator_version=19,**v)
    pending=[]
    for source in reg['sources']:
        for file in source['files']:
            row=db.execute('SELECT * FROM files WHERE path=?',(file['path'],)).fetchone()
            if row['state'] in ('complete','hold'):continue
            if Path(file['path']).suffix!='.parquet':raise ValueError('Pending non-Parquet input; no implicit skip')
            if file_sha(Path(file['path']))!=file['sha256']:raise ValueError('Registered source changed')
            pending.append((source,file,row['id'],row['next_pos']))
    write(Path(cfg['distributed_root'])/('START_AFFINITY_V19_'+digest(dict(code=file_sha(Path(__file__)),config=cfg))+'.json'),dict(version=19,config_sha256=digest(cfg),coordinator_sha256=file_sha(Path(__file__)),registry_sha256=cfg['registry_sha256'],nodes=nodes,training_started=False))
    reader=ArrowRawReader()
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(cfg['tokenizer']);completed=0;rows_imported=0;started=time.time()
    # One future/file/node. When a file is committed, that node gets its next file.
    # No completion-order-dependent corpus choices or parallel database writers.
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as pool:
            inflight=[];next_file=0
            def submit(node):
                nonlocal next_file
                if next_file>=len(pending):return
                item=pending[next_file];next_file+=1;s,f,fid,cursor=item
                future=pool.submit(process_file,node,cfg,s,f,cursor)
                inflight.append((node,item,future))
            for node in nodes:submit(node)
            while inflight:
                node,item,future=inflight.pop(0);source,file,fid,cursor=item
                status(stage='FILE_AFFINITY_COMPUTE_AND_ORDERED_IMPORT_NOT_READY',source=source['id'],file=Path(file['path']).name,committed_files=completed,imported_rows=rows_imported,active_nodes=[node['id']]+[n['id'] for n,_,_ in inflight],range_workers={n['id']:n['workers'] for n in nodes})
                result=future.result() # fails closed; results/cursors remain resumable
                for task,measurement in result['results']:
                    base.import_result(db,cfg,task,reader,tok)
                    rows_imported+=measurement['rows']
                if file_sha(Path(file['path']))!=file['sha256']:raise ValueError('Registered source changed during extraction')
                if db.execute('SELECT next_pos FROM files WHERE id=?',(fid,)).fetchone()[0]!=result['total']:raise ValueError('File cursor coverage differs')
                with db:db.execute("UPDATE files SET state='complete' WHERE id=?",(fid,))
                completed+=1
                print('AFFINITY_FILE_COMPLETE',node['id'],source['id'],Path(file['path']).name,result['total'],flush=True)
                status(stage='FILE_AFFINITY_COMMITTED_NOT_READY',source=source['id'],committed_files=completed,imported_rows=rows_imported,elapsed_seconds=time.time()-started)
                # Next prepare on this node cleans only its previous hashed file cache.
                submit(node)
        status(stage='ALL_AFFINITY_PENDING_PARQUET_COMMITTED_NOT_QUALITY_APPROVAL',committed_files=completed,imported_rows=rows_imported,elapsed_seconds=time.time()-started)
    finally:db.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text())
    base.SSH+=['-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath='+str(Path.home()/'.ssh/curation-v12/%C')]
    Path(cfg['distributed_root']).mkdir(parents=True,exist_ok=True)
    with (Path(cfg['distributed_root'])/'coordinator.lock').open('a') as lock,(Path(cfg['root'])/'scan.lock').open('a') as scan:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(scan,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(cfg)
        except Exception as exc:base.status(cfg,coordinator_version=19,stage='AFFINITY_CPU_HOLD_NOT_READY',reason=type(exc).__name__+':'+str(exc)[-1500:]);raise

if __name__=='__main__':main()
