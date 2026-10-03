#!/usr/bin/env python3
"""Full source extraction/counting with atomic pointer+cursor commits and per-file isolation."""
import argparse,fcntl,json,sqlite3,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.full_inventory_v11 import adapter,rule,layer,open_store,FIELDS
from small_agent.curation.supply_control import file_sha,digest
from small_agent.curation.release_gates_v5 import write
from p0_curation_worker_v3 import atomic_status


def rows(file,start,offset):
    p=Path(file['path'])
    if p.suffix=='.jsonl':
        with p.open('rb') as f:
            f.seek(offset);pos=start
            while True:
                byte=f.tell();line=f.readline()
                if not line:break
                try:row=json.loads(line)
                except (ValueError,UnicodeDecodeError):row={'_decode_error':True}
                yield pos,byte,f.tell(),row;pos+=1
    else:
        import pyarrow.parquet as pq
        pf=pq.ParquetFile(p);pos=0;columns=[k for k in FIELDS if k in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=128,columns=columns):
            for row in batch.to_pylist():
                if pos>=start:yield pos,0,0,row
                pos+=1


def scan(cfg,limit=None):
    root=Path(cfg['root']);registry=json.loads((root/'REGISTRY.json').read_text())
    if registry['tokenizer_sha256']!=file_sha(Path(cfg['tokenizer'])):raise ValueError('Frozen tokenizer changed')
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(cfg['tokenizer']);d=open_store(root);binding=digest(dict(registry=file_sha(root/'REGISTRY.json'),adapter=file_sha(ROOT/'src/small_agent/curation/full_inventory_v11.py'),scanner=file_sha(Path(__file__))))
    d.execute('ATTACH DATABASE ? AS supply',(str(Path(cfg['database']).resolve()),))
    try:
        with d:
            old=d.execute("SELECT v FROM meta WHERE k='scan_identity'").fetchone()
            if old and old[0]!=binding:raise ValueError('Frozen scanner changed; explicit new version required')
            d.execute("INSERT OR IGNORE INTO meta VALUES ('scan_identity',?)",(binding,))
            for source in registry['sources']:
                for f in source['files']:d.execute('INSERT OR IGNORE INTO files(path,sha,source) VALUES (?,?,?)',(f['path'],f['sha256'],source['id']))
        observed=0
        for source in registry['sources']:
            for file in source['files']:
                rec=d.execute('SELECT * FROM files WHERE path=?',(file['path'],)).fetchone()
                if rec['state'] in ('complete','hold'):continue
                fid=rec['id'];stats=Counter(json.loads(rec['stats']));pending=[]
                try:
                    if file_sha(Path(file['path']))!=file['sha256']:raise ValueError('Raw/census SHA changed')
                    def commit(next_pos,next_byte):
                        nonlocal pending
                        with d:
                            for r,gate,reason,pos,byte in pending:
                                stats['rows']+=1;stats[gate+':'+reason]+=1
                                if r is None:continue
                                n=r['own_text_tokens'];stats['body_tokens:'+r['split']]+=n or 0
                                known=d.execute('SELECT split FROM supply.documents WHERE body_sha=?',(r['normalized_sha256'],)).fetchone()
                                if known:stats['inherited_or_READY_exact_reuse']+=1;continue
                                family=d.execute('SELECT split FROM supply.legacy_families WHERE family_id=?',(r['family_id'],)).fetchone()
                                if family and family[0]!=r['split']:gate='HOLD';stats['legacy_family_split_hold']+=1
                                h=r['normalized_sha256'];old=d.execute('SELECT split,n FROM records WHERE h=?',(h,)).fetchone()
                                if old:
                                    stats['candidate_exact_duplicates']+=1
                                    if old[0]!=r['split']:d.execute("UPDATE records SET gate='HOLD_SPLIT_CONFLICT' WHERE h=?",(h,))
                                    if old[1] is not None and n is not None and old[1]!=n:raise ValueError('Same body/tokenizer different counts')
                                    continue
                                d.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL,?)',(h,fid,pos,byte,r['domain'],r['language'],r['family_id'],r['split'],n,sha_rank(h),layer(r) if n else 'unencoded/'+r['domain']+'/'+r['language'],gate,gate))
                            d.execute("UPDATE files SET next_pos=?,next_byte=?,stats=?,state='scanning' WHERE id=?",(next_pos,next_byte,json.dumps(dict(stats),sort_keys=True),fid))
                        pending=[]
                        atomic_status(root/'SCAN_STATUS.json',dict(version=11,stage='ALL_EXISTING_SOURCES_CPU_NORMALIZATION_NOT_READY',source=source['id'],file=Path(file['path']).name,next_pos=next_pos,completed_files=d.execute("SELECT COUNT(*) FROM files WHERE state='complete'").fetchone()[0],registered_files=registry['files'],training_started=False))
                    next_pos=rec['next_pos'];next_byte=rec['next_byte']
                    for pos,byte,end,row in rows(file,next_pos,next_byte):
                        r=None
                        try:
                            if row.get('_decode_error'):raise ValueError('invalid_jsonl')
                            r=adapter(row,source,file,pos,byte);gate,reason=rule(r)
                            if gate=='CANDIDATE':
                                ids=None
                                if r['own_text_tokens'] is None:
                                    ids=tok.encode(r['text'],add_special_tokens=False).ids;r['own_text_tokens']=len(ids)
                                if not r['own_text_tokens']:gate='DROP';reason='empty_encoded_body'
                                if ids and any(x<15 for x in ids):gate='HOLD';reason='reserved_token_in_body'
                                if source.get('normalized') and any(x in r['text'] for x in cfg['reserved_tokens']):gate='HOLD';reason='reserved_token_literal'
                        except (ValueError,TypeError,KeyError) as exc:gate='HOLD';reason='adapter_'+str(exc)[:90]
                        pending.append((r,gate,reason,pos,byte));next_pos=pos+1;next_byte=end;observed+=1
                        if len(pending)>=256:commit(next_pos,next_byte)
                        if limit and observed>=limit:
                            commit(next_pos,next_byte);return dict(stage='controlled_resume_stop',rows=observed)
                    if pending:commit(next_pos,next_byte)
                    if file_sha(Path(file['path']))!=file['sha256']:raise ValueError('Source changed during scan')
                    with d:d.execute("UPDATE files SET state='complete' WHERE id=?",(fid,))
                except Exception as exc:
                    with d:d.execute("UPDATE files SET state='hold',stats=? WHERE id=?",(json.dumps(dict(stats,terminal_reason=type(exc).__name__+':'+str(exc))),fid));d.execute("UPDATE records SET gate='HOLD_FILE' WHERE fid=?",(fid,))
                    print('SOURCE_FILE_HOLD',file['path'],type(exc).__name__,str(exc),flush=True)
        counts=[dict(domain=x[0],language=x[1],split=x[2],gate=x[3],documents=x[4],body_tokens=x[5]) for x in d.execute('SELECT domain,lang,split,gate,COUNT(*),SUM(n) FROM records GROUP BY domain,lang,split,gate')]
        state=dict(version=11,stage='all_registered_files_processed_with_explicit_holds_not_quality_approval',registry_sha256=file_sha(root/'REGISTRY.json'),scan_identity=binding,files=registry['files'],held_files=d.execute("SELECT COUNT(*) FROM files WHERE state='hold'").fetchone()[0],counts=counts,training_started=False)
        write(root/'SCAN_COMPLETE.json',state);atomic_status(root/'SCAN_STATUS.json',state);return state
    finally:d.close()

def sha_rank(h):return digest(['candidate-order-v11',h])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--limit',type=int);a=p.parse_args();cfg=json.loads(a.config.read_text());root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
 with (root/'scan.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);print(json.dumps(scan(cfg,a.limit),ensure_ascii=False))
