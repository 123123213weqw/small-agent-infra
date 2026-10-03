#!/usr/bin/env python3
"""Register the entire frozen 117B candidate-train inventory, without body copies.

Jobs are metadata pointers until admission; missing source/reference/licenses
remain explicit HOLD. These records never advance the READY token ledger.
"""
import argparse,fcntl,json,sqlite3,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.supply_control import file_sha,digest
from small_agent.curation.release_gates_v5 import sha,write
from p0_count_supply_candidates_v4 import LICENSES

def open_plan(root):
    root.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(root/'stock.sqlite',timeout=60)
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA cache_size=-131072')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS sources(id INTEGER PRIMARY KEY,path TEXT UNIQUE,sha TEXT,source_id TEXT,revision TEXT,complete INTEGER DEFAULT 0);
      CREATE TABLE IF NOT EXISTS records(h TEXT PRIMARY KEY,sid INTEGER,pos INTEGER,split TEXT,language TEXT,domain TEXT,n INTEGER,family TEXT,license TEXT,bucket TEXT,rank BLOB,gate TEXT,job INTEGER);
      CREATE INDEX IF NOT EXISTS by_source ON records(sid,split,rank);
      CREATE INDEX IF NOT EXISTS by_job ON records(job);
      CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY,sid INTEGER,name TEXT UNIQUE,scope_sha TEXT,state TEXT DEFAULT 'planned',reason TEXT);
      CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT);
    ''');db.commit();return db

def scan(root,census,plan):
    import pyarrow.parquet as pq
    db=open_plan(root);status=json.loads((census/'status.json').read_text())
    if status['stage']!='candidate_token_census_complete_not_ready':raise ValueError('Full census not complete')
    config=json.loads(plan.read_text());planned={x['path']:x for x in config['items']}
    c=sqlite3.connect('file:'+str((census/'census.sqlite').resolve())+'?mode=ro',uri=True)
    try:files=[json.loads(r[0]) for r in c.execute('SELECT receipt FROM shards ORDER BY key')]
    finally:c.close()
    binding=dict(version=6,plan_sha256=file_sha(plan),census_identity=status['identity_sha256'],scanner_sha256=file_sha(Path(__file__)),seed=20261002)
    old=db.execute("SELECT v FROM meta WHERE k='binding'").fetchone()
    if old and json.loads(old[0])!=binding:raise ValueError('Stock scan identity differs')
    with db:db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)',('binding',json.dumps(binding,sort_keys=True)))
    cols=['normalized_sha256','source_revision','split','language','domain','own_text_tokens','family_id','record_license','policy_bucket']
    for f in files:
        p=census/f['output_relative_path'];item=planned[f['source_file']]
        if item['sha256']!=f['source_sha256'] or file_sha(p)!=f['output_sha256']:raise ValueError('Frozen source changed')
        old=db.execute('SELECT id,sha,complete FROM sources WHERE path=?',(str(p.resolve()),)).fetchone()
        if old and old[1]!=f['output_sha256']:raise ValueError('Stock source changed')
        if old and old[2]:continue
        if not old:
            with db:db.execute('INSERT INTO sources(path,sha,source_id,revision) VALUES (?,?,?,?)',(str(p.resolve()),f['output_sha256'],f['source_id'],item['revision']))
        sid=db.execute('SELECT id FROM sources WHERE path=?',(str(p.resolve()),)).fetchone()[0];pos=0
        for b in pq.ParquetFile(p).iter_batches(batch_size=8192,columns=cols):
            rows=[]
            for r in b.to_pylist():
                if r['source_revision']!=item['revision']:raise ValueError('Source revision differs')
                gate='source_calibration_pending'
                if r['split'] not in ('train','validation','test'):gate='unassigned_split_hold'
                elif r['domain']=='code' and (r['policy_bucket']!='record_license_whitelisted' or r['record_license'].lower() not in LICENSES):gate='record_license_hold'
                elif r['domain']=='code' and r['language'].lower()!='python':gate='independent_language_reference_pending'
                rows.append((r['normalized_sha256'],sid,pos,r['split'],r['language'].lower(),r['domain'],r['own_text_tokens'],r['family_id'],r['record_license'],r['policy_bucket'],bytes.fromhex(sha('20261002|'+r['normalized_sha256'])),gate));pos+=1
            with db:db.executemany('INSERT OR IGNORE INTO records(h,sid,pos,split,language,domain,n,family,license,bucket,rank,gate) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',rows)
        if file_sha(p)!=f['output_sha256']:raise ValueError('Source changed during metadata scan')
        with db:db.execute('UPDATE sources SET complete=1 WHERE id=?',(sid,))
        print('STOCK_REGISTERED_SHARD',f['source_id'],f['source_file'],pos,flush=True)
    count=db.execute('SELECT COUNT(*) FROM records').fetchone()[0]
    if count!=status['counts']['candidate_documents']:raise ValueError('Stock count differs from whole candidate census')
    report=dict(version=6,stage='whole_candidate_stock_registered_not_ready',sources=len(files),documents=count,
                candidate_train_body_tokens=db.execute("SELECT SUM(n) FROM records WHERE split='train'").fetchone()[0],
                domain_split_gates=[dict(domain=d,split=s,gate=g,documents=n,body_tokens=t) for d,s,g,n,t in db.execute('SELECT domain,split,gate,COUNT(*),SUM(n) FROM records GROUP BY domain,split,gate')],
                binding=binding,new_ready_tokens=0,training_started=False)
    write(root/'STOCK_REGISTERED.json',report);db.close();return report

def allocate(root,scope,target=30000000):
    if scope['status']!='passed':return 0
    db=open_plan(root);key=digest(scope)
    existing=db.execute("SELECT v FROM meta WHERE k='math_scope'").fetchone()
    if existing:
        if existing[0]!=key:raise ValueError('Frozen allocation scope changed; do not silently resample')
        db.close();return 0
    if not (root/'STOCK_REGISTERED.json').exists():raise ValueError('All stock metadata must be registered first')
    sizes={h.split('/')[0] for h in scope['approved_strata']};bad=set(scope['known_audit_bad_body_sha256']);created=0
    def allowed(row):
        h,n=row[0],row[1];size='short' if n<=512 else 'medium' if n<=4096 else 'long';return size in sizes and h not in bad
    for sid, in db.execute("SELECT id FROM sources WHERE source_id='finemath_4plus_supply_v4' ORDER BY path").fetchall():
        source_key='allocated_source:'+str(sid)
        done=db.execute('SELECT v FROM meta WHERE k=?',(source_key,)).fetchone()
        if done:
            if done[0]!=key:raise ValueError('Source allocation identity changed')
            continue
        chunks=[];chunk=[];tokens=0
        for h,n in db.execute("SELECT h,n FROM records WHERE sid=? AND split='train' AND language='en' AND job IS NULL ORDER BY rank",(sid,)):
            if not allowed((h,n)):continue
            chunk.append(h);tokens+=n
            if tokens>=target:chunks.append(chunk);chunk=[];tokens=0
        if chunk:
            if chunks and tokens<target/2:chunks[-1].extend(chunk)
            else:chunks.append(chunk)
        if not chunks:continue
        held={s:[h for h,n in db.execute('SELECT h,n FROM records WHERE sid=? AND split=? AND language=\'en\' ORDER BY rank',(sid,s)) if allowed((h,n))] for s in ('validation','test')}
        if any(len(v)<len(chunks) for v in held.values()):
            with db:db.execute("UPDATE records SET gate='insufficient_heldout_in_source_hold' WHERE sid=? AND domain='math'",(sid,))
            continue
        for records in held.values():
            for i,h in enumerate(records):chunks[i%len(chunks)].append(h)
        with db:
            for i,chunk in enumerate(chunks):
                name='math_stock_'+digest(dict(source=sid,chunk=i,scope=key,records=chunk))[:20]+'_v6'
                cur=db.execute('INSERT INTO jobs(sid,name,scope_sha) VALUES (?,?,?)',(sid,name,key));jid=cur.lastrowid
                db.executemany("UPDATE records SET job=?,gate='planned_source_scoped_not_ready' WHERE h=?",[(jid,h) for h in chunk]);created+=1
            db.execute('INSERT INTO meta VALUES (?,?)',(source_key,key))
        print('ALLOCATED_SOURCE_BATCHES',sid,len(chunks),flush=True)
    with db:db.execute('INSERT INTO meta VALUES (?,?)',('math_scope',key))
    db.close();return created

def materialize(root,jid,output):
    import pyarrow as pa,pyarrow.parquet as pq
    db=open_plan(root)
    try:
        sid,name=db.execute('SELECT sid,name FROM jobs WHERE id=?',(jid,)).fetchone()
        path,h=db.execute('SELECT path,sha FROM sources WHERE id=?',(sid,)).fetchone();path=Path(path)
        if file_sha(path)!=h:raise ValueError('Frozen stock source changed')
        wanted={pos:sha256 for pos,sha256 in db.execute('SELECT pos,h FROM records WHERE job=?',(jid,))}
        rows=[];pos=0
        for b in pq.ParquetFile(path).iter_batches(batch_size=256):
            for r in b.to_pylist():
                if pos in wanted:
                    if r['normalized_sha256']!=wanted[pos] or sha(r['text'])!=wanted[pos]:raise ValueError('Stock row pointer changed')
                    rows.append(r)
                pos+=1
        if len(rows)!=len(wanted):raise ValueError('Stock pointers missing')
        rows.sort(key=lambda r:sha('20261002|'+r['normalized_sha256']))
        output.parent.mkdir(parents=True,exist_ok=True)
        tmp=output.with_suffix('.tmp');pq.write_table(pa.Table.from_pylist(rows),tmp,compression='zstd')
        if output.exists():
            if file_sha(tmp)!=file_sha(output):tmp.unlink();raise ValueError('Frozen materialized job changed')
            tmp.unlink()
        else:tmp.replace(output)
        write(output.parent/'CANDIDATE_BINDING.json',dict(job_id=jid,name=name,source_path=str(path),source_sha256=h,candidate_sha256=file_sha(output),documents=len(wanted),pointer_identity=digest(sorted(wanted.items()))))
        return output
    finally:db.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('output','census-root','download-plan'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'scanner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(scan(a.output,a.census_root,a.download_plan),ensure_ascii=False))
if __name__=='__main__':main()
