"""Allocate ONLY frozen filtered IDs; the 11.7B original stock is not approved."""
import json,sqlite3,sys
from pathlib import Path
from p0_stock_plan_v6 import open_plan,materialize
from small_agent.curation.release_gates_v5 import bound_json,write
from small_agent.curation.supply_control import digest,file_sha

def allocate_clean(root,original,filtered,scope,target=30000000):
    if scope['status']!='passed':raise ValueError('Filtered scope not approved')
    receipt=bound_json(filtered/'FILTER_DATABASE_BOUND.json');key=digest(dict(scope=scope,filter_database_sha256=receipt['database_sha256']))
    root.mkdir(parents=True,exist_ok=True)
    # Canonical SQLite path only: symlink aliases can split WAL sidecars.
    write(root/'BOOK_POINTER.json',dict(database=str((original/'stock.sqlite').resolve()),stock_registration=str((original/'STOCK_REGISTERED.json').resolve()),stock_registration_sha256=file_sha(original/'STOCK_REGISTERED.json')))
    write(root/'STOCK_REGISTERED.json',json.loads((original/'STOCK_REGISTERED.json').read_text()))
    db=open_plan(original.resolve());db.execute('ATTACH DATABASE ? AS eligible',('file:'+str((filtered/'eligible.sqlite').resolve())+'?mode=ro',))
    try:
        for sid, in db.execute('SELECT DISTINCT sid FROM eligible.eligible ORDER BY sid').fetchall():
            done='math_clean_v8_allocated_source:'+str(sid);old=db.execute('SELECT v FROM meta WHERE k=?',(done,)).fetchone()
            if old:
                if old[0]!=key:raise ValueError('Frozen clean source scope differs')
                continue
            approved=set(scope['approved_strata']);bad=set(scope['known_audit_bad_body_sha256']);chunks=[];chunk=[];tokens=0
            for h,n,label in db.execute("SELECT r.h,r.n,e.stratum FROM records r JOIN eligible.eligible e ON e.h=r.h AND e.sid=r.sid AND e.pos=r.pos AND e.n=r.n AND e.split=r.split WHERE r.sid=? AND r.split='train' AND r.job IS NULL ORDER BY r.rank",(sid,)):
                if h in bad or label not in approved:continue
                chunk.append(h);tokens+=n
                if tokens>=target:chunks.append(chunk);chunk=[];tokens=0
            if chunk:
                if chunks and tokens<target/2:chunks[-1].extend(chunk)
                else:chunks.append(chunk)
            if not chunks:continue
            held={s:[h for h,label in db.execute('SELECT r.h,e.stratum FROM records r JOIN eligible.eligible e ON e.h=r.h AND e.sid=r.sid AND e.pos=r.pos AND e.n=r.n AND e.split=r.split WHERE r.sid=? AND r.split=? ORDER BY r.rank',(sid,s)) if label in approved and h not in bad] for s in ['validation','test']}
            if any(len(v)<len(chunks) for v in held.values()):continue # Never manufacture heldout from train.
            for records in held.values():
                for i,h in enumerate(records):chunks[i%len(chunks)].append(h)
            with db:
                for i,chunk in enumerate(chunks):
                    name='math_clean_'+digest(dict(sid=sid,i=i,scope=key,records=chunk))[:20]+'_v8';cur=db.execute('INSERT INTO jobs(sid,name,scope_sha) VALUES (?,?,?)',(sid,name,key));jid=cur.lastrowid
                    db.executemany("UPDATE records SET job=?,gate='v8_clean_planned_not_ready' WHERE h=?",[(jid,h) for h in chunk])
                db.execute('INSERT INTO meta VALUES (?,?)',(done,key))
        return db.execute('SELECT COUNT(*) FROM jobs WHERE name LIKE ? ',('math_clean_%_v8',)).fetchone()[0]
    finally:db.close()


def open_shared_plan(root):
    pointer=json.loads((root/'BOOK_POINTER.json').read_text());p=Path(pointer['database']).resolve()
    if p.name!='stock.sqlite' or file_sha(Path(pointer['stock_registration']))!=pointer['stock_registration_sha256']:raise ValueError('Shared stock metadata pointer differs')
    return open_plan(p.parent)
