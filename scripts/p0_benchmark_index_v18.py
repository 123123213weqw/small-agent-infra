#!/usr/bin/env python3
"""Real full-index clone A/B. Never writes production or calls API."""
import argparse,fcntl,json,sqlite3,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.stock_v6 import DiskNearIndex
from small_agent.curation.release_gates_v5 import ledger_snapshot,write
from small_agent.curation.supply_control import digest,file_sha
from p0_batched_index_v18 import BatchedDiskNearIndex

def backup(source,dest):
 with sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True) as a,sqlite3.connect(dest) as b:a.backup(b)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text());r=a.output;r.mkdir(parents=True,exist_ok=True)
 if (r/'RESULT.json').exists():print((r/'RESULT.json').read_text());return
 base=r/'base.sqlite';ledger=r/'ledger.sqlite'
 if not base.exists():
  with (Path(cfg['database']).parent/'production-release.lock').open('a') as lock:
   while True:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
    except BlockingIOError:time.sleep(2)
   backup(Path(cfg['inventory_root'])/'postings.sqlite',base);backup(Path(cfg['database']),ledger)
  # Reinsert the largest recently appended source, preserving all other global background.
  with sqlite3.connect(base) as db:
   sid,path,docs=db.execute('SELECT s.id,s.path,COUNT(*) n FROM sources s JOIN docs d ON d.source=s.id WHERE s.path LIKE ? GROUP BY s.id ORDER BY s.id DESC LIMIT 1',('%/batches/%',)).fetchone()
   db.execute('DELETE FROM postings WHERE d IN (SELECT id FROM docs WHERE source=?)',(sid,));db.execute('DELETE FROM docs WHERE source=?',(sid,));db.execute('UPDATE sources SET complete=0 WHERE id=?',(sid,));db.commit()
   write(r/'INPUT.json',dict(reinserted_source=path,reinserted_documents=docs,ledger=ledger_snapshot(ledger),production_modified=False,api_requests=0))
 results=[]
 for name,cls in [('v18',BatchedDiskNearIndex),('v6',DiskNearIndex)]:
  out=r/name;out.mkdir(exist_ok=True);backup(base,out/'postings.sqlite');idx=cls(out)
  try:
   start=time.monotonic();m=idx.update(ledger,Path(cfg['legacy_receipt']),Path(cfg['sketch_root']));elapsed=time.monotonic()-start
   sums={}
   for table,order in [('docs','id'),('sources','id'),('postings','h,d')]:
    h=hashlib.sha256();n=0
    for row in idx.db.execute('SELECT * FROM '+table+' ORDER BY '+order):
     h.update(repr(row).encode());h.update(b'\n');n+=1
    sums[table]=dict(rows=n,sha256=h.hexdigest())
   results.append(dict(engine=name,seconds=elapsed,logical_tables=sums,ledger=json.loads(m.read_text())['ledger']))
   print(json.dumps(results[-1]),flush=True)
  finally:idx.close()
 if results[0]['logical_tables']!=results[1]['logical_tables']:raise ValueError('Full logical index A/B differs')
 write(r/'RESULT.json',dict(trials=results,same_logical_index=True,stage_speedup=results[1]['seconds']/results[0]['seconds'],production_modified=False,api_requests=0))
if __name__=='__main__':main()
