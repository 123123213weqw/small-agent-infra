#!/usr/bin/env python3
"""Four-trial near A/B on complete immutable full-inventory clone, no production lock."""
import json,statistics,sys,time,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import p0_inventory_supply_v17 as q
from p0_parallel_near_v13 import ParallelNearIndex as Old
from p0_parallel_near_v20 import ParallelNearIndex as New
from small_agent.curation.release_gates_v5 import write,ledger_snapshot
from small_agent.curation.supply_control import digest,file_sha

def main():
 cfg=json.loads((ROOT/'configs/curation/p0_full_inventory_supply_v11_r2.json').read_text());root=Path(cfg['root']);bench=root/'parallel_v13';idxroot=bench/'index_ab_v18/v18';ledger=bench/'index_ab_v18/ledger.sqlite'
 while not (idxroot/'CURRENT').exists():time.sleep(3)
 manifest=Path((idxroot/'CURRENT').read_text().strip());ref=json.loads(manifest.read_text())
 if ref['ledger']!=ledger_snapshot(ledger):raise ValueError('Clone snapshot differs')
 out=bench/'near_clone_ab_v20';out.mkdir(exist_ok=True)
 if (out/'RESULT.json').exists():print((out/'RESULT.json').read_text());return
 c=sqlite3.connect('file:'+str(root/'stock.sqlite')+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
 query=q.source_query(c,"r.gate='CANDIDATE_FALLBACK' AND r.job IS NULL AND r.split='train'")+' ORDER BY r.rank LIMIT 256'
 records=c.execute(query,('github_code_clean_supply_v4','code','javascript')).fetchall()
 rows=q.retrieve(cfg,c,records,json.loads((root/'REGISTRY.json').read_text()),q.ArrowRawReader());c.close()
 import pyarrow as pa,pyarrow.parquet as pq
 pq.write_table(pa.Table.from_pylist(rows),out/'candidates.parquet',compression='zstd');policy=json.loads(Path(cfg['release_policy']).read_text())['near'];trials=[];expected=None
 for name,cls in [('v20',New),('v13',Old),('v13',Old),('v20',New)]:
  idx=cls(idxroot);start=time.monotonic()
  try:value=idx.filter(rows,manifest,policy,ledger,16)
  finally:idx.close()
  if expected is None:expected=value
  if value!=expected:raise ValueError('Complete decisions/report differ')
  item=dict(engine=name,seconds=time.monotonic()-start,documents=len(rows),kept=len(value[0]),pairs=value[1]['exact_jaccard_pairs'],result_sha256=digest(value));trials.append(item);print(json.dumps(item),flush=True)
 med={x:statistics.median(t['seconds'] for t in trials if t['engine']==x) for x in ('v13','v20')}
 write(out/'RESULT.json',dict(trials=trials,median_seconds=med,stage_speedup=med['v13']/med['v20'],same_complete_report=True,inventory_snapshot=ref['ledger'],candidate_sha256=file_sha(out/'candidates.parquet'),production_modified=False,api_requests=0))
if __name__=='__main__':main()
