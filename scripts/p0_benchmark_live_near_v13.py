#!/usr/bin/env python3
"""READ ONLY full-inventory comparison. Abort on changed snapshot; no production writer lock."""
import argparse,json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p0_parallel_near_v13 import ParallelNearIndex,ReadOnlyIndex
from small_agent.curation.stock_v6 import DiskNearIndex
from small_agent.curation.release_gates_v5 import write
from small_agent.curation.supply_control import file_sha

def main():
    import pyarrow.parquet as pq
    p=argparse.ArgumentParser()
    for k in ('candidates','inventory-index','database','policy','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--documents',type=int,default=256);a=p.parse_args()
    if a.output.exists():raise ValueError('Benchmark receipt already exists')
    rows=[]
    for b in pq.ParquetFile(a.candidates).iter_batches(batch_size=64):
        rows+=b.to_pylist()
        if len(rows)>=a.documents:break
    rows=rows[:a.documents];ref=json.loads(a.inventory_index.read_text());policy=json.loads(a.policy.read_text())['near']
    result=[];baseline=None
    for workers in (8,1,1,8):  # counterbalance warm filesystem cache
        idx=ParallelNearIndex(Path(ref['database']).parent)
        t=time.monotonic()
        try:
            value=(DiskNearIndex.filter(idx,rows,a.inventory_index,policy,a.database) if workers==1
                   else idx.filter(rows,a.inventory_index,policy,a.database,workers))
        finally:idx.close()
        elapsed=time.monotonic()-t
        if baseline is None:baseline=value
        if baseline!=value:raise ValueError('Live snapshot or exclusion decisions changed; cannot accept benchmark')
        measured=dict(workers=workers,seconds=elapsed,retained_documents=len(value[0]),excluded_documents=value[1]['excluded_documents'],exact_jaccard_pairs=value[1]['exact_jaccard_pairs'],identical_full_report=True)
        result.append(measured);print(json.dumps(measured),flush=True)
    medians={w:statistics.median(r['seconds'] for r in result if r['workers']==w) for w in (1,8)}
    write(a.output,dict(version=13,stage='READ_ONLY_FULL_INVENTORY_NEAR_AB',results=result,documents=len(rows),
        median_seconds=medians,speedup=medians[1]/medians[8],inventory_snapshot=ref['ledger'],
        candidate_sha256=file_sha(a.candidates),production_ledger_written=False,api_requests=0))

if __name__=='__main__':main()
