#!/usr/bin/env python3
"""Real fresh fallback rows, serial/16 pre-billing AB. No API/ledger writes.

A bounded exclusive inventory lock keeps all four read-only comparisons on the
same full snapshot. Existing production handles this lock as backpressure.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import p0_inventory_supply_v17 as queue
from p0_parallel_near_v13 import ParallelNearIndex
from small_agent.curation.stock_v6 import DiskNearIndex
from small_agent.curation.release_gates_v5 import write,ledger_snapshot
from small_agent.curation.supply_control import file_sha,digest


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--documents',type=int,default=512)
    a=p.parse_args();cfg=json.loads(a.config.read_text());root=Path(cfg['root'])
    if not 64<=a.documents<=1024:raise ValueError('Bounded read-only sample required')
    receipt=a.output/'RESULT.json'
    if a.output.exists():raise ValueError('Never overwrite a benchmark')
    a.output.mkdir(parents=True)
    lock=(Path(cfg['database']).parent/'production-release.lock').open('a')
    try:
        for _ in range(36):
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:time.sleep(5);continue
            current=Path((Path(cfg['inventory_root'])/'CURRENT').read_text().strip())
            ref=json.loads(current.read_text())
            if ref['ledger']==ledger_snapshot(Path(cfg['database'])):break
            fcntl.flock(lock,fcntl.LOCK_UN);time.sleep(5)
        else:raise ValueError('No stable full index within bounded wait; no acceptance')
        store=sqlite3.connect('file:'+str(root/'stock.sqlite')+'?mode=ro',uri=True)
        store.row_factory=sqlite3.Row
        try:
            query=queue.source_query(store,"r.gate='CANDIDATE_FALLBACK' AND r.job IS NULL AND r.split='train'")+' ORDER BY r.rank LIMIT ?'
            records=store.execute(query,('github_code_clean_supply_v4','code','javascript',a.documents)).fetchall()
            if len(records)<64:raise ValueError('Insufficient fresh JS fallback population')
            rows=queue.retrieve(cfg,store,records,json.loads((root/'REGISTRY.json').read_text()),queue.ArrowRawReader())
        finally:store.close()
        candidate=a.output/'fresh_candidates.parquet'
        pq.write_table(pa.Table.from_pylist(rows),candidate,compression='zstd')
        policy=json.loads(Path(cfg['release_policy']).read_text())['near']
        result=[];baseline=None
        for workers in (16,1,1,16):
            idx=ParallelNearIndex(Path(ref['database']).parent);started=time.monotonic()
            try:
                value=DiskNearIndex.filter(idx,rows,current,policy,Path(cfg['database'])) if workers==1 else idx.filter(rows,current,policy,Path(cfg['database']),workers)
            finally:idx.close()
            elapsed=time.monotonic()-started
            if baseline is None:baseline=value
            if value!=baseline:raise ValueError('Complete keep/drop/report differs')
            item=dict(workers=workers,seconds=elapsed,retained=len(value[0]),excluded=value[1]['excluded_documents'],pairs=value[1]['exact_jaccard_pairs'],full_result_sha256=digest(value),identical_full_report=True)
            result.append(item);print(json.dumps(item),flush=True)
        medians={str(w):statistics.median(x['seconds'] for x in result if x['workers']==w) for w in (1,16)}
        write(receipt,dict(version=17,stage='LIVE_FULL_INVENTORY_FRESH_PRE_BILLING_AB',documents=len(rows),results=result,median_seconds=medians,speedup=medians['1']/medians['16'],inventory_snapshot=ref['ledger'],index_manifest_sha256=file_sha(current),candidate_sha256=file_sha(candidate),policy_sha256=file_sha(Path(cfg['release_policy'])),production_ledger_written=False,production_stock_rows_changed=False,api_requests=0,quality_thresholds_changed=False))
    finally:lock.close()


if __name__=='__main__':main()
