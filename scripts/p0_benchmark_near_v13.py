#!/usr/bin/env python3
"""Read-only algorithm comparison on real code in an isolated shadow inventory."""
import argparse,json,sys,time,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.stock_v6 import DiskNearIndex
from small_agent.curation.release_gates_v5 import sha,write
from small_agent.curation.supply_control import file_sha
from p0_parallel_near_v13 import ParallelNearIndex


def main():
    import pyarrow.parquet as pq
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--sketch-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--documents',type=int,default=2048);a=p.parse_args()
    if a.output.exists():raise ValueError('Never overwrite a completed benchmark')
    a.output.mkdir(parents=True)
    rows=pq.read_table(a.source).to_pylist();database=a.output/'shadow.sqlite'
    with sqlite3.connect(database) as db:
        db.executescript('CREATE TABLE documents(body_sha TEXT PRIMARY KEY,split TEXT,domain TEXT,tokens INTEGER);CREATE TABLE releases(batch_id TEXT);')
        db.executemany('INSERT INTO documents VALUES (?,?,?,?)',[(r['normalized_sha256'],r['split'],r['domain'],r['own_text_tokens']) for r in rows])
    write(a.output/'legacy.json',dict(evidence_files=[dict(path=str(a.source),sha256=file_sha(a.source))]))
    idx=DiskNearIndex(a.output/'index')
    try:manifest=idx.update(database,a.output/'legacy.json',a.sketch_root)
    finally:idx.close()
    candidates=[]
    for i,r in enumerate(rows[:a.documents]):
        if i%3==0:text=r['text']
        elif i%3==1:text=r['text']+'\n// additional independent check\n'
        else:text=('completely_new_%d vocabulary_%d mathematics explanation independent_record_%d\n'%(i,i,i))*50
        candidates.append(dict(r,text=text,normalized_sha256=sha(text)))
    # Force local duplicates across chunk boundaries, not just inside each worker's chunk.
    for r in candidates[2::100][:20]:
        candidates.append(dict(r,text=r['text']+'\n',normalized_sha256=sha(r['text']+'\n')))
    policy=dict(threshold=.9,sketch_size=64,minimum_shared_fraction=.5,maximum_pair_checks_per_document=2000)
    results=[];baseline=None
    for workers in (1,8):
        idx=DiskNearIndex(a.output/'index') if workers==1 else ParallelNearIndex(a.output/'index')
        t=time.monotonic()
        try:out=idx.filter(candidates,manifest,policy,database) if workers==1 else idx.filter(candidates,manifest,policy,database,workers)
        finally:idx.close()
        seconds=time.monotonic()-t
        if baseline is None:baseline=out
        if out!=baseline:raise ValueError('Parallel exclusion decision/report differs')
        measured=dict(workers=workers,seconds=seconds,documents=len(candidates),retained=len(out[0]),exclusions=out[1]['excluded_documents'],exact_jaccard_pairs=out[1]['exact_jaccard_pairs'],decisions_and_reports_identical=True)
        results.append(measured);print(json.dumps(measured),flush=True)
    results[1]['speedup']=results[0]['seconds']/results[1]['seconds']
    write(a.output/'RESULT.json',dict(version=13,stage='MEASURED_NEAR_EQUIVALENCE_NOT_CORPUS_APPROVAL',results=results,production_ledger_written=False,api_requests=0))


if __name__=='__main__':main()
