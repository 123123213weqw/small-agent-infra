#!/usr/bin/env python3
"""Read-only counterbalanced comparison, all v11 fields and adapted records equal."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.full_inventory_v11 import RawReader,adapter
from small_agent.curation.supply_control import digest,file_sha
from small_agent.curation.release_gates_v5 import write
from p0_arrow_reader_v15 import ArrowRawReader


def main():
    import pyarrow.parquet as pq
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--registry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Immutable benchmark receipt exists')
    registry=json.loads(a.registry.read_text());rows=pq.read_table(a.candidate,columns=['origin_file','origin_row','origin_byte_offset']).to_pylist()
    path=Counter(r['origin_file'] for r in rows).most_common(1)[0][0]
    positions=sorted([r for r in rows if r['origin_file']==path],key=lambda x:x['origin_row'])[:128]
    source=next(s for s in registry['sources'] if any(f['path']==path for f in s['files']));file=next(f for f in source['files'] if f['path']==path)
    if file_sha(Path(path))!=file['sha256']:raise ValueError('Original changed')
    results=[];baseline=None
    for version,cls in ((11,RawReader),(15,ArrowRawReader),(15,ArrowRawReader),(11,RawReader)):
        reader=cls();start=time.monotonic();raw=[];adapted=[]
        for r in positions:
            row=reader.row(file,r['origin_row'],r['origin_byte_offset']);raw.append(row)
            adapted.append(adapter(row,source,file,r['origin_row'],r['origin_byte_offset']))
        seconds=time.monotonic()-start;fingerprint=dict(raw=digest(raw),adapted=digest(adapted))
        if baseline is None:baseline=fingerprint
        if fingerprint!=baseline:raise ValueError('Reader or adapter changed any field')
        value=dict(version=version,seconds=seconds,rows=len(raw),all_fields_and_adapted_records_identical=True);results.append(value);print(json.dumps(value),flush=True)
        del reader,raw,adapted
    if file_sha(Path(path))!=file['sha256']:raise ValueError('Original changed during benchmark')
    medians={v:statistics.median(x['seconds'] for x in results if x['version']==v) for v in (11,15)}
    write(a.output,dict(stage='RAW_READER_AB_PASSED',results=results,median_seconds=medians,speedup=medians[11]/medians[15],fingerprint=baseline,
        source_file=path,source_sha256=file['sha256'],candidate_sha256=file_sha(a.candidate),registry_sha256=file_sha(a.registry),api_requests=0,production_ledger_written=False))

if __name__=='__main__':main()
