"""Persist exact computed near sketches in unchanged v5 cache schema; no admission decision."""
from pathlib import Path
import os
from small_agent.curation.supply_control import digest,file_sha
from small_agent.curation.release_gates_v5 import write

def store_checked_sketch(entry,rows,root):
 import pyarrow as pa,pyarrow.parquet as pq
 root=Path(root);root.mkdir(parents=True,exist_ok=True);key=digest(entry)
 source=Path(entry['path']);dest=root/(key+'.sketch.parquet');marker=root/(key+'.done.json')
 if marker.exists():
  import json
  m=json.loads(marker.read_text())
  if m['input']!=entry or m['sha256']!=file_sha(dest):raise ValueError('Exact sketch cache changed')
  return m
 if file_sha(source)!=entry['sha256']:raise ValueError('Source changed before sketch reuse')
 # Bind every cached record to the final exact source row/position/split.
 source_rows=pq.read_table(source,columns=['normalized_sha256','split'],use_threads=False).to_pylist()
 if len(rows)!=len(source_rows):raise ValueError('Sketch coverage differs')
 for i,(r,s) in enumerate(zip(rows,source_rows)):
  if r['row']!=i or r['body_sha256']!=s['normalized_sha256'] or r['split']!=s['split'] or len(r['sketch'])!=min(64,r['shingle_count']) or r['sketch']!=sorted(set(r['sketch'])):raise ValueError('Sketch row binding differs')
 schema=pa.schema([('body_sha256',pa.string()),('split',pa.string()),('row',pa.int64()),('shingle_count',pa.int32()),('sketch',pa.list_(pa.uint64()))])
 tmp=dest.with_suffix('.v18.tmp');pq.write_table(pa.Table.from_pylist(rows,schema=schema),tmp,compression='zstd')
 if file_sha(source)!=entry['sha256']:raise ValueError('Source changed during sketch reuse')
 with tmp.open('rb') as f:os.fsync(f.fileno())
 os.replace(tmp,dest);m=dict(input=entry,path=str(dest),sha256=file_sha(dest),documents=len(rows));write(marker,m);return m
