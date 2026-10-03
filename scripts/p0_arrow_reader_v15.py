"""The exact v11 raw fields, but convert only requested Arrow rows to Python."""
from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from small_agent.curation.full_inventory_v11 import FIELDS


class ArrowRawReader:
    def __init__(self):self.cache=OrderedDict()
    def row(self,file,pos,offset):
        path=Path(file['path'])
        if path.suffix=='.jsonl':
            with path.open('rb') as f:f.seek(offset);return json.loads(f.readline())
        import pyarrow.parquet as pq
        key=str(path)
        if key not in self.cache:
            pf=pq.ParquetFile(path);starts=[];total=0
            for i in range(pf.num_row_groups):starts.append(total);total+=pf.metadata.row_group(i).num_rows
            self.cache[key]=(pf,starts,None,None)
            if len(self.cache)>2:self.cache.popitem(last=False)
        pf,starts,g,rows=self.cache[key];want=bisect_right(starts,pos)-1
        if want<0:raise ValueError('Invalid original row pointer')
        if g!=want:
            rows=pf.read_row_group(want,columns=[k for k in FIELDS if k in pf.schema_arrow.names],use_threads=False);g=want
            self.cache[key]=(pf,starts,g,rows)
        n=pos-starts[want]
        return {name:rows.column(i)[n].as_py() for i,name in enumerate(rows.column_names)}
