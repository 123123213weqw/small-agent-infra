#!/usr/bin/env python3
"""CPU-only all-accounted-inventory index. No API, GPU, release or training."""
import argparse,fcntl,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from small_agent.curation.release_gates_v5 import build_inventory_index,write
p=argparse.ArgumentParser(description=__doc__)
for n in ('database','legacy-receipt','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
with (a.output/'index.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 path=build_inventory_index(a.database,a.legacy_receipt,a.output)
 # Pointer is mutable, generations and measured manifests are immutable.
 tmp=a.output/'.CURRENT.tmp';tmp.write_text(str(path.resolve())+'\n');tmp.replace(a.output/'CURRENT')
 print(json.dumps(dict(stage='complete_inventory_index',manifest=str(path),training_started=False)))
