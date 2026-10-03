#!/usr/bin/env python3
import argparse,fcntl,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from small_agent.curation.stock_v6 import DiskNearIndex
def main():
 p=argparse.ArgumentParser()
 for n in ('database','legacy-receipt','sketch-root','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args()
 with (a.database.parent/'production-release.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  idx=DiskNearIndex(a.output)
  try:path=idx.update(a.database,a.legacy_receipt,a.sketch_root)
  finally:idx.close()
 print(json.dumps(dict(stage='complete_registered_disk_index',manifest=str(path),training_started=False)))
if __name__=='__main__':main()
