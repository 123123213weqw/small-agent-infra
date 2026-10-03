#!/usr/bin/env python3
"""Transactional secondary indexes only. Pause the owned importer outside its write transaction."""
import argparse
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from small_agent.curation.release_gates_v5 import write

INDEXES={
    'records_calibration_v15':"CREATE INDEX records_calibration_v15 ON records(domain,lang,split,stratum,rank,fid) WHERE initial_gate='CANDIDATE'",
    'records_selection_v15':"CREATE INDEX records_selection_v15 ON records(domain,lang,gate,split,rank,fid)",
    'records_job_v15':"CREATE INDEX records_job_v15 ON records(job) WHERE job IS NOT NULL",
    'records_source_v15':"CREATE INDEX records_source_v15 ON records(fid,gate,domain,lang)",
}


def resume(pid):
    try:cmd=Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:return
    if any(Path(x.decode()).name=='p0_distributed_scan_v14.py' for x in cmd if x):os.kill(pid,signal.SIGCONT)


def build(database,output,scanner_pid=None):
    db=sqlite3.connect('file:'+str(database.resolve())+'?mode=rw',uri=True,timeout=60)
    db.execute('PRAGMA temp_store=MEMORY');db.execute('PRAGMA cache_size=-262144')
    stopped=False;started=time.monotonic();timings={}
    try:
        db.execute('BEGIN IMMEDIATE')  # importer cannot be stopped inside a writer transaction
        if scanner_pid:
            cmd=Path('/proc/'+str(scanner_pid)+'/cmdline').read_bytes().split(b'\0')
            if not any(Path(x.decode()).name=='p0_distributed_scan_v14.py' for x in cmd if x):raise ValueError('Not the owned v14 importer')
            os.kill(scanner_pid,signal.SIGSTOP);stopped=True
        initial_changes=db.total_changes
        for name,sql in INDEXES.items():
            existing=db.execute('SELECT sql FROM sqlite_master WHERE name=?',(name,)).fetchone()
            if existing:
                if existing[0]!=sql:raise ValueError('Existing index definition differs')
                timings[name]=0;continue
            t=time.monotonic();db.execute(sql);timings[name]=time.monotonic()-t
            print(json.dumps(dict(index=name,seconds=timings[name])),flush=True)
        if db.total_changes!=initial_changes:raise ValueError('Index builder modified corpus rows')
        db.commit()
        plans={}
        for source,domain,lang in (('github_code_clean_new','code','cpp'),('github_code_clean_supply_v4','code','java'),('finemath_4plus','math','en')):
            query="SELECT stratum,COUNT(*) FROM records r JOIN files f ON f.id=r.fid WHERE f.source=? AND r.domain=? AND r.lang=? AND r.initial_gate='CANDIDATE' AND r.split='train' GROUP BY stratum"
            plans[source]=db.execute('EXPLAIN QUERY PLAN '+query,(source,domain,lang)).fetchall()
        write(output,dict(stage='SECONDARY_QUEUE_INDEXES_READY',database=str(database),definitions=INDEXES,timings_seconds=timings,
            elapsed_seconds=time.monotonic()-started,corpus_rows_modified=db.total_changes-initial_changes,query_plans=plans,
            quality_thresholds_changed=False,api_requests=0))
    except BaseException:
        db.rollback();raise
    finally:
        if stopped:resume(scanner_pid)
        db.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--database',type=Path);p.add_argument('--output',type=Path);p.add_argument('--pause-scanner-pid',type=int);p.add_argument('--resume-scanner-pid',type=int);a=p.parse_args()
    if a.resume_scanner_pid:resume(a.resume_scanner_pid);return
    if not a.database or not a.output:p.error('database and output required')
    def terminate(signum,frame):raise SystemExit('Index build terminated; resume importer')
    signal.signal(signal.SIGTERM,terminate)
    build(a.database,a.output,a.pause_scanner_pid)

if __name__=='__main__':main()
