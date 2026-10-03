#!/usr/bin/env python3
"""Reconcile shared READY bodies into stock metadata; never count twice or grant a source license."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p0_stock_plan_v6 import open_plan
from p0_curation_worker_v3 import atomic_status
from small_agent.curation.release_gates_v5 import ledger_snapshot

def reconcile(stock,database):
    if not (stock/'STOCK_REGISTERED.json').exists():raise ValueError('Stock scan must finish before reconciliation')
    db=open_plan(stock);db.execute('ATTACH DATABASE ? AS supply',('file:'+str(database.resolve())+'?mode=ro',))
    try:
        db.execute('BEGIN')
        db.execute("UPDATE records SET gate='accounted_READY_same_body_not_source_approval',job=COALESCE(job,-1) WHERE h IN (SELECT d.body_sha FROM supply.documents d JOIN supply.releases r ON r.batch_id=d.batch_id)")
        ready=db.execute("SELECT split,COUNT(*),SUM(n) FROM records WHERE gate='accounted_READY_same_body_not_source_approval' GROUP BY split").fetchall()
        gates=[dict(domain=d,split=s,gate=g,documents=n,body_tokens=t) for d,s,g,n,t in db.execute('SELECT domain,split,gate,COUNT(*),SUM(n) FROM records GROUP BY domain,split,gate')]
        db.commit()
    finally:db.close()
    result=dict(version=6,stage='stock_shared_ledger_reconciled_without_double_counting',existing_ready_overlap=[dict(split=s,documents=n,body_tokens=t) for s,n,t in ready],domain_split_gates=gates,ledger=ledger_snapshot(database),new_ready_tokens=0,training_started=False)
    atomic_status(stock/'STOCK_GATE_STATUS.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('stock','database'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();print(json.dumps(reconcile(a.stock,a.database)))
if __name__=='__main__':main()
