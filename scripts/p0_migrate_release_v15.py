#!/usr/bin/env python3
"""Explicit, recoverable CPU-only migration. Never rewrite a frozen release directory."""
import argparse
import fcntl
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.supply_control import digest,file_sha,recipe_identity
from small_agent.curation.release_gates_v5 import bound_json,write
from small_agent.curation.full_inventory_v11 import AllStockControl
from small_agent.curation.quality_v11 import publication_quality


def terminal_review(folder):
    p=Path(folder)/'batch/worker/deepseek/ADAPTIVE_STATUS.json'
    s=json.loads(p.read_text())
    if s.get('stop_reason') or s.get('stage')!='all_selected_attempted_with_holds' or s['selected']!=s['considered']:
        raise ValueError('Review not terminal; migration must never resend paid requests')
    if s['counts'].get('unknown_or_fatal',0) or s['ledger'].get('cap_exceeded') is not False:
        raise ValueError('Unknown paid status or exhausted budget')
    campaign=Path(s['configuration']['campaign_ledger'])
    db=sqlite3.connect('file:'+str(campaign)+'?mode=ro',uri=True)
    try:
        states=dict(db.execute('SELECT state,COUNT(*) FROM requests GROUP BY state'))
        if states.get('inflight',0):raise ValueError('Campaign has inflight requests; do not stop')
        total=db.execute('SELECT SUM(counted) FROM requests').fetchone()[0]
    finally:db.close()
    return dict(status_sha256=file_sha(p),campaign=str(campaign),requests_by_state=states,counted_units=total)


def validate(intent):
    args=intent['arguments'];old=Path(args['output']);cfg=json.loads(Path(args['config']).read_text())
    batch=old.parent
    if old!=Path(cfg['root'])/'batches'/batch.name/'release' or batch.name!=intent['job']:
        raise ValueError('Migration output is not the exact registered batch directory')
    if Path(args['database']).resolve()!=Path(cfg['database']).resolve():raise ValueError('Wrong shared ledger')
    binding=json.loads((old/'INPUT_BINDING.json').read_text())
    if file_sha(old/'INPUT_BINDING.json')!=intent['old_binding_sha256']:raise ValueError('Old binding changed')
    mapping={'candidate_sha256':'candidates','candidate_binding_sha256':'candidate-binding','config_sha256':'config',
             'source_terms_sha256':'source-terms','policy_sha256':'policy','tokenizer_sha256':'tokenizer',
             'eval_registry_sha256':'eval-registry','inventory_manifest_sha256':'inventory-index'}
    for field,arg in mapping.items():
        if binding[field]!=file_sha(Path(args[arg])):raise ValueError('Changed frozen input: '+arg)
    for path,h in binding['engine_sha256'].items():
        if file_sha(Path(path))!=h:raise ValueError('Historical engine changed')
    for name in ('READY_INTERNAL.json','RELEASE_INTENT.json'):
        if (old/name).exists():raise ValueError('Old release reached publication; recover original engine instead')
    for p in old.glob('*.receipt.json'):bound_json(p)
    reviews={k:terminal_review(args[k]) for k in ('source-audit','reference','batch-audit')}
    if reviews!=intent['paid_reviews']:raise ValueError('Paid review/campaign changed since migration preflight')
    quality=publication_quality(cfg,Path(args['candidates']),Path(args['source-audit']),Path(args['reference']),Path(args['batch-audit']),args['mode'])
    if quality['status']!='passed' or digest(quality)!=intent['quality_digest']:
        raise ValueError('Quality gate differs from preflight')
    return binding,cfg


def prepare(pid,output):
    cmd=Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(b'\0')
    cmd=[x.decode() for x in cmd if x]
    if len(cmd)<3 or Path(cmd[1]).name!='p0_release_inventory_batch_v11.py':raise ValueError('Not the owned old CPU publisher')
    cli=cmd[2:]
    if len(cli)%2 or any(not x.startswith('--') for x in cli[::2]):raise ValueError('Unexpected publisher command')
    args={k[2:]:v for k,v in zip(cli[::2],cli[1::2])};old=Path(args['output'])
    binding=json.loads((old/'INPUT_BINDING.json').read_text());cfg=json.loads(Path(args['config']).read_text())
    recipe=json.loads(Path(args['recipe']).read_text())
    quality=publication_quality(cfg,Path(args['candidates']),Path(args['source-audit']),Path(args['reference']),Path(args['batch-audit']),args['mode'])
    intent=dict(version=15,job=old.parent.name,old_pid=pid,arguments=args,old_binding_sha256=file_sha(old/'INPUT_BINDING.json'),
        old_batch_id=digest(dict(domain=args['domain'],input_binding=binding,recipe=recipe_identity(recipe))),
        new_output=str(old.parent/'release_v15'),paid_reviews={k:terminal_review(args[k]) for k in ('source-audit','reference','batch-audit')},
        quality_digest=digest(quality),quality_thresholds_changed=False,created_unix=time.time())
    validate(intent)
    if output.exists():
        if json.loads(output.read_text())!=intent:raise ValueError('Existing migration intent must be used, not overwritten')
    else:write(output,intent)
    print(json.dumps(dict(stage='MIGRATION_PREFLIGHT_PASSED_NO_STOP_YET',job=intent['job'],old_batch_id=intent['old_batch_id'])))


def apply(path):
    intent=bound_json(path);binding,cfg=validate(intent);old=Path(intent['arguments']['output'])
    receipt=old.parent/'CPU_MIGRATION_V15.json'
    reason='CPU_MIGRATION_V15:'+file_sha(path)
    with (Path(cfg['root'])/'queue.lock').open('a') as q, (Path(cfg['database']).parent/'production-release.lock').open('a') as p:
        fcntl.flock(q,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(p,fcntl.LOCK_EX|fcntl.LOCK_NB)
        validate(intent)  # under both writer locks, including READY/INTENT check
        ctl=AllStockControl(Path(cfg['database']),json.loads(Path(intent['arguments']['recipe']).read_text()))
        try:
            row=ctl.db.execute('SELECT status,reason,input_binding FROM batches WHERE batch_id=?',(intent['old_batch_id'],)).fetchone()
            if not row or json.loads(row['input_binding'])!=binding:raise ValueError('Wrong old ledger batch')
            if ctl.db.execute('SELECT 1 FROM releases WHERE batch_id=?',(intent['old_batch_id'],)).fetchone():raise ValueError('Committed old batch must never be superseded')
            if row['status']=='quarantined':
                if row['reason']!=reason:raise ValueError('Unrelated quarantine; explicit investigation required')
            else:ctl.hold(intent['old_batch_id'],reason,quarantine=True)
            obj=dict(version=15,status='CPU_ONLY_SUPERSEDED_UNCOMMITTED',job=intent['job'],old_batch_id=intent['old_batch_id'],
                old_output=str(old),new_output=intent['new_output'],intent_path=str(path.resolve()),intent_sha256=file_sha(path),
                candidate_sha256=binding['candidate_sha256'],old_binding_sha256=intent['old_binding_sha256'],
                paid_reviews=intent['paid_reviews'],quality_thresholds_changed=False,old_files_preserved=True,api_requests=0)
            if receipt.exists():
                if json.loads(receipt.read_text())!=obj:raise ValueError('Migration receipt differs')
            else:write(receipt,obj)
        finally:ctl.close()
    print(json.dumps(dict(stage='CPU_MIGRATION_APPLIED',job=intent['job'],api_requests=0)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-pid',type=int);p.add_argument('--intent',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
    if a.apply:apply(a.intent)
    elif a.prepare_pid:prepare(a.prepare_pid,a.intent)
    else:p.error('Choose --prepare-pid or --apply')

if __name__=='__main__':main()
