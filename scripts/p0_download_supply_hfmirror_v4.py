#!/usr/bin/env python3
"""Pinned, resumable hf-mirror raw download worker; no training approval."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import re
import shutil
import subprocess
import tempfile
import time
from urllib.parse import quote

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.download-')
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def url(item):
    if (not re.fullmatch(r'[\w.-]+/[\w.-]+',item['dataset']) or
            not re.fullmatch(r'[0-9a-f]{40}',item['revision']) or
            '..' in PurePosixPath(item['path']).parts or PurePosixPath(item['path']).is_absolute()):
        raise ValueError('pinned dataset/revision/path required')
    return 'https://hf-mirror.com/datasets/'+item['dataset']+'/resolve/'+item['revision']+'/'+quote(item['path'],safe='/')

def validate(plan):
    if plan.get('stage')!='raw_download_only_not_training_approved' or not plan.get('items'):
        raise ValueError('raw-only nonempty plan required')
    root=Path(plan['allowed_root']).resolve();seen=set()
    for i in plan['items']:
        p=Path(i['destination']).resolve()
        if root not in p.parents or p in seen or not p.name.endswith('.parquet'):
            raise ValueError('destination must be unique and inside registered root')
        seen.add(p);url(i)
        if (i['action'] not in ('download','reuse') or type(i['bytes']) is not int or i['bytes']<=0 or
                not re.fullmatch(r'[0-9a-f]{64}',i['sha256'])):
            raise ValueError('size, upstream LFS SHA and action required')

def verify_file(path,item):
    if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
        raise ValueError('file does not match upstream LFS size/SHA: '+str(path))

def fetch(item,state_root,minimum_free,*,runner=subprocess.run):
    final=Path(item['destination']);result=state_root/(item['sha256']+'.json')
    started=time.time()
    if final.exists():
        verify_file(final,item);status='reused_verified' if item['action']=='reuse' else 'downloaded_verified'
    else:
        if item['action']=='reuse':raise FileNotFoundError('registered old file missing: '+str(final))
        final.parent.mkdir(parents=True,exist_ok=True);partial=final.with_name(final.name+'.part')
        if partial.exists() and partial.stat().st_size>item['bytes']:raise ValueError('oversized partial; inspection required')
        remaining=item['bytes']-(partial.stat().st_size if partial.exists() else 0)
        # Conservative allowance for our two admitted network workers.
        if shutil.disk_usage(final.parent).free<minimum_free+2*remaining:
            raise RuntimeError('disk backpressure: preserving reserved free space')
        if remaining:
            print('FETCH',item['source_id'],Path(item['path']).name,'resume_bytes',item['bytes']-remaining,flush=True)
            runner(['curl','--silent','--show-error','--fail','--location',
                    '--connect-timeout','20','--max-time','1800','--retry','5','--retry-delay','10',
                    '--retry-max-time','3600','--retry-all-errors','--continue-at','-',
                    '--max-filesize',str(item['bytes']),
                    '--output',str(partial),url(item)],check=True)
        verify_file(partial,item)
        # Lock prevents another instance; never silently overwrite an existing final.
        os.link(partial,final);partial.unlink();status='downloaded_verified'
    record={'status':status,'source_id':item['source_id'],'dataset':item['dataset'],
        'revision':item['revision'],'upstream_path':item['path'],'local_path':str(final),
        'bytes':item['bytes'],'sha256':item['sha256'],'training_approved':False,
        'elapsed_s':round(time.time()-started,3),'transport':'hf-mirror_direct_on_v100'}
    atomic_json(result,record);print('VERIFIED',item['source_id'],final.name,item['bytes'],flush=True)
    return record

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--state-root',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();plan=json.loads(a.plan.read_text());validate(plan)
    if not 1<=a.workers<=2:raise ValueError('download concurrency must be 1 or 2')
    a.state_root.mkdir(parents=True,exist_ok=True)
    with (a.state_root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        binding=sha(a.plan);identity=a.state_root/'plan.sha256'
        if identity.exists() and identity.read_text().strip()!=binding:raise ValueError('plan changed; use a new state root')
        if not identity.exists():identity.write_text(binding+'\n')
        records=[];failures=[]
        # Verify reused files first; discovery is not enough to skip downloads.
        for i in plan['items']:
            if i['action']=='reuse':records.append(fetch(i,a.state_root,plan['minimum_disk_free_bytes']))
        downloads=[i for i in plan['items'] if i['action']=='download']
        def progress():
            new=[r for r in records if r['status']=='downloaded_verified']
            atomic_json(a.state_root/'status.json',{'stage':'raw_download_only_not_training_approved',
                'plan_sha256':binding,'verified_files':len(records),'planned_files':len(plan['items']),
                'reused_verified_files':len(records)-len(new),'downloaded_verified_files':len(new),
                'planned_download_files':len(downloads),'downloaded_verified_bytes':sum(r['bytes'] for r in new),
                'planned_download_bytes':sum(i['bytes'] for i in downloads),
                'verified_bytes_including_reuse':sum(r['bytes'] for r in records),'failures':failures,
                'training_approved':False,'new_ready_train_tokens':0})
        progress()
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            jobs={pool.submit(fetch,i,a.state_root,plan['minimum_disk_free_bytes']):i for i in downloads}
            for job in as_completed(jobs):
                i=jobs[job]
                try:records.append(job.result())
                except Exception as e:
                    failures.append({'source_id':i['source_id'],'path':i['path'],'error_type':type(e).__name__})
                    print('FAILED',i['source_id'],i['path'],type(e).__name__,flush=True)
                progress()
        summary={'stage':'raw_download_complete_not_training_approved' if not failures else 'raw_download_incomplete',
            'plan_sha256':binding,'files':records,'failures':failures,'new_ready_train_tokens':0,'training_started':False}
        atomic_json(a.state_root/'summary.json',summary)
        if failures:raise SystemExit(1)

if __name__=='__main__':main()
