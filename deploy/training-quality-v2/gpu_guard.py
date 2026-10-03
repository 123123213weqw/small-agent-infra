#!/usr/bin/env python3
"""Observe shared GPUs; stop only this exact acceptance unit on contention."""
import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=int,required=True);p.add_argument('--unit',default='p0-quality-v2-campaign.service');a=p.parse_args()
unit=a.unit
own=Path('/proc/self/cgroup').read_text()
if os.getppid()!=a.parent or Path(f'/proc/{a.parent}/cgroup').read_text()!=own or not any(Path(x.split(':',2)[-1]).name==unit for x in own.splitlines()):raise RuntimeError('exact own parent cgroup required')
while os.getppid()==a.parent:
 try:
  raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
  outside=[]
  for line in raw.splitlines():
   if ',' not in line:continue
   uuid,pid=line.split(',');pid=int(pid)
   try:group=Path(f'/proc/{pid}/cgroup').read_text()
   except FileNotFoundError:continue
   if group!=own:outside.append({'pid':pid,'gpu_uuid':uuid.strip()})
  if outside:
   (a.root/'reports/CONTENTION.json').write_text(json.dumps({'outside_processes':outside,'action':'stop only our exact cgroup'})+'\n')
   subprocess.run(['systemctl','--user','stop','--no-block',unit],check=True);break
 except Exception as e:
  (a.root/'reports/GUARD_ERROR.json').write_text(json.dumps({'error':repr(e)})+'\n')
  subprocess.run(['systemctl','--user','stop','--no-block',unit]);raise
 time.sleep(3)
