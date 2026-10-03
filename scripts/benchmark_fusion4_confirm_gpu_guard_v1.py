#!/usr/bin/env python3
"""GPU contention guard; stops only our exact systemd cgroup, not other jobs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time
import benchmark_fusion4_confirm_v1 as bench


def cgroup(pid):
    return (Path('/proc')/str(pid)/'cgroup').read_text()


def validate_parent(parent, actual_parent, own_group, parent_group):
    if parent<=1 or parent!=actual_parent or own_group!=parent_group:
        raise ValueError('must be direct parent in same cgroup')
    paths = [line.split(':',2)[-1] for line in own_group.splitlines()]
    if not any(Path(p).name==bench.UNIT for p in paths):
        raise ValueError('guard requires the exact dedicated unit, not a substring')


def outside_processes(processes, own_group, get_group=cgroup):
    outside=[]
    for process in processes:
        try: group=get_group(process['pid'])
        except FileNotFoundError: continue
        if group!=own_group: outside.append(dict(process,cgroup=group))
    return outside


def stop_own_scope(parent, own_group):
    validate_parent(parent,os.getppid(),own_group,cgroup(parent))
    # SIGTERM on a bash parent may be deferred by foreground torchrun. Stop the
    # exact KillMode=control-group unit instead, so all *our* GPU workers exit.
    subprocess.run(['systemctl','--user','stop','--no-block',bench.UNIT],check=True,timeout=10)


def run(parent):
    own_group=cgroup(os.getpid())
    validate_parent(parent,os.getppid(),own_group,cgroup(parent))
    bench.write(bench.OUT/'guard.started.json',{'parent_pid':parent,'guard_pid':os.getpid(),
                'cgroup':own_group,'unit':bench.UNIT,'interval_seconds':2,
                'started_at':datetime.now(timezone.utc).isoformat(),
                'note':'Not a reservation. Never stops or signals another user or unit.'})
    checks=0
    while os.getppid()==parent:
        snap=bench.gpu_snapshot()
        uuids={g['uuid'] for g in snap['gpus'] if g['index'] in bench.GPU_INDICES}
        outside=outside_processes([p for p in snap['processes'] if p['uuid'] in uuids],own_group)
        checks+=1
        bench.write(bench.OUT/'guard.latest.json',{'checks':checks,'captured_at':snap['captured_at'],
                    'outside_processes':outside,'gpu_snapshot':snap})
        if outside:
            bench.write(bench.OUT/'CONTENTION.json',{'status':'external_gpu_contention',
                      'captured_at':snap['captured_at'],'outside_processes':outside,
                      'action':'stop only own exact dedicated systemd unit'})
            stop_own_scope(parent,own_group)
            return
        time.sleep(2)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-pid',required=True,type=int)
    args=p.parse_args()
    try: run(args.parent_pid)
    except Exception as error:
        bench.write(bench.OUT/'GUARD_ERROR.json',{'status':'guard_failed_closed','error':repr(error)})
        stop_own_scope(args.parent_pid,cgroup(os.getpid()))
        raise


if __name__=='__main__':main()
