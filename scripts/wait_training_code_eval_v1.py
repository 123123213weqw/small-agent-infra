#!/usr/bin/env python3
"""Bounded CPU-only dependency runner for an already-started model eval."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for k in ('eval_root','base_deps','data','generations','config'):
        p.add_argument('--'+k.replace('_','-'),type=Path,required=True)
    p.add_argument('--model-unit',required=True)
    p.add_argument('--timeout',type=int,default=1800)
    a=p.parse_args();end=time.monotonic()+a.timeout
    while not (a.generations/'model_eval_complete.json').exists():
        status=subprocess.run(['systemctl','--user','is-active',a.model_unit],capture_output=True,text=True).stdout.strip()
        if status not in ('active','activating'): raise RuntimeError(f'model eval exited before completion: {status}')
        if time.monotonic()>end: raise TimeoutError('waiting for model eval')
        time.sleep(10)
    value=json.loads((a.generations/'model_eval_complete.json').read_text())
    if not value['complete']: raise RuntimeError('model evaluation not complete')
    cmd=[sys.executable,'-u',str(Path(__file__).with_name('eval_training_code_sandbox_v1.py'))]
    for k in ('eval_root','base_deps','data','generations','config'): cmd += ['--'+k.replace('_','-'),str(getattr(a,k))]
    subprocess.run(cmd,check=True)


if __name__=='__main__': main()
