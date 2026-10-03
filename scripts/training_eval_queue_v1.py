#!/usr/bin/env python3
"""Checkpoint dispatcher, explicit one-shot or user-started watch; never resumes training."""
import argparse
import fcntl
import json
import os
import shutil
import signal
from pathlib import Path
import subprocess
import sys
import time
from training_eval_common_v1 import due_checks, sha, write_json


def free_gpus(ids):
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    mapping={int(x.split(',')[0]):x.split(',')[1].strip() for x in devices.splitlines()}
    occupied={x.split(',')[0].strip() for x in text.splitlines() if ',' in x}
    return all(mapping[i] not in occupied for i in ids)


def dispatch(a,cp,cfg,request=None):
    rm=json.loads((a.run/'run_manifest.json').read_text())
    step=int(json.loads((cp/'manifest.json').read_text())['step'])
    config=json.loads(Path(rm['run_config']['path']).read_text())
    checks=request['checks'] if request else due_checks(step,config['max_steps'],cfg)
    if a.once and request is None: checks=['generation','validation_full','abilities_'+a.scope,'code_'+a.scope]
    native='full' if 'validation_full' in checks else ('quick' if 'validation_quick' in checks else None)
    ability=any(x.startswith('abilities_') for x in checks)
    code=any(x.startswith('code_') for x in checks)
    scope='full' if 'abilities_full' in checks else 'pilot'
    prefix=a.run.name+'_' if request else ''
    root=a.eval_root/'runs'/f'{prefix}queue_step_{step:08d}_{("once_"+scope) if a.once else "scheduled"}'
    if root.exists() and (root/'COMPLETE.json').is_file():
        old=json.loads((root/'COMPLETE.json').read_text())
        if old['checkpoint_manifest_sha256']!=sha(cp/'manifest.json') or old['config_sha256']!=sha(a.config) or old.get('prompts_sha256')!=sha(a.prompts): raise ValueError('completed eval inputs changed')
        return
    if root.exists(): raise RuntimeError(f'incomplete prior run; inspect before retry: {root}')
    root.mkdir()
    # Every child reads the same immutable protocol/prompts, even if the user
    # edits the originals while this bounded evaluation is running.
    frozen_config=root/'eval_config.json'; frozen_prompts=root/'prompts.json'
    shutil.copy2(a.config,frozen_config); shutil.copy2(a.prompts,frozen_prompts)
    config_digest,prompts_digest=sha(frozen_config),sha(frozen_prompts)
    env=os.environ.copy()
    env.update(PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HOME=str(a.eval_root/'cache/huggingface'),HF_DATASETS_CACHE=str(a.eval_root/'cache/datasets'),TRITON_CACHE_DIR=str(a.base_root/'triton-cache'),TMPDIR=str(a.base_root/'tmp'),CUDA_VISIBLE_DEVICES=','.join(map(str,a.gpus)))
    src=Path(rm['model_spec']['path']).parents[2]
    common=[sys.executable,'-u']
    scripts=root/'evaluation_code'
    shutil.copytree(Path(__file__).parent,scripts,ignore=shutil.ignore_patterns('__pycache__'))
    write_json(root/'evaluation_source_manifest.json',{str(p.relative_to(scripts)):sha(p) for p in sorted(scripts.glob('*.py'))})
    args=['--checkpoint',str(cp),'--tokenizer-dir',str(a.base_root/'data/tokenizer_v1'),'--config',str(frozen_config)]
    def execute(command, log):
        with log.open('w') as f:
            proc=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                rc=proc.wait(timeout=7200 if scope=='pilot' else 43200)
            except BaseException:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise
            if rc: raise subprocess.CalledProcessError(rc,command)
    if native:
        env['PYTHONPATH']=f'{a.base_root}/deps:{src}/src:{src}:{scripts}'
        execute(common+['-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(a.gpus)}',str(scripts/'eval_training_validation_v1.py'),'--training-run',str(a.run),*args,'--scope',native,'--output',str(root/'validation')],root/'validation.log')
    env['CUDA_VISIBLE_DEVICES']=str(a.gpus[0])
    env['PYTHONPATH']=f'{a.eval_root}/deps:{a.base_root}/deps:{src}/src:{src}:{scripts}'
    targets=['generation']+(['abilities'] if ability else [])+(['code'] if code else [])
    execute(common+[str(scripts/'eval_training_abilities_v1.py'),'--run',str(a.run),*args,'--prompts',str(frozen_prompts),'--data',str(a.eval_root/'data'),'--scope',scope,'--checks',*targets,'--output',str(root/'abilities')],root/'abilities.log')
    execute(common+[str(scripts/'eval_training_cache_generation_v1.py'),'--run',str(a.run),'--checkpoint',str(cp),'--tokenizer-dir',str(a.base_root/'data/tokenizer_v1'),'--prompts',str(frozen_prompts),'--generations',str(root/'abilities')],root/'cache_generation.log')
    if code:
        env['CUDA_VISIBLE_DEVICES']=''
        execute(['systemd-run','--user','--scope','--quiet','--property=TasksMax=64','--property=MemoryMax=4G','--property=CPUQuota=200%',*common,str(scripts/'eval_training_code_sandbox_v1.py'),'--eval-root',str(a.eval_root),'--base-deps',str(a.base_root/'deps'),'--data',str(a.eval_root/'data'),'--generations',str(root/'abilities'),'--config',str(frozen_config)],root/'code_execution.log')
    if native=='full' and ability and code:
        execute(common+[str(scripts/'finalize_training_eval_v1.py'),'--eval-root',str(a.eval_root),'--abilities',str(root/'abilities'),'--validation',str(root/'validation'),'--config',str(frozen_config)],root/'finalize.log')
    if sha(a.config)!=config_digest or sha(a.prompts)!=prompts_digest:
        raise ValueError('evaluation protocol/prompts changed during scoring; frozen results retained for inspection')
    write_json(root/'COMPLETE.json',{'step':step,'checks':checks,'checkpoint_manifest_sha256':sha(cp/'manifest.json'),'config_sha256':config_digest,'prompts_sha256':prompts_digest,'complete':True,'formal_training_started':False})


def main():
    p=argparse.ArgumentParser()
    for k in ('run','eval_root','base_root','config','prompts'): p.add_argument('--'+k.replace('_','-'),type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--once',action='store_true');mode.add_argument('--watch',action='store_true')
    p.add_argument('--scope',choices=['pilot','full'],default='pilot',help='--once only')
    p.add_argument('--plan-only',action='store_true',help='print actual checkpoint/actions without starting a job or creating results')
    p.add_argument('--outbox',action='store_true',help='consume trainer-pinned eval snapshots; removes only completed eval-only weight pins')
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if a.plan_only:
        rm=json.loads((a.run/'run_manifest.json').read_text())
        config=json.loads(Path(rm['run_config']['path']).read_text())
        cp=sorted((a.run/'checkpoints').glob('step_*'))[-1]
        cm=json.loads((cp/'manifest.json').read_text())
        checks=['generation','validation_full','abilities_'+a.scope,'code_'+a.scope] if a.once else due_checks(cm['step'],config['max_steps'],cfg)
        print(json.dumps({'checkpoint':str(cp),'step':cm['step'],'checks':checks,'gpus':a.gpus,'selected_gpus_free':free_gpus(a.gpus),'gpu_job_started':False,'formal_training_started':False},ensure_ascii=False,indent=2))
        return
    with (a.eval_root/'.dispatch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            if a.outbox:
                items=sorted((a.run/'evaluation_requests').glob('step_*'))
                pending=[item for item in items if (item/'request.json').is_file() and not (item/'COMPLETE.json').is_file()]
                if not free_gpus(a.gpus):
                    if a.once: raise RuntimeError('selected GPUs occupied')
                    time.sleep(30);continue
                for item in pending:
                    request=json.loads((item/'request.json').read_text())
                    if request['training_run']!=str(a.run.resolve()) or not request['eval_only_not_resumable']:
                        raise ValueError('outbox run/snapshot binding changed')
                    if request['checkpoint_manifest_sha256']!=sha(item/'checkpoint/manifest.json'):
                        raise ValueError('pinned checkpoint manifest changed')
                    dispatch(a,item/'checkpoint',cfg,request=request)
                    write_json(item/'COMPLETE.json',{'step':request['step'],'checks':request['checks'],'weight_pin_released':True})
                    (item/'checkpoint/model.safetensors').unlink()
                state=json.loads((a.run/'status.json').read_text()) if (a.run/'status.json').exists() else {}
                if a.once or state.get('state') in ('completed','failed'): return
                time.sleep(30);continue
            cps=sorted((a.run/'checkpoints').glob('step_*'))
            cps=[cp for cp in cps if (cp/'manifest.json').is_file()]
            if not cps: raise FileNotFoundError('no committed checkpoint')
            if not free_gpus(a.gpus):
                if a.once: raise RuntimeError('selected GPUs occupied; refusing to overlap or kill tasks')
                time.sleep(30);continue
            for cp in cps[-1:] if a.once else cps:
                dispatch(a,cp,cfg)
            if a.once: return
            rm=json.loads((a.run/'run_manifest.json').read_text())
            final=int(json.loads(Path(rm['run_config']['path']).read_text())['max_steps'])
            last=int(json.loads((cps[-1]/'manifest.json').read_text())['step'])
            if last==final and (a.eval_root/'runs'/f'queue_step_{final:08d}_scheduled/COMPLETE.json').is_file(): return
            time.sleep(30)


if __name__=='__main__': main()
