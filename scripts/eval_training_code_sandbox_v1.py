#!/usr/bin/env python3
"""CPU-only, per-task bubblewrap runner. Never executes model code on the host."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
from training_eval_common_v1 import sha, wilson, write_json


def command(a, inp, out, mode):
    b=a.eval_root/'sandbox/bwrap-package/usr/bin/bwrap'
    cmd=[str(b),'--unshare-all','--die-with-parent','--new-session','--clearenv','--cap-drop','ALL', '--proc','/proc','--dev','/dev','--tmpfs','/dev/shm','--tmpfs','/tmp','--dir','/home/worker','--chdir','/tmp']
    for p in ['/usr','/lib','/lib64']:
        if Path(p).exists(): cmd += ['--ro-bind',p,p]
    # Only packages and this runner, never the host home, model, optimizer, or key files.
    v=Path(sys.executable).parent.parent
    for p in [v/'bin',v/'lib',v/'pyvenv.cfg',a.eval_root/'deps',a.base_deps,Path(__file__).parent]:
        if p.exists(): cmd += ['--ro-bind',str(p),str(p)]
    cmd += ['--ro-bind',str(inp),'/input','--bind',str(out),'/output','--ro-bind',str(a.data/'evalplus'),'/benchmarks']
    env={'PATH':str(v/'bin')+':/usr/bin:/bin','PYTHONPATH':str(a.eval_root/'deps')+':'+str(a.base_deps),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','HOME':'/home/worker','XDG_CACHE_HOME':'/tmp/cache','TMPDIR':'/tmp','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','TRAINING_EVAL_SANDBOX':'1','SANDBOX_MODE':mode,'HUMANEVAL_OVERRIDE_PATH':'/benchmarks/HumanEvalPlus-v0.1.10.jsonl','MBPP_OVERRIDE_PATH':'/benchmarks/MbppPlus-v0.2.0.jsonl'}
    for k,vv in env.items(): cmd += ['--setenv',k,vv]
    return cmd+['--',sys.executable,'-u',str(Path(__file__).with_name('eval_training_code_worker_v1.py'))]


def execute(a, inp, out, mode, bounds):
    out.mkdir(parents=True,exist_ok=False)
    cmd=command(a,inp,out,mode)
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU,(bounds['cpu_seconds'],bounds['cpu_seconds']+1))
        resource.setrlimit(resource.RLIMIT_AS,(bounds['memory_bytes'],bounds['memory_bytes']))
        resource.setrlimit(resource.RLIMIT_FSIZE,(16*1024*1024,16*1024*1024))
        resource.setrlimit(resource.RLIMIT_NOFILE,(256,256))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    with (out/'worker.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
        try:
            rc=proc.wait(timeout=bounds['wall_seconds_per_task'])
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise RuntimeError(f'sandbox wall timeout: {out}')
    if rc or not (out/'result.json').is_file():
        raise RuntimeError(f'sandbox infrastructure/canonical failure rc={rc}: {out}/worker.log')
    return json.loads((out/'result.json').read_text())


def main():
    p=argparse.ArgumentParser()
    for k in ('eval_root','base_deps','data','generations','config'):
        p.add_argument('--'+k.replace('_','-'),type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());bounds=cfg['code_sandbox']
    b=a.eval_root/'sandbox/bwrap-package/usr/bin/bwrap'
    if not b.is_file(): raise FileNotFoundError(b)
    out=a.generations/'code_execution'
    if out.exists(): raise FileExistsError(out)
    out.mkdir()
    probe=out/'probe_input';probe.mkdir()
    probe_report=execute(a,probe,out/'probe_output','probe',bounds)
    write_json(out/'sandbox_manifest.json',{'binary':str(b),'binary_sha256':sha(b),'unshare_all':True,'readonly_packages':True,'network':False,'host_keys_models_optimizer_not_mounted':True,'bounded':bounds,'probe':probe_report})
    results={}
    for ds in ('humaneval','mbpp'):
        generations=json.loads((a.generations/f'{ds}_generations.json').read_text())
        if not generations['complete']: raise ValueError('incomplete code generations')
        rows=[]
        for record in generations['records']:
            task=out/f'{ds}_{record["task_id"].split("/")[-1]}'
            inp=task/'input';inp.mkdir(parents=True)
            write_json(inp/'request.json',{'dataset':ds,'task_id':record['task_id'],'solution':record['solution']})
            rows.append(execute(a,inp,task/'output','score',bounds))
            write_json(out/f'{ds}_progress.json',rows)
            print(json.dumps(rows[-1]),flush=True)
        n=len(rows);base=sum(x['base_pass'] for x in rows);plus=sum(x['plus_pass'] for x in rows)
        results[ds]={'n':n,'base_pass_at_1':base/n,'plus_pass_at_1':plus/n,'base_wilson95':wilson(base,n),'plus_wilson95':wilson(plus,n),'scope':generations['scope'],'all_official_test_inputs':True,'one_greedy_completion_per_task':True,'failures_counted_not_discarded':True,'records':rows}
    write_json(out/'result.json',{'complete':True,'evalplus_version':'0.3.1','results':results,'generation_identity_sha256':sha(a.generations/'identity.json')})
    from torch.utils.tensorboard import SummaryWriter
    step=json.loads((a.generations/'identity.json').read_text())['checkpoint_step']
    with SummaryWriter(str(a.generations/'tensorboard')) as tb:
        for ds,result in results.items():
            for metric in ('base_pass_at_1','plus_pass_at_1','n'):
                tb.add_scalar(f'eval/{result["scope"]}/{ds}/{metric}',result[metric],step)
            tb.add_text(f'eval/{ds}/protocol',json.dumps(result,indent=2),step)
    print('code eval complete',flush=True)


if __name__=='__main__': main()
