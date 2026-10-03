#!/usr/bin/env python3
"""ONLY run inside eval_training_code_sandbox_v1.py's namespace sandbox."""
import json
import os
from pathlib import Path
import socket


def main():
    if os.environ.get('TRAINING_EVAL_SANDBOX')!='1':
        raise RuntimeError('sandbox required')
    mode = os.environ.get('SANDBOX_MODE')
    if mode=='probe':
        if Path('/home/user/.ssh').exists() or Path('/srv/small-agent/datasets').exists():
            raise RuntimeError('host credentials/data exposed')
        try:
            Path('/usr/eval_write_probe').write_text('bad')
        except OSError:
            pass
        else:
            raise RuntimeError('host /usr writable')
        try:
            with socket.create_connection(('1.1.1.1',443),timeout=1): pass
        except OSError:
            pass
        else:
            raise RuntimeError('external network reachable')
        Path('/output/result.json').write_text(json.dumps({'probe_pass':True,'host_secret_visible':False,'network':False,'host_usr_readonly':True}))
        return
    import importlib.metadata
    if importlib.metadata.version('evalplus')!='0.3.1': raise RuntimeError('unlocked evalplus')
    from evalplus.data import get_human_eval_plus,get_mbpp_plus
    from evalplus.evaluate import check_correctness
    from evalplus.gen.util import trusted_exec
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.sanitize import sanitize
    req=json.loads(Path('/input/request.json').read_text())
    dataset,tid=req['dataset'],req['task_id']
    problems=get_human_eval_plus() if dataset=='humaneval' else get_mbpp_plus()
    problem=problems[tid]
    special=[] if dataset=='humaneval' else MBPP_OUTPUT_NOT_NONE_TASKS
    canonical=problem['prompt']+problem['canonical_solution']
    # Same official oracle computation, without get_groundtruth's disk pickle.
    # HumanEval/15's complete plus outputs exceed the sandbox's 16 MiB file cap.
    # Keep all cases and the file cap; hold bounded groundtruth in RAM instead.
    groundtruth={}
    for part in ('base','plus'):
        groundtruth[part],groundtruth[part+'_time']=trusted_exec(canonical,problem[part+'_input'],problem['entry_point'],record_time=True,output_not_none=problem['entry_point'] in special)
    sanity=check_correctness(dataset,0,problem,canonical,groundtruth,fast_check=False)
    if sanity['base'][0]!='pass' or sanity['plus'][0]!='pass':
        raise RuntimeError('canonical solution did not pass official tests')
    solution=sanitize(req['solution'],entrypoint=problem['entry_point'])
    result=check_correctness(dataset,0,problem,solution,groundtruth,fast_check=False)
    # Do not expose or serialize test inputs / canonical solutions in result.
    report={'dataset':dataset,'task_id':tid,'canonical_sanity_pass':True,'base_status':result['base'][0],'plus_status':result['plus'][0], 'base_pass':result['base'][0]=='pass','plus_pass':result['base'][0]=='pass' and result['plus'][0]=='pass','sanitized_changed':solution!=req['solution'],'base_cases':len(problem['base_input']),'plus_cases':len(problem['plus_input'])}
    Path('/output/result.json').write_text(json.dumps(report))


if __name__=='__main__': main()
