#!/usr/bin/env python3
"""Bounded CPU post-stage evaluation dispatcher and final campaign receipt."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from training_eval_common_v1 import write_json, sha


def unit_state(name):
    raw=subprocess.check_output(['systemctl','--user','show',name,'-p','ActiveState','-p','MainPID','-p','ExecMainStatus'],text=True)
    return dict(line.split('=',1) for line in raw.splitlines() if '=' in line)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base-root',type=Path,required=True);p.add_argument('--eval-v1',type=Path,required=True);p.add_argument('--wait-seconds',type=int,default=21600)
    p.add_argument('--primary-unit',default='p0-quality-v2-campaign.service')
    a=p.parse_args();deadline=time.monotonic()+a.wait_seconds
    while unit_state(a.primary_unit)['MainPID']!='0':
        if time.monotonic()>deadline: raise TimeoutError('acceptance campaign did not finish within wait budget')
        time.sleep(30)
    state=unit_state(a.primary_unit)
    from training_quality_campaign_v2 import production_identity
    cfg=json.loads((a.root/'config.json').read_text())
    unchanged=production_identity(cfg)==json.loads((a.root/'PREPARED.json').read_text())['production_identity']
    if state['ExecMainStatus']!='0' or not (a.root/'reports/paired_numeric_acceptance.json').is_file():
        write_json(a.root/'reports/FINAL.json',{'status':'acceptance_failed_no_optimized_adoption','primary_unit':state,'production_identity_unchanged':unchanged,'automatic_adoption_allowed':False})
        return
    from training_eval_queue_v1 import dispatch,free_gpus
    # The independent outbox smoke uses GPU4; these evaluations use only 0..3.
    if not free_gpus([0,1,2,3]): raise RuntimeError('selected evaluation GPUs occupied; no overlap started')
    eval_cfg=a.root/'code/configs/eval/training_quality_v2.json'
    protocol=json.loads(eval_cfg.read_text())

    def evaluate(case,gpu):
        run=a.root/'runs'/case
        target=run/'evaluation'
        target.mkdir(exist_ok=True)
        (target/'runs').mkdir(exist_ok=True)
        for name,src in [('deps',a.eval_v1/'deps'),('data',a.eval_v1/'data'),('cache',a.eval_v1/'cache'),('sandbox',a.eval_v1/'sandbox')]:
            if not (target/name).exists(): (target/name).symlink_to(src,target_is_directory=True)
        cp=Path(json.loads((run/'RESULT.json').read_text())['checkpoint'])
        args=SimpleNamespace(run=run,eval_root=target,base_root=a.base_root,config=eval_cfg,
            prompts=a.root/'code/configs/eval/training_generation16_v1.json',gpus=[gpu],once=False,scope='pilot')
        request={'checks':['generation','abilities_pilot','code_pilot']}
        dispatch(args,cp,protocol,request=request)
        output=target/'runs'/f'{case}_queue_step_{json.loads((cp/"manifest.json").read_text())["step"]:08d}_scheduled/abilities'
        return {'case':case,'directory':str(output),'complete':True}

    results=[];errors=[]
    cases=[f'seed{seed}_{variant}' for seed in cfg['seeds'] for variant in ('baseline','candidate')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(evaluate,case,gpu):case for gpu,case in enumerate(cases)}
        for future in as_completed(futures):
            try:results.append(future.result())
            except Exception as error:errors.append({'case':futures[future],'error':repr(error)})
            write_json(a.root/'reports/abilities_progress.json',{'completed':results,'errors':errors,'total':len(cases)})
    numerical=json.loads((a.root/'reports/paired_numeric_acceptance.json').read_text())
    capability=[]
    for seed in cfg['seeds']:
        directories={r['case']:Path(r['directory']) for r in results}
        names=[f'seed{seed}_{variant}' for variant in ('baseline','candidate')]
        if not all(name in directories for name in names): continue
        def scores(path):
            out={}
            for task in ('arc_easy','piqa','hellaswag','gsm8k'):
                report=json.loads((path/f'{task}_result.json').read_text())
                for name,row in report['results'].items():
                    for metric,value in row.items():
                        if metric.startswith(('acc,','acc_norm,','exact_match,')):out[f'{task}/{metric}']=float(value)
            code=json.loads((path/'code_execution/result.json').read_text())
            for task,row in code['results'].items():
                out[f'{task}/base_pass@1']=row['base_pass_at_1'];out[f'{task}/plus_pass@1']=row['plus_pass_at_1']
            generation=json.loads((path/'generation16.json').read_text())
            out['generation/repeated_4gram_fraction']=sum(row['uncached']['metrics']['repeated_4gram_rate'] for row in generation['records'])/len(generation['records'])
            return out
        try:
            left,right=[scores(directories[name]) for name in names]
            gaps={key:right[key]-left[key] for key in left}
            floor=[key for key in left if 'pass@1' in key or key.startswith('gsm8k/') if left[key]==0 and right[key]==0]
            passed=all(value>=-.03 for key,value in gaps.items() if not key.startswith('generation/')) and gaps['generation/repeated_4gram_fraction']<=.05
            capability.append({'seed':seed,'score_gaps':gaps,'diagnostic_gate_passed':passed,'floor_metrics_inconclusive':floor})
        except Exception as error: errors.append({'seed':seed,'error':repr(error),'phase':'aggregation'})
    complete=not errors and len(results)==4 and len(capability)==2
    receipt={'status':'all_measurements_completed' if complete else 'evaluation_incomplete','numeric':numerical,'capability_pairs':capability,'errors':errors,
        'original_fast_convolution_bitwise_equivalence_certified':False,
        'production_identity_unchanged':production_identity(cfg)==json.loads((a.root/'PREPARED.json').read_text())['production_identity'],
        'explicit_adoption_authorized':False,'automatic_adoption_allowed':False,
        'floor_metrics_inconclusive':any(x['floor_metrics_inconclusive'] for x in capability),
        'formal_training_started':False,'gpu_processes_after':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)}
    write_json(a.root/'reports/FINAL.json',receipt)


if __name__=='__main__':main()
