#!/usr/bin/env python3
"""Wait for the finite campaign, then exercise the real producer/consumer on GPU4."""
import json,os,subprocess,sys,time
from pathlib import Path
Q=Path('/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1/evaluation/training_quality_v2')
R=Q.parents[1];B=Path('/srv/small-agent/experiments/small-agent-base-1b-v1-engineering');E=R/'evaluation/training_quality_v1'
def state():
 s=subprocess.check_output(['systemctl','--user','show','p0-quality-v2-campaign.service','-p','MainPID','-p','ExecMainStatus'],text=True)
 return dict(x.split('=',1) for x in s.splitlines() if '=' in x)
while state()['MainPID']!='0':time.sleep(30)
# This is an independent small-model infrastructure check. A failed 1B
# numerical gate must still block the long study, not this queue lifecycle test.
primary_status=state()['ExecMainStatus']
(Q/'reports/outbox_smoke.json').write_text(json.dumps({'status':'running','primary_exit_status':primary_status,'scope':'independent_small_model_queue_lifecycle','gpu_work_started':True})+'\n')
root=Q/'native_outbox_evaluation';root.mkdir();(root/'runs').mkdir()
for name in ('data','deps','sandbox','cache'):(root/name).symlink_to(E/name,target_is_directory=True)
cfg=json.load(open(Q/'code/configs/eval/training_quality_v2.json'));cfg['pilot']={'multiple_choice_limit':8,'gsm8k_limit':2,'code_limit_per_dataset':2};cfg['gsm8k_max_gen_toks']=32;cfg['code_max_new_tokens']=32;cfg['precision']='native_3547740_param_infrastructure_smoke_fp32_bf16_autocast';(root/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
prompts=json.load(open(Q/'code/configs/eval/training_generation16_v1.json'));prompts['max_new_tokens']=16;(root/'prompts.json').write_text(json.dumps(prompts,indent=2)+'\n')
env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONPATH=f'{E}/deps:{B}/deps:{Q}/code/src:{Q}/code/scripts',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
command=[sys.executable,'-u',str(Q/'code/scripts/training_eval_queue_v1.py'),'--run',str(Q/'runs/native_m8_smoke_v2'),'--eval-root',str(root),'--base-root',str(B),'--config',str(root/'config.json'),'--prompts',str(root/'prompts.json'),'--gpus','4','--watch','--outbox']
with (root/'run.log').open('w') as log:result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
(Q/'reports/outbox_smoke.json').write_text(json.dumps({'status':'passed' if result.returncode==0 else 'failed','exit_code':result.returncode,'primary_exit_status':primary_status,'scope':'real_pinned_checkpoint_outbox_lifecycle_small_model_not_1B_capability'},indent=2)+'\n')
sys.exit(result.returncode)
