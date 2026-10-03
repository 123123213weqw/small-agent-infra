#!/usr/bin/env python3
"""Diagnostic cached generation even when cache gate failed; NEVER changes benchmark policy."""
import argparse
import json
from pathlib import Path
from training_eval_common_v1 import sha, verify_inputs, write_json


def main():
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from eval_training_abilities_v1 import load_reader
    from small_agent.evaluation.generation_smoke import generation_metrics,compare_cache_tokens
    p=argparse.ArgumentParser()
    for k in ('run','checkpoint','tokenizer_dir','generations','prompts'):
        p.add_argument('--'+k.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    out=a.generations/'cached_generation_diagnostic.json'
    if out.exists(): raise FileExistsError(out)
    rm,cm,identity=verify_inputs(a.run,a.checkpoint,a.tokenizer_dir)
    old=json.loads((a.generations/'identity.json').read_text())
    if old['identity']!=identity or old['prompt_sha256']!=sha(a.prompts): raise ValueError('comparison inputs changed')
    uncached=json.loads((a.generations/'generation16.json').read_text())
    if not uncached['complete']: raise ValueError('incomplete reference')
    cfg=json.loads(a.prompts.read_text())
    model,tok=load_reader(rm,a.checkpoint,a.tokenizer_dir)
    records=[]
    with SummaryWriter(str(a.generations/'tensorboard')) as tb:
        for record in uncached['records']:
            ids=tok.encode(record['text'],add_special_tokens=False)
            if ids!=record['prompt_token_ids']: raise ValueError('prompt token changed')
            inputs=torch.tensor([ids],device='cuda:0')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                output=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),max_new_tokens=cfg['max_new_tokens'],do_sample=False,use_cache=True,eos_token_id=2,pad_token_id=0)
            tokens=output[0,len(ids):].tolist();text=tok.decode(tokens,skip_special_tokens=True)
            comparison=compare_cache_tokens(tokens,record['uncached']['token_ids'])
            records.append({'id':record['id'],'cached':{'token_ids':tokens,'text':text,'metrics':generation_metrics(tokens,text,2)},'comparison':comparison})
            tb.add_text(f'eval/generation/{record["id"]}/cached_diagnostic',text,cm['step'])
            print(json.dumps({'cached_diagnostic':record['id'],**comparison}),flush=True)
        tb.add_scalar('eval/generation/cache_exact_fraction',sum(r['comparison']['exact_match'] for r in records)/len(records),cm['step'])
    write_json(out,{'complete':True,'benchmark_cache_policy_unchanged':True,'identity':identity,'scope':'diagnostic_not_cache_approval','records':records,'exact_fraction':sum(r['comparison']['exact_match'] for r in records)/len(records)})


if __name__=='__main__': main()
