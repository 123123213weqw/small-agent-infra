#!/usr/bin/env python3
"""CPU-only coverage/CI/event roundtrip receipt. Refuses incomplete required stages."""
import argparse
import json
from pathlib import Path
from training_eval_common_v1 import sha, wilson, weighted_loss, write_json


def aggregate_domains(domains):
    mapping={'finewiki_zh':'zh','general_zh':'zh','general_en':'en','wave2_general_en':'en','code_other':'code','code_python':'code','code_shell':'code','math_en':'math'}
    grouped={}
    for domain,value in domains.items():
        if domain not in mapping: raise ValueError(f'unregistered evaluation domain: {domain}')
        grouped.setdefault(mapping[domain],{})[domain]=value
    return {name:{**weighted_loss(values),'samples':sum(v['samples'] for v in values.values()),'raw_domains':sorted(values),'coverage_warning':'small_domain' if sum(v['samples'] for v in values.values())<128 else None} for name,values in grouped.items()}


def main():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter
    p=argparse.ArgumentParser()
    p.add_argument('--eval-root',type=Path,required=True)
    p.add_argument('--abilities',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    a=p.parse_args()
    full=json.loads((a.validation/'result.json').read_text())
    ident=json.loads((a.abilities/'identity.json').read_text())
    model=json.loads((a.abilities/'model_eval_complete.json').read_text())
    code=json.loads((a.abilities/'code_execution/result.json').read_text())
    cache=json.loads((a.abilities/'cache_parity.json').read_text())
    adapter=json.loads((a.abilities/'adapter_parity.json').read_text())
    generation=json.loads((a.abilities/'generation16.json').read_text())
    cached=json.loads((a.abilities/'cached_generation_diagnostic.json').read_text())
    if not all(x['complete'] for x in [model,code,generation,cached]) or not adapter['gate_pass']: raise ValueError('incomplete/gate failed')
    if full['identity']!=ident['identity'] or cached['identity']!=ident['identity']: raise ValueError('evaluation model identity mismatch')
    step=ident['checkpoint_step'];tasks={}
    categories=aggregate_domains(full['domains'])
    with SummaryWriter(str(a.validation/'tensorboard')) as tb:
        for category,value in categories.items():
            for key in ('loss','ppl','predicted_tokens','samples'):
                tb.add_scalar(f'eval/{full["scope"]}/category/{category}/{key}',value[key],step)
    with SummaryWriter(str(a.abilities/'tensorboard')) as tb:
        for task in ['arc_easy','piqa','hellaswag','gsm8k']:
            result=json.loads((a.abilities/f'{task}_result.json').read_text())
            samples=json.loads((a.abilities/f'{task}_samples.json').read_text())['tq_'+task]
            n=len({s['doc_id'] for s in samples})
            metrics=result['results']['tq_'+task]
            cis={k:wilson(round(v*n),n) for k,v in metrics.items() if k.startswith(('acc,','acc_norm,','exact_match,'))}
            tasks[task]={'unique_docs':n,'metrics':metrics,'wilson95':cis,'scope':ident['scope']}
            # Recompute CI from unique docs; never count repeated GSM8K filter rows as extra questions.
            for k,interval in cis.items(): tb.add_text(f'eval/{ident["scope"]}/{task}/{k}_wilson95',json.dumps({'n':n,'interval':interval}),step)
            if task=='gsm8k':
                from tokenizers import Tokenizer
                cfg=json.loads(Path(ident['run_manifest']['run_config']['path']).read_text())
                tok=Tokenizer.from_file(cfg['tokenizer_json'])
                maxgen=json.loads(a.config.read_text())['gsm8k_max_gen_toks']
                lengths={s['doc_id']:len(tok.encode(s['arguments'][0][0],add_special_tokens=False).ids) for s in samples}
                tasks[task]['left_truncated_unique_docs']=sum(x>2048-maxgen for x in lengths.values())
                tasks[task]['max_new_tokens']=maxgen
        report={'version':'training_quality_acceptance_receipt_v1','complete':True,'checkpoint_step':step,'tokens_seen':ident['tokens_seen'],'identity':ident['identity'],'full_validation':full['overall'],'domains':full['domains'],'aggregate_validation':categories,'tasks':tasks,'code':code['results'],'generation':{'n':len(generation['records']),'uncached_mean_repeated_4gram_rate':sum(r['uncached']['metrics']['repeated_4gram_rate'] for r in generation['records'])/len(generation['records']),'cache_token_exact_fraction':cached['exact_fraction']},'cache':{'gate_pass':cache['gate_pass'],'worst_max_abs':max(x['max_abs'] for x in cache['samples']),'worst_tv':max(x['tv'] for x in cache['samples']),'benchmarks_use_cache':cache['gate_pass']},'adapter_gate_pass':adapter['gate_pass'],'training_resumed':False,'optimized_stack_adopted':False}
        tb.add_text('eval/acceptance_summary',json.dumps(report,ensure_ascii=False,indent=2),step)
    events={}
    for root in [a.validation,a.abilities]:
        ea=EventAccumulator(str(root/'tensorboard'),size_guidance={'scalars':0,'tensors':0});ea.Reload()
        events[root.name]={'scalar_tags':ea.Tags()['scalars'],'text_tags':ea.Tags()['tensors']}
        for tag in ea.Tags()['scalars']:
            values=ea.Scalars(tag)
            if not values or any(v.step!=step for v in values): raise ValueError('event checkpoint axis mismatch')
    report['event_roundtrip']=events
    write_json(a.eval_root/'reports'/f'step_{step:08d}_acceptance.json',report)
    print(json.dumps({'complete':True,'tasks':{k:v['unique_docs'] for k,v in tasks.items()},'validation':full['overall']}),flush=True)


if __name__=='__main__': main()
