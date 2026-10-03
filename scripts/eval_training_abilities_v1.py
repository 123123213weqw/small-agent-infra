#!/usr/bin/env python3
"""Native frozen reader + official lm-eval prompts/metrics, isolated from training."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
from training_eval_common_v1 import sha, verify_inputs, wilson, write_json


def prepare_tasks(data, package, out):
    out.mkdir(parents=True, exist_ok=False)
    records = {}
    tasks = [
        ('arc_easy','arc/arc_easy.yaml','allenai/ai2_arc','ARC-Easy', {s:data/'ai2_arc/ARC-Easy'/f'{s}-00000-of-00001.parquet' for s in ('train','validation','test')}),
        ('hellaswag','hellaswag/hellaswag.yaml','Rowan/hellaswag','null', {s:data/'hellaswag/data'/f'{s}-00000-of-00001.parquet' for s in ('train','validation')}),
        ('piqa','piqa/piqa.yaml','baber/piqa','null', {s:data/'piqa'/f'{s}.parquet' for s in ('train','validation')}),
        ('gsm8k','gsm8k/gsm8k.yaml','openai/gsm8k','main', {s:data/'gsm8k'/f'{s}.parquet' for s in ('train','test')}),
    ]
    for name, rel, repo, subset, files in tasks:
        src = package/'tasks'/rel
        text = src.read_text()
        for old, new in [(f'task: {name}', f'task: tq_{name}'), (f'dataset_path: {repo}', 'dataset_path: parquet'), (f'dataset_name: {subset}', 'dataset_name: null')]:
            if old not in text: raise ValueError(f'official task structure changed: {old}')
            text = text.replace(old,new,1)
        lines = ['dataset_kwargs:', '  data_files:']
        for split,path in files.items():
            if not path.is_file(): raise FileNotFoundError(path)
            lines.append(f'    {split}: {json.dumps(str(path))}')
        text = text.replace('output_type:', '\n'.join(lines)+'\noutput_type:',1)
        dest = out/f'tq_{name}.yaml'
        dest.write_text(text)
        records[name] = {'official_yaml_sha256':sha(src), 'local_yaml_sha256':sha(dest), 'data':{s:sha(p) for s,p in files.items()},'only_changes':'task name and dataset transport'}
    shutil.copyfile(package/'tasks/hellaswag/utils.py',out/'utils.py')
    records['hellaswag']['utils_sha256'] = sha(out/'utils.py')
    write_json(out/'manifest.json',records)


def load_reader(rm, checkpoint, tokenizer_dir):
    import torch
    from safetensors.torch import load_model
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    from small_agent.models import build_model, load_model_spec
    raw = Tokenizer.from_file(str(tokenizer_dir/'tokenizer.json'))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='<pad>',bos_token='<bos>',eos_token='<eos>',model_max_length=2048,clean_up_tokenization_spaces=False)
    tokenizer.model_input_names=['input_ids','attention_mask']
    if len(tokenizer)!=32768 or (tokenizer.pad_token_id,tokenizer.bos_token_id,tokenizer.eos_token_id)!=(0,1,2):
        raise ValueError('HF tokenizer changed vocabulary')
    for probe in ['机器学习。\n','def add(a, b):\n    return a + b\n','x² + π = 3.14\n  two spaces']:
        ids = raw.encode(probe,add_special_tokens=False).ids
        if tokenizer.encode(probe,add_special_tokens=False)!=ids or tokenizer.decode(ids,skip_special_tokens=False)!=probe:
            raise ValueError('tokenizer adapter parity failed')
    model = build_model(load_model_spec(rm['model_spec']['path']))
    missing,unexpected = load_model(model,str(checkpoint/'model.safetensors'),strict=True)
    if missing or unexpected: raise ValueError('checkpoint load differs')
    model.to('cuda:0').eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = False
    return model,tokenizer


def teacher_forced_cache(model, tokenizer, out):
    import torch
    from small_agent.data import ShardedTokenDataset
    # Real heldout token prefixes, not only short hand-written strings.
    rm = json.loads((out.parent/'identity.json').read_text())['run_manifest']
    data = ShardedTokenDataset(Path(rm['data_manifest']['path']).parent,'validation',2048,verify_hashes=False)
    samples = []
    for ref in [0, len(data)//2]:
        ids = data[ref]['input_ids'].to('cuda:0')[None,:]
        for prefix in [32,128,512,1024]:
            with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
                cached = model(input_ids=ids[:,:prefix],use_cache=True)
                cache = cached.past_key_values
                for offset in range(4):
                    position = prefix + offset
                    full = model(input_ids=ids[:,:position],use_cache=False).logits[:,-1].float()
                    incremental = cached.logits[:,-1].float()
                    p,q = full.softmax(-1), incremental.softmax(-1)
                    margin = float(full.topk(2).values.diff().abs().item())
                    maxabs = float((full-incremental).abs().max())
                    tv = float((p-q).abs().sum()/2)
                    same = bool(full.argmax(-1)==incremental.argmax(-1))
                    samples.append({'sample_ref':ref,'position':position,'max_abs':maxabs,'tv':tv,'top1_same':same,'reference_margin':margin,'gate_pass':maxabs<=.1 and tv<=.02 and (same or margin<=2*maxabs)})
                    cached = model(input_ids=ids[:,position:position+1],past_key_values=cache,use_cache=True,cache_position=torch.tensor([position],device=ids.device), attention_mask=torch.ones((1,position+1),device=ids.device,dtype=torch.long))
                    cache = cached.past_key_values
    report={'version':'teacher_forced_cache_v1','samples':samples,'gate_pass':all(x['gate_pass'] for x in samples),'gate':'max_abs<=0.1, TV<=0.02; top1 difference only allowed inside 2*max_abs near-tie band','not_bitwise_equivalence':True}
    write_json(out,report)
    return report


def generation(model,tokenizer,prompts,config,cache_ok,out,tb,step):
    import torch
    from small_agent.evaluation.generation_smoke import generation_metrics
    records=[]
    for p in prompts['prompts']:
        inputs=tokenizer(p['text'],return_tensors='pt',add_special_tokens=False).to('cuda:0')
        item={**p,'prompt_token_ids':inputs.input_ids[0].tolist()}
        for mode in [False,True] if cache_ok else [False]:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                x=model.generate(**inputs,max_new_tokens=prompts['max_new_tokens'],do_sample=False,use_cache=mode,pad_token_id=0,eos_token_id=2)
            ids=x[0,inputs.input_ids.shape[1]:].tolist()
            text=tokenizer.decode(ids,skip_special_tokens=True)
            name='cached' if mode else 'uncached'
            item[name]={'token_ids':ids,'text':text,'metrics':generation_metrics(ids,text,2)}
            tb.add_text(f'eval/generation/{p["id"]}/{name}',f'输入：{p["text"]}\n\n输出：{text}',step)
        if cache_ok: item['cache_token_exact']=item['cached']['token_ids']==item['uncached']['token_ids']
        records.append(item)
        write_json(out,{'records':records,'complete':len(records)==len(prompts['prompts']),'cache_enabled':cache_ok})
        print(json.dumps({'generation':p['id'],'complete':len(records)}),flush=True)
    tb.add_scalar('eval/generation/uncached_repeated_4gram_rate',sum(x['uncached']['metrics']['repeated_4gram_rate'] for x in records)/len(records),step)
    if cache_ok: tb.add_scalar('eval/generation/cache_exact_fraction',sum(x['cache_token_exact'] for x in records)/len(records),step)
    return records


def adapter_gate(lm,model,out):
    import torch
    from lm_eval.api.instance import Instance
    records=[]
    for context,continuation in [('The capital of France is',' Paris.'),('2 + 3 =',' 5'),('机器学习是','一种方法。')]:
        ce,co=lm._encode_pair(context,continuation)
        ids=torch.tensor([ce+co],device='cuda:0')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(ids[:,:-1],use_cache=False).logits.float()
        logprobs=logits[0,len(ce)-1:].log_softmax(-1)
        expected=float(logprobs.gather(-1,torch.tensor(co,device=ids.device)[:,None]).sum())
        actual=lm.loglikelihood([Instance('loglikelihood',{},(context,continuation),0)])[0][0]
        records.append({'context':context,'continuation':continuation,'native_loglikelihood':expected,'harness_loglikelihood':actual,'abs_difference':abs(actual-expected),'targets':len(co)})
    passed=all(x['abs_difference']<=1e-4 for x in records)
    write_json(out,{'gate_pass':passed,'tolerance':1e-4,'records':records})
    if not passed: raise ValueError('harness adapter parity failed')


def code_generate(model,tokenizer,data,out,limit,maximum,cache_ok,excluded_ids=()):
    import torch
    from evalplus.data import get_human_eval_plus,get_mbpp_plus
    # Override paths are set before importing EvalPlus data modules.
    for name, problems in [('humaneval',get_human_eval_plus()),('mbpp',get_mbpp_plus())]:
        chosen=sorted((x for x in problems if x not in set(excluded_ids)),key=lambda x:int(x.split('/')[-1]))
        if limit is not None: chosen=chosen[:limit]
        records=[]
        for task in chosen:
            problem=problems[task]
            prompt=problem['prompt'].strip()+'\n' # EvalPlus direct-completion convention.
            inputs=tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to('cuda:0')
            if inputs.input_ids.shape[1]+maximum>2048: raise ValueError('code prompt exceeds trained context')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                x=model.generate(**inputs,max_new_tokens=maximum,do_sample=False,use_cache=cache_ok,pad_token_id=0,eos_token_id=2)
            ids=x[0,inputs.input_ids.shape[1]:].tolist()
            completion=tokenizer.decode(ids,skip_special_tokens=True)
            records.append({'task_id':task,'solution':prompt+completion,'prompt':prompt,'completion':completion,'token_ids':ids,'hit_token_budget':len(ids)==maximum})
            write_json(out/f'{name}_generations.json',{'scope':'pilot' if limit else 'full','dataset':name,'records':records,'complete':len(records)==len(chosen)})
            print(json.dumps({'code_generation':task,'done':len(records),'total':len(chosen)}),flush=True)


def main():
    import torch
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from torch.utils.tensorboard import SummaryWriter
    p=argparse.ArgumentParser()
    for k in ['run','checkpoint','tokenizer_dir','config','prompts','data','output']:
        p.add_argument('--'+k.replace('_','-'),type=Path,required=True)
    p.add_argument('--scope',choices=['pilot','full'],default='pilot')
    p.add_argument('--checks',nargs='+',choices=['generation','abilities','code'],default=['generation','abilities','code'])
    p.add_argument('--resume-verified',action='store_true',help='explicitly reuse completed stages with identical inputs; never overwrite a completed suite')
    a=p.parse_args()
    if a.output.exists() and not a.resume_verified: raise FileExistsError(a.output)
    if (a.output/'model_eval_complete.json').exists(): raise FileExistsError('completed model suite')
    a.output.mkdir(parents=True,exist_ok=a.resume_verified)
    rm,cm,identity=verify_inputs(a.run,a.checkpoint,a.tokenizer_dir)
    cfg=json.loads(a.config.read_text())
    prompts=json.loads(a.prompts.read_text())
    dm=json.loads((a.data/'manifest.json').read_text())
    for item in dm['files']:
        if sha(a.data/item['file'])!=item['sha256']: raise ValueError('benchmark data hash changed')
    os.environ['HUMANEVAL_OVERRIDE_PATH']=str(a.data/'evalplus/HumanEvalPlus-v0.1.10.jsonl')
    os.environ['MBPP_OVERRIDE_PATH']=str(a.data/'evalplus/MbppPlus-v0.2.0.jsonl')
    os.environ['HF_DATASETS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
    versions={x:importlib.metadata.version(x) for x in ['torch','transformers','lm_eval','evalplus','accelerate','datasets']}
    source_hash=sha(__file__)
    binding={'identity':identity,'run_manifest':rm,'versions':versions,'config_sha256':sha(a.config),'prompt_sha256':sha(a.prompts),'data_manifest_sha256':sha(a.data/'manifest.json'),'scope':a.scope,'checkpoint_step':cm['step'],'tokens_seen':cm['tokens_seen'],'precision':cfg['precision'],'apply_chat_template':False}
    if a.resume_verified:
        old=json.loads((a.output/'identity.json').read_text())
        if any(old[k]!=v for k,v in binding.items()): raise ValueError('resumed evaluation inputs changed')
        attempt_number=1
        while (a.output/f'resumed_execution_{attempt_number:04d}.json').exists(): attempt_number+=1
        attempt=a.output/f'resumed_execution_{attempt_number:04d}.json'
        write_json(attempt,{'execution_source_sha256':source_hash,'common_source_sha256':sha(Path(__file__).with_name('training_eval_common_v1.py')),'reused_artifacts':{p.name:sha(p) for p in sorted(a.output.glob('*.json'))},'fix':'HF harness hardcodes use_cache=True; adapter now preserves official stop criteria while explicitly selecting audited cache policy'})
        manifest=json.loads((a.output/'tasks/manifest.json').read_text())
        for task,item in manifest.items():
            if sha(a.output/'tasks'/f'tq_{task}.yaml')!=item['local_yaml_sha256']: raise ValueError('local task changed')
    else:
        write_json(a.output/'identity.json',{**binding,'execution_source_sha256':source_hash,'common_source_sha256':sha(Path(__file__).with_name('training_eval_common_v1.py'))})
        prepare_tasks(a.data,Path(lm_eval.__file__).parent,a.output/'tasks')
    torch.manual_seed(cfg['seed'])
    model,tokenizer=load_reader(rm,a.checkpoint,a.tokenizer_dir)
    t0=time.monotonic()
    cache=json.loads((a.output/'cache_parity.json').read_text()) if a.resume_verified else teacher_forced_cache(model,tokenizer,a.output/'cache_parity.json')
    cache_ok=cache['gate_pass'] and not cfg.get('force_uncached_generation',False)
    class NativeHFLM(HFLM):
        def _model_generate(self,context,max_length,stop,**kw):
            from training_eval_harness_adapter_v1 import generate
            return generate(self,context,max_length,stop,cache_ok,kw)
    lm=NativeHFLM(pretrained=model,tokenizer=tokenizer,batch_size=1,max_length=2048,add_bos_token=False,prefix_token_id=1,mixed_precision_dtype=torch.bfloat16,softmax_dtype=torch.float32)
    if not a.resume_verified: adapter_gate(lm,model,a.output/'adapter_parity.json')
    elif not json.loads((a.output/'adapter_parity.json').read_text())['gate_pass']: raise ValueError('previous adapter gate failed')
    from training_eval_harness_adapter_v1 import generate as generate_adapter
    probe=tokenizer('The capital of France is',return_tensors='pt',add_special_tokens=False).to('cuda:0')
    test=generate_adapter(lm,probe.input_ids,probe.input_ids.shape[1]+2,[],False,{'do_sample':False,'attention_mask':probe.attention_mask})
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        expected=model.generate(**probe,max_new_tokens=2,do_sample=False,use_cache=False,pad_token_id=0,eos_token_id=2)
    if not torch.equal(test,expected): raise ValueError('generation harness adapter parity failed')
    write_json(a.output/'generation_adapter_parity.json',{'gate_pass':True,'use_cache':False,'token_ids':test[0].tolist(),'adapter_source_sha256':sha(Path(__file__).with_name('training_eval_harness_adapter_v1.py'))})
    manager=TaskManager(include_path=str(a.output/'tasks'))
    with SummaryWriter(str(a.output/'tensorboard')) as tb:
        tb.add_text('eval/identity',json.dumps(json.loads((a.output/'identity.json').read_text()),ensure_ascii=False,indent=2),cm['step'])
        tb.add_scalar('eval/cache/parity_gate',int(cache_ok),cm['step'])
        if 'generation' in a.checks and not a.resume_verified:
            generation(model,tokenizer,prompts,cfg,cache_ok,a.output/'generation16.json',tb,cm['step'])
        metrics={}
        for task in ['arc_easy','piqa','hellaswag','gsm8k'] if 'abilities' in a.checks else []:
            if a.resume_verified and (a.output/f'{task}_result.json').exists():
                existing=json.loads((a.output/f'{task}_result.json').read_text())
                if existing['protocol_scope']!=a.scope or not (a.output/f'{task}_samples.json').exists(): raise ValueError('incomplete/mismatched task')
                metrics[task]=existing['results']['tq_'+task]
                print(json.dumps({'verified_existing_task':task,'metrics':metrics[task]}),flush=True)
                continue
            limit=None if a.scope=='full' else cfg['pilot']['gsm8k_limit' if task=='gsm8k' else 'multiple_choice_limit']
            result=lm_eval.simple_evaluate(model=lm,tasks=['tq_'+task],num_fewshot=cfg['gsm8k_fewshot'] if task=='gsm8k' else cfg['multiple_choice_fewshot'],limit=limit,batch_size=1,task_manager=manager,bootstrap_iters=100,random_seed=cfg['seed'],numpy_random_seed=cfg['seed'],torch_random_seed=cfg['seed'],fewshot_random_seed=cfg['seed'],apply_chat_template=False,log_samples=True,gen_kwargs={'max_gen_toks':cfg['gsm8k_max_gen_toks']} if task=='gsm8k' else None)
            samples=result.pop('samples')
            write_json(a.output/f'{task}_samples.json',samples)
            result['protocol_scope']=a.scope
            if task=='gsm8k':
                # Actual recorded sample arguments are inspected below, not rebuilt fewshots.
                lengths=[]
                for sample in samples['tq_'+task]:
                    prompt=sample['arguments'][0][0]
                    lengths.append(len(tokenizer.encode(prompt,add_special_tokens=False)))
                result['prompt_token_lengths']=lengths
                result['logged_filter_rows']=len(lengths)
                result['left_truncated_unique_docs']=len({sample['doc_id'] for sample,n in zip(samples['tq_'+task],lengths) if n>2048-cfg['gsm8k_max_gen_toks']})
                result['standard_context_protocol']=result['left_truncated_unique_docs']==0
            write_json(a.output/f'{task}_result.json',result)
            metrics[task]=result['results']['tq_'+task]
            for key,value in metrics[task].items():
                if isinstance(value,(int,float)):
                    tb.add_scalar(f'eval/{a.scope}/{task}/{key}',value,cm['step'])
                    if 'stderr' not in key and key.startswith(('acc,','acc_norm,','exact_match,')):
                        n=len({s['doc_id'] for s in samples['tq_'+task]});tb.add_text(f'eval/{a.scope}/{task}/{key}_wilson95',json.dumps({'n':n,'interval':wilson(round(value*n),n)}),cm['step'])
            tb.flush()
            print(json.dumps({'task_complete':task,'metrics':metrics[task]}),flush=True)
        if 'code' in a.checks:
            code_generate(model,tokenizer,a.data,a.output,None if a.scope=='full' else cfg['pilot']['code_limit_per_dataset'],cfg['code_max_new_tokens'],cache_ok,cfg.get('exclude_evalplus_ids',()))
        if sha(__file__)!=source_hash: raise ValueError('evaluation source changed while running')
        write_json(a.output/'model_eval_complete.json',{'complete':True,'scope':a.scope,'checks':a.checks,'metrics':metrics,'seconds':time.monotonic()-t0,'code_execution_status':'awaiting_sandbox' if 'code' in a.checks else 'not_requested','checkpoint_step':cm['step']})


if __name__=='__main__': main()
