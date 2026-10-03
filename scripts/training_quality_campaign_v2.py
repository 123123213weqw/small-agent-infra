#!/usr/bin/env python3
"""Isolated 1B eight-rank resume and paired convergence acceptance campaign.

The production directory is read-only. Final experimental checkpoints cannot
be adopted automatically; a successful result is bounded evidence, not a
certificate for every optimizer phase or a from-scratch convergence claim.
"""
from contextlib import nullcontext
from datetime import timedelta
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from training_eval_common_v1 import sha, write_json


def read(path): return json.loads(Path(path).read_text())


def same_state(left,right):
    """Full value/dtype comparison, independent of Torch pickle storage IDs."""
    import torch
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and left.dtype==right.dtype and torch.equal(left.cpu(),right.cpu())
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(same_state(v,right[k]) for k,v in left.items())
    if isinstance(left,(list,tuple)):return type(left)==type(right) and len(left)==len(right) and all(same_state(x,y) for x,y in zip(left,right))
    return type(left)==type(right) and left==right


def verify_loaded_payload(cp,model,optimizer):
    import torch
    from safetensors import safe_open
    with safe_open(cp/'model.safetensors',framework='pt',device='cpu') as saved:
        aliases=saved.metadata() or {}
        model_exact=all(same_state(parameter.detach(),saved.get_tensor(aliases.get(name,name))) for name,parameter in model.named_parameters())
    original=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=True)
    optimizer_exact=same_state(optimizer.state_dict(),original)
    return {'rank0_full_model_payload_exact':model_exact,'rank0_full_optimizer_payload_exact':optimizer_exact,
            'all_rank_artifact_integrity_checked_by_checkpoint_loader':True,'source_checkpoint':str(cp)}


def create_case_directory(out,rank):
    import torch.distributed as dist
    error=[None]
    if rank==0:
        try:out.mkdir(exist_ok=False)
        except OSError as exc:error[0]=f'{type(exc).__name__}: {exc}'
    dist.broadcast_object_list(error,src=0)
    if error[0] is not None:raise RuntimeError('case directory initialization failed: '+error[0])


def resume_ddp_options(config,is_resume):
    """Public DDP option for a stable reference layout; NEVER used by long A/B.

    Torch 2.5.1's find_unused_parameters=True, static_graph=False disables
    history-dependent bucket rebuilding and uses fixed construction buckets.
    This is a diagnostic execution policy, not a native reducer certificate.
    """
    fixed=is_resume and config.get('resume_fixed_ddp_bucket_plan',False)
    if fixed and config.get('resume_conv_reduction')!='deterministic_weight_reference':
        raise ValueError('fixed bucket diagnostic requires explicitly scoped reference convolution')
    return {'find_unused_parameters':True,'static_graph':False} if fixed else {}


def tensor_fingerprint(tensor):
    import torch
    raw=tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


def production_identity(config):
    from small_agent.training.trainer import sha256_source_tree
    run = Path(config['production_run']); cp = Path(config['checkpoint'])
    rm, cm = read(run/'run_manifest.json'), read(cp/'manifest.json')
    identity = {'source_tree_sha256': sha256_source_tree(Path(config['production_code'])),
                'run_manifest_sha256': sha(run/'run_manifest.json'), 'checkpoint_manifest_sha256': sha(cp/'manifest.json')}
    if identity['source_tree_sha256'] != rm['source_tree_sha256']: raise ValueError('formal source changed')
    for key in ('run_config','model_spec','data_manifest'):
        actual = sha(rm[key]['path'])
        if actual != rm[key]['sha256']: raise ValueError('formal input changed: '+key)
        identity[key+'_sha256'] = actual
    for name in ('model.safetensors','optimizer.pt'):
        actual = sha(cp/name)
        if actual != cm['artifacts'][name]['sha256']: raise ValueError('formal checkpoint changed: '+name)
        identity[name+'_sha256'] = actual
    return identity


def prepare(config, root):
    identity = production_identity(config)
    if (root/'PREPARED.json').exists(): raise FileExistsError('campaign already prepared')
    write_json(root/'PREPARED.json', {'production_identity': identity,
        'config_sha256': sha(root/'config.json'), 'worker_sha256': sha(Path(__file__)),
        'experiment_source_manifest': {str(p.relative_to(root/'code')): sha(p) for p in sorted((root/'code').rglob('*.py')) if '__pycache__' not in p.parts},
        'formal_training_started': False, 'automatic_adoption_allowed': False})


def worker(config, root, case):
    if case.startswith('resume_'):
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        # NCCL can otherwise select different reduction trees/protocols across
        # fresh processes. This is an isolated reproducibility diagnostic, not
        # a change to the formal training communication policy.
        os.environ['NCCL_ALGO']='Ring'
        os.environ['NCCL_PROTO']='Simple'
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from small_agent.models import build_model, load_model_spec
    from small_agent.data import ShardedTokenDataset, DeterministicMixtureSchedule, MixtureDistributedSampler
    from small_agent.training.trainer import sharded_train_batches
    from small_agent.training.checkpoint import load_distributed_full_checkpoint, save_distributed_full_checkpoint
    from small_agent.training.scheduler import WarmupCosineConfig, TokenLRScheduler
    from small_agent.training.quality import HealthWindow, LayerDiagnostics, ReferencePanel, domain_plan, quick_validation

    rank, local, world = int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    if world != 8 or os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3,4,5,6,7': raise ValueError('exact eight-card campaign required')
    device = torch.device('cuda',local); torch.cuda.set_device(device)
    if case.startswith('resume_'):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    prepared = read(root/'PREPARED.json')
    if sha(root/'config.json') != prepared['config_sha256'] or sha(Path(__file__)) != prepared['worker_sha256']:
        raise ValueError('campaign inputs changed')
    for name, digest in prepared['experiment_source_manifest'].items():
        if sha(root/'code'/name) != digest: raise ValueError('campaign source changed: '+name)
    out = root/'runs'/case; create_case_directory(out,rank)
    production_run, original_cp = Path(config['production_run']), Path(config['checkpoint'])
    rm, cm = read(production_run/'run_manifest.json'), read(original_cp/'manifest.json')
    training = read(rm['run_config']['path'])
    is_resume = case.startswith('resume_')
    candidate = case.endswith('_candidate')
    seed = training['seed'] if is_resume else int(case.split('_')[0][4:])
    target_updates = {'resume_continuous':8,'resume_first':4,'resume_second':8}.get(case,config['normal_updates']+config['high_lr_updates'])
    if not is_resume:
        if rank==0:
            eval_config=dict(training,max_steps=cm['step']+target_updates,seed=seed,purpose='isolated_quality_campaign_not_formal_training')
            write_json(out/'run_config.json',eval_config)
        dist.barrier()
    torch.manual_seed(training['seed']); torch.cuda.manual_seed_all(training['seed']); torch.set_float32_matmul_precision('high')
    model = build_model(load_model_spec(rm['model_spec']['path'])).to(device)
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if is_resume and config.get('resume_conv_reduction','native')=='deterministic_weight_reference':
        from training_conv_reduction_diagnostic_v3 import install
        policy=install()
        if rank==0:write_json(out/'convolution_diagnostic_policy.json',policy)
    if candidate:
        from benchmark_fusion_backend_v1 import install_fusion
        info=install_fusion(model,True,False)
        for i,layer in enumerate(model.model.layers): layer.gradient_checkpointing=i%2==0
        if rank==0: write_json(out/'fusion.json',info)
    embedding_trace=[];trace_position={'update':0,'microstep':0}
    if is_resume and config.get('trace_local_embedding_gradient',False):
        def trace_embedding(parameter):
            if 0<trace_position['update']<=2:
                embedding_trace.append(dict(trace_position,sha256=tensor_fingerprint(parameter.grad),
                    dtype=str(parameter.grad.dtype),shape=list(parameter.grad.shape)))
        model.get_input_embeddings().weight.register_post_accumulate_grad_hook(trace_embedding)
    ddp_options=resume_ddp_options(config,is_resume)
    ddp=DDP(model,device_ids=[local],output_device=local,broadcast_buffers=False,gradient_as_bucket_view=True,bucket_cap_mb=25,**ddp_options)
    if is_resume:
        write_json(out/f'ddp_policy_rank_{rank:05d}.json',{'options':ddp_options,'torch':torch.__version__,
            'diagnostic_only':True,'original_dynamic_reducer_bitwise_equivalence_certified':False})
    if candidate:
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        ddp.register_comm_hook(dist.group.WORLD,default_hooks.bf16_compress_hook)
    opt_cfg=training['optimizer']
    optimizer=torch.optim.AdamW(ddp.parameters(),lr=opt_cfg['maximum_learning_rate'],betas=tuple(opt_cfg['betas']),weight_decay=opt_cfg['weight_decay'],fused=True)
    scheduler=TokenLRScheduler(optimizer,WarmupCosineConfig(opt_cfg['maximum_learning_rate'],opt_cfg['minimum_learning_rate'],training['scheduler']['warmup_tokens'],training['scheduler']['total_tokens']))
    dataset=ShardedTokenDataset(training['data_dir'],'train',2048,verify_hashes=False)
    schedule=DeterministicMixtureSchedule(dataset,seed=training['seed'],block_size=training['mixture_block_size'])
    sampler=MixtureDistributedSampler(schedule,num_replicas=world,rank=rank,micro_batch_size=1,gradient_accumulation_steps=4)
    generator=torch.Generator().manual_seed(training['seed']+10000+rank)
    compatibility={'version':'training_quality_campaign_v2','source_checkpoint_manifest_sha256':sha(original_cp/'manifest.json'),
                   'config_sha256':sha(root/'config.json'),'world_size':8,'global_batch':32,'context_length':2048,
                   'seed':seed,'variant':'candidate' if candidate else 'original','experiment_only':True}
    if not is_resume:
        compatibility.update(run_config_sha256=sha(out/'run_config.json'),model_spec_sha256=rm['model_spec']['sha256'],data_manifest_sha256=rm['data_manifest']['sha256'])
        if rank==0:
            write_json(out/'run_manifest.json',dict(rm,run_id=case,purpose='isolated_quality_campaign',
                run_config={'path':str(out/'run_config.json'),'sha256':sha(out/'run_config.json')},
                parent_checkpoint=str(original_cp),experimental_training_only=True))
    source_cp=root/'runs/resume_first/checkpoints'/f"step_{cm['step']+4:08d}" if case=='resume_second' else original_cp
    source_cm=read(source_cp/'manifest.json')
    load_distributed_full_checkpoint(source_cp,model=model,optimizer=optimizer,scheduler=scheduler,sampler=sampler,loader_generator=generator,
        compatibility=compatibility if case=='resume_second' else cm['compatibility'],device=device,rank=rank,world_size=world)
    if is_resume and config.get('verify_loaded_payload',False):
        verdict=[None]
        if rank==0:
            check=verify_loaded_payload(source_cp,model,optimizer)
            write_json(out/'loaded_payload_verification.json',check)
            verdict[0]=check['rank0_full_model_payload_exact'] and check['rank0_full_optimizer_payload_exact']
        dist.broadcast_object_list(verdict,src=0)
        if not verdict[0]:raise ValueError('checkpoint payload hydration changed tensor values/dtypes')
    if case=='resume_first':
        # Both independent paths start with the same observed numerical kernel
        # plan; no random number or parameter/optimizer state is changed here.
        from small_agent.training.kernel_state import restore_kernel_state
        control=read(root/'runs/resume_continuous/RESULT.json')
        runtime=torch.load(Path(control['checkpoint'])/f'runtime_rank_{rank:05d}.pt',map_location='cpu',weights_only=True)
        restore_kernel_state(runtime['kernel_autotune'])
    from safetensors import safe_open
    dtype_counts={}
    with safe_open(original_cp/'model.safetensors',framework='pt',device='cpu') as source:
        for name,parameter in model.named_parameters():
            # save_model stores one copy of tied embeddings and records aliases.
            stored_name=(source.metadata() or {}).get(name,name)
            expected={'F32':torch.float32,'BF16':torch.bfloat16}[source.get_slice(stored_name).get_dtype()]
            if parameter.dtype!=expected:raise ValueError('original parameter precision layout changed: '+name)
            dtype_counts[str(parameter.dtype)]=dtype_counts.get(str(parameter.dtype),0)+parameter.numel()
    for parameter,state in optimizer.state.items():
        for key in ('exp_avg','exp_avg_sq'):
            if state[key].dtype!=parameter.dtype: raise ValueError('original Adam moment precision layout changed')
    if rank==0:write_json(out/'precision.json',{'parameter_counts_by_dtype':dtype_counts,'source_precision_layout_preserved':True,
        'note':'original checkpoint contains 2304 BF16 gated-GDN norm weights; all remaining parameters and matching Adam moments are FP32; no new precision downgrade'})
    start_offset=source_cm['step']-cm['step']
    if not is_resume:
        # Same checkpoint/Adam; independent paired data seeds and identical rank
        # assignment. Optimized micro4 combines the original rank's four samples.
        sampler.schedule=DeterministicMixtureSchedule(dataset,seed=seed,block_size=training['mixture_block_size'])
        torch.manual_seed(seed+rank);torch.cuda.manual_seed_all(seed+rank)
    micro=4 if candidate else 1;accum=1 if candidate else 4
    batches=sharded_train_batches(dataset,sampler,micro,generator,dict(num_workers=0,pin_memory=True))
    health=HealthWindow(device,opt_cfg['gradient_clip'])
    diagnostics=LayerDiagnostics(model,rank==0,interval=200)
    tb=SummaryWriter(str(out/'tensorboard')) if rank==0 else None
    val=ShardedTokenDataset(training['data_dir'],'validation',2048,verify_hashes=False)
    plan=domain_plan(val,128)
    external=ReferencePanel(config['reference_panel'],dataset.manifest['tokenizer_sha256'],sha(Path(training['data_dir'])/'manifest.json')) if not is_resume else None
    ext_plan=domain_plan(external,100000) if external is not None else None
    records=[]

    def validation(offset, full=False):
        selected=domain_plan(val,100000) if full else plan
        result=quick_validation(model,val,selected,device,rank,world,1,(generator,))
        reference=quick_validation(model,external,ext_plan,device,rank,world,1,(generator,)) if external else None
        if rank==0:
            write_json(out/f'validation_{offset:04d}.json',{'corpus':result,'reference':reference,'training_updates':offset,'full_corpus':full})
            for group,value in [('corpus',result),('reference',reference)]:
                if value:
                    for domain,row in {'overall':value['overall'],**value['domains']}.items():
                        tb.add_scalar(f'eval/{group}/{domain}/loss',row['loss'],cm['step']+offset)
                        tb.add_scalar(f'eval/{group}/{domain}/ppl',row['ppl'],cm['step']+offset)
            tb.flush()

    try:
        if not is_resume: validation(0,full=True)
        for offset in range(start_offset+1,target_updates+1):
            trace_position['update']=offset
            model.train();diagnostics.begin(offset)
            optimizer.zero_grad(set_to_none=True)
            losses=torch.zeros((),device=device,dtype=torch.float64)
            local_tokens=0;refs=[]
            torch.cuda.synchronize(device);started=time.perf_counter()
            for microstep in range(accum):
                trace_position['microstep']=microstep
                batch=next(batches);refs.extend(batch['sample_ref'].tolist())
                targets=int((batch['labels'][:,1:]!=-100).sum())
                if targets != micro*2047: raise ValueError('paired study requires full equal-length training samples')
                inputs={key:batch[key].to(device,non_blocking=True) for key in ('input_ids','labels','attention_mask')}
                with nullcontext() if microstep==accum-1 else ddp.no_sync():
                    with torch.autocast('cuda',dtype=torch.bfloat16): loss=ddp(**inputs,use_cache=False).loss
                    if microstep==0: diagnostics.close()
                    if not torch.isfinite(loss): raise FloatingPointError('nonfinite campaign loss')
                    (loss/accum).backward()
                losses+=loss.detach().double()*targets;local_tokens+=targets
            stats=diagnostics.gradients()
            if is_resume and offset<=2 and rank==0:
                # Full per-parameter byte fingerprints, before clipping, let
                # us localize the first divergence without saving 4 GB dumps.
                fingerprints={}
                for name,parameter in model.named_parameters():
                    if parameter.grad is not None:
                        fingerprints[name]=tensor_fingerprint(parameter.grad)
                write_json(out/f'gradients_{offset:04d}.json',fingerprints)
            if is_resume:
                write_json(out/f'ddp_reducer_rank_{rank:05d}_{offset:04d}.json',ddp._get_ddp_logging_data())
                if embedding_trace:
                    write_json(out/f'local_embedding_rank_{rank:05d}.json',embedding_trace)
            norm=torch.nn.utils.clip_grad_norm_(ddp.parameters(),opt_cfg['gradient_clip'],error_if_nonfinite=True)
            lr=scheduler.step(local_tokens*world)
            if not is_resume:
                lr=config['low_learning_rate'] if offset<=config['normal_updates'] else config['high_learning_rate']
                for group in optimizer.param_groups: group['lr']=lr
            optimizer.step();sampler.commit_step()
            torch.cuda.synchronize(device);elapsed=time.perf_counter()-started
            health.add(losses,local_tokens,norm,elapsed)
            gathered=[None]*world;dist.all_gather_object(gathered,refs)
            # Record every consumed reference, not an inferred schedule digest.
            row={'update':offset,'lr':lr,'loss':float(losses/local_tokens),'grad_norm':float(norm),
                 'seconds_rank0':elapsed,'global_refs':gathered,'tokens_seen':scheduler.tokens_seen}
            if rank==0:
                with (out/'steps.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
                for tag,value in stats.items():tb.add_scalar(tag,value,cm['step']+offset)
            if offset%10==0 or offset==target_updates:
                window=health.report();health.reset()
                if rank==0:
                    print(json.dumps({'case':case,'update':offset,'lr':lr,**window}),flush=True)
                    for tag,value in window.items():tb.add_scalar('train/'+tag,value,cm['step']+offset)
                    tb.add_scalar('train/learning_rate',lr,cm['step']+offset);tb.flush()
                    write_json(out/'progress.json',{'case':case,'update':offset,'target_updates':target_updates,'state':'running','updated_unix':time.time(),**window})
            if not is_resume and (offset%500==0 or offset==target_updates): validation(offset,full=offset==target_updates)
        cp,saved=save_distributed_full_checkpoint(out/'checkpoints',step=cm['step']+target_updates,model=model,optimizer=optimizer,scheduler=scheduler,
            sampler=sampler,loader_generator=generator,compatibility=compatibility,rank=rank,world_size=world,keep=1)
        if rank==0:
            write_json(out/'RESULT.json',{'case':case,'checkpoint':str(cp),'source_checkpoint':str(original_cp),'updates':target_updates,
                'model_sha256':saved['artifacts']['model.safetensors']['sha256'],'optimizer_sha256':saved['artifacts']['optimizer.pt']['sha256'],
                'sampler':sampler.state_dict(),'scheduler':scheduler.state_dict(),'formal_training_started':False,'automatic_adoption_allowed':False})
            write_json(out/'progress.json',{'case':case,'update':target_updates,'state':'completed'})
    finally:
        diagnostics.close()
        if tb:tb.close()
        dist.destroy_process_group()


def compare_resume(root):
    import torch
    from safetensors import safe_open
    a,b=read(root/'runs/resume_continuous/RESULT.json'),read(root/'runs/resume_second/RESULT.json')
    p,q=Path(a['checkpoint']),Path(b['checkpoint'])
    differences={}
    with safe_open(p/'model.safetensors',framework='pt',device='cpu') as left, safe_open(q/'model.safetensors',framework='pt',device='cpu') as right:
        if list(left.keys())!=list(right.keys()): raise ValueError('resume tensor coverage differs')
        maximum=square=reference=0.
        for name in left.keys():
            x,y=left.get_tensor(name),right.get_tensor(name)
            delta=x-y;maximum=max(maximum,float(delta.abs().max()));square+=float(delta.double().square().sum());reference+=float(x.double().square().sum())
        differences={'model_max_abs':maximum,'model_relative_l2':math.sqrt(square/max(reference,1e-60))}
    # Exact serialization hash fast path; otherwise compare ALL optimizer
    # tensor values/dtypes and metadata, never just sampled moment tensors.
    optimizer_bytes_exact=a['optimizer_sha256']==b['optimizer_sha256']
    optimizer_exact=optimizer_bytes_exact
    if not optimizer_exact and differences['model_max_abs']==0:
        optimizer_exact=same_state(torch.load(p/'optimizer.pt',map_location='cpu',weights_only=True),torch.load(q/'optimizer.pt',map_location='cpu',weights_only=True))
    runtime_exact=all(sha(p/f'runtime_rank_{rank:05d}.pt')==sha(q/f'runtime_rank_{rank:05d}.pt') for rank in range(8))
    # torch serialization bytes can differ even for identical states. Compare
    # runtime tensors directly when hashes differ.
    if not runtime_exact:
        runtime_exact=all(same_state(torch.load(p/f'runtime_rank_{rank:05d}.pt',weights_only=True),torch.load(q/f'runtime_rank_{rank:05d}.pt',weights_only=True)) for rank in range(8))
    def rows(case): return [read_json for line in (root/'runs'/case/'steps.jsonl').read_text().splitlines() for read_json in [json.loads(line)]]
    continuous=rows('resume_continuous');resumed=rows('resume_first')+rows('resume_second')
    order_exact=[r['global_refs'] for r in continuous]==[r['global_refs'] for r in resumed]
    lr_exact=[r['lr'] for r in continuous]==[r['lr'] for r in resumed]
    sampler_exact=all(read(p/f'sampler_rank_{rank:05d}.json')==read(q/f'sampler_rank_{rank:05d}.json') for rank in range(8))
    passed=differences['model_max_abs']==0 and optimizer_exact and runtime_exact and order_exact and lr_exact and sampler_exact and a['scheduler']==b['scheduler']
    report={'passed':passed,'exact_model':differences['model_max_abs']==0,'exact_optimizer':optimizer_exact,
            'optimizer_serialization_bytes_exact':optimizer_bytes_exact,
            'exact_runtime_rng':runtime_exact,'exact_data_order':order_exact,'exact_learning_rate':lr_exact,'exact_sampler':sampler_exact,
            'exact_scheduler':a['scheduler']==b['scheduler'],**differences,'world_size':8,'updates':8,'split_at_update':4,
            'reproducibility_policy':'fixed_original_Triton_choices_PyTorch_deterministic_CUBLAS_4096_8_NCCL_Ring_Simple',
            'uncontrolled_autotune_runtime_bitwise_equivalence_certified':False}
    config=read(root/'config.json')
    report['fixed_ddp_bucket_reference']=bool(config.get('resume_fixed_ddp_bucket_plan',False))
    report['original_dynamic_reducer_bitwise_equivalence_certified']=False
    if report['fixed_ddp_bucket_reference']:
        def bucket_plan(case,rank,offset):
            info=read(root/'runs'/case/f'ddp_reducer_rank_{rank:05d}_{offset:04d}.json')
            if info.get('has_rebuilt_buckets',0) or info.get('find_unused_parameters')!=1:
                raise ValueError('fixed diagnostic DDP policy was not preserved')
            return info['bucket_sizes']
        plans=[bucket_plan('resume_continuous',rank,offset) for rank in range(8) for offset in range(1,9)]
        other=[bucket_plan('resume_first' if offset<=4 else 'resume_second',rank,offset) for rank in range(8) for offset in range(1,9)]
        report['fixed_ddp_bucket_sizes_exact']=len(set(plans+other))==1
        report['passed']=passed=passed and report['fixed_ddp_bucket_sizes_exact']
    report['convolution_reduction']=config.get('resume_conv_reduction','native')
    report['original_fast_convolution_bitwise_equivalence_certified']=False
    if config.get('verify_loaded_payload',False):
        checks=[read(root/'runs'/case/'loaded_payload_verification.json') for case in ('resume_continuous','resume_first','resume_second')]
        payload_exact=all(x['rank0_full_model_payload_exact'] and x['rank0_full_optimizer_payload_exact'] for x in checks)
        report['checkpoint_payload_hydration_exact']=payload_exact
        report['passed']=passed=passed and payload_exact
    if report['convolution_reduction']=='deterministic_weight_reference':
        report['scope']='checkpoint_restore_on_deterministic_dweight_and_fixed_ddp_reference_not_native_bitwise_certificate' if report['fixed_ddp_bucket_reference'] else 'checkpoint_restore_on_deterministic_dweight_reference_only_not_native_bitwise_certificate'
        report['conv_preflight']=read(root/'reports/conv_preflight.json')
        report['passed']=passed=passed and report['conv_preflight']['passed']
    write_json(root/'reports/resume_acceptance.json',report)
    if not passed:raise RuntimeError('exact resume gate failed; inspect before any long study')


def summarize(config,root):
    paired=[]
    for seed in config['seeds']:
        left,right=[root/'runs'/f'seed{seed}_{variant}' for variant in ('baseline','candidate')]
        final=config['normal_updates']+config['high_lr_updates']
        a,b=[read(p/f'validation_{final:04d}.json') for p in (left,right)]
        left_rows=[json.loads(x) for x in (left/'steps.jsonl').read_text().splitlines()]
        right_rows=[json.loads(x) for x in (right/'steps.jsonl').read_text().splitlines()]
        order=[(x['global_refs'],x['lr'],x['tokens_seen']) for x in left_rows]==[(x['global_refs'],x['lr'],x['tokens_seen']) for x in right_rows]
        gaps={f'{group}/{domain}':b[group]['domains'][domain]['loss']-values['loss'] for group in ('corpus','reference') for domain,values in a[group]['domains'].items()}
        delta=b['corpus']['overall']['loss']-a['corpus']['overall']['loss']
        # Conservative screening limits, explicitly not a statistical proof.
        passed=order and abs(delta)<=config['maximum_overall_loss_gap'] and all(abs(v)<=config['maximum_domain_loss_gap'] for v in gaps.values())
        paired.append({'seed':seed,'data_lr_token_order_exact':order,'overall_loss_gap':delta,'domain_loss_gaps':gaps,'diagnostic_gate_passed':passed})
    report={'status':'paired_long_numeric_screen_completed','pairs':paired,'all_numeric_gates_passed':all(p['diagnostic_gate_passed'] for p in paired),
            'resume_gate':read(root/'reports/resume_acceptance.json'),'production_identity_unchanged':production_identity(config)==read(root/'PREPARED.json')['production_identity'],
            'abilities_and_generation_comparison':'pending_independent_checkpoint_evaluation',
            'automatic_adoption_allowed':False,'formal_training_started':False,'from_scratch_quality_certified':False}
    write_json(root/'reports/paired_numeric_acceptance.json',report)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','compare-resume','summarize']);p.add_argument('--root',type=Path,required=True);p.add_argument('--case')
    a=p.parse_args();config=read(a.root/'config.json')
    if a.mode=='prepare':prepare(config,a.root)
    elif a.mode=='worker':worker(config,a.root,a.case)
    elif a.mode=='compare-resume':compare_resume(a.root)
    else:summarize(config,a.root)


if __name__=='__main__':main()
