#!/usr/bin/env python3
"""Four-card isolated performance experiment; production weights are read-only."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_microbatch_v1 as common
ROOT, RUN, CHECKPOINT, CODE = common.ROOT, common.RUN, common.CHECKPOINT, common.CODE
OUT = ROOT / 'benchmarks/perf4_v1'
sha, read, write = common.sha, common.read, common.write
verify_source, rank_microbatches = common.verify_source, common.rank_microbatches
gradient_passed, summarize_cuda_activity = common.gradient_passed, common.summarize_cuda_activity


def validate_config(cfg):
    if cfg['version'] != 'perf4_benchmark_v1':
        raise ValueError('invalid version')
    if (cfg['world_size'], cfg['global_batch'], cfg['sequence_length']) != (4,32,2048):
        raise ValueError('preserve four ranks, global32, context2048')
    if len(cfg['gpu_indices']) != 4 or len(set(cfg['gpu_indices'])) != 4 or any(i not in range(8) for i in cfg['gpu_indices']):
        raise ValueError('exactly four unique physical GPUs required')
    ids = [v['id'] for v in cfg['variants']]
    if len(ids) != len(set(ids)) or ids[0] != cfg['baseline_id']:
        raise ValueError('unique variants with baseline first required')
    if min(cfg['warmup_steps'],cfg['measurement_steps']) < 1 or cfg['profile_steps'] < 0:
        raise ValueError('invalid steps')
    if cfg['validation_samples'] < 4 or cfg['validation_samples'] % 4:
        raise ValueError('validation count must divide ranks')
    for v in cfg['variants']:
        if Path(v['id']).name != v['id'] or not v['id'] or v['id'] in ('.','..','winner_confirmation'):
            raise ValueError('unsafe or reserved variant ID')
        if (v['micro_batch'],v['accumulation']) != (4,2):
            raise ValueError('fixed micro4 accumulation2 required')
        if v['communication'] not in ('fp32','bf16') or v['checkpointing'] not in ('full','half','none') or v['bucket_mib'] not in (25,100,200):
            raise ValueError('unsupported optimization')


def check_selected_gpus(cfg, gpus, processes):
    selected = [g for g in gpus if g['index'] in cfg['gpu_indices']]
    if len(selected) != 4:
        raise RuntimeError('missing selected GPU')
    uuids = {g['uuid'] for g in selected}
    busy = [p for p in processes if p['uuid'] in uuids]
    if busy or any(g['memory_used_mib'] > 100 for g in selected):
        raise RuntimeError('selected cards not free: '+str(busy or selected))
    return selected


def gpu_snapshot():
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpus=[]
    for line in raw.splitlines():
        i,u,m,t,a = [x.strip() for x in line.split(',')]
        gpus.append({'index':int(i),'uuid':u,'memory_used_mib':int(m),'memory_total_mib':int(t),'utilization':int(a)})
    raw = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    processes=[]
    for line in raw.splitlines():
        u,p,m = [x.strip() for x in line.split(',')]
        processes.append({'uuid':u,'pid':int(p),'memory_mib':m})
    return {'gpus':gpus,'processes':processes,'captured_at':datetime.now(timezone.utc).isoformat()}


def gpu_is_free():
    cfg=read(OUT/'config.json'); validate_config(cfg)
    snap=gpu_snapshot(); check_selected_gpus(cfg,snap['gpus'],snap['processes'])
    return snap


def checkpoint_mask(policy, layers=32):
    if policy not in ('full','half','none'):
        raise ValueError('unknown checkpoint policy')
    return [policy=='full' or (policy=='half' and i%2==0) for i in range(layers)]


def configure_checkpointing(model, policy):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    layers=model.model.layers
    mask=checkpoint_mask(policy,len(layers))
    for layer,enabled in zip(layers,mask):
        if not hasattr(layer,'gradient_checkpointing') or not hasattr(layer,'_gradient_checkpointing_func'):
            raise ValueError('decoder does not support layer-level checkpointing')
        layer.gradient_checkpointing=enabled
    if [l.gradient_checkpointing for l in layers] != mask:
        raise ValueError('checkpoint flags failed')
    return sum(mask)


def validation_refs(total, count):
    if not 1 <= count <= total:
        raise ValueError('invalid validation sample count')
    return [i*(total-1)//(count-1) for i in range(count)] if count>1 else [0]


def prepare():
    from small_agent.data import ShardedTokenDataset, DeterministicMixtureSchedule
    cfg=read(OUT/'config.json'); validate_config(cfg)
    snap=gpu_is_free()
    if (OUT/'PREPARED.json').exists() or (OUT/'COMPLETED.json').exists():
        raise FileExistsError('one-shot benchmark already prepared')
    source=verify_source(); train=source['training_config']
    ds=ShardedTokenDataset(train['data_dir'],'train',2048,verify_hashes=True)
    schedule=DeterministicMixtureSchedule(ds,seed=train['seed'],block_size=train['mixture_block_size'])
    count=1+cfg['warmup_steps']+cfg['measurement_steps']+cfg['profile_steps']
    refs=[[schedule.sample_ref_at(s*32+k) for k in range(32)] for s in range(count)]
    if len({r for b in refs for r in b}) != count*32:
        raise ValueError('duplicate training samples')
    if any(int(ds[r]['valid_tokens'])!=2048 for b in refs for r in b):
        raise ValueError('equal full training sequences required')
    val=ShardedTokenDataset(train['data_dir'],'validation',2048,verify_hashes=False)
    vr=validation_refs(len(val),cfg['validation_samples'])
    write(OUT/'sample_plan.json',{'split':'train','global_batch':32,'global_refs':refs,'rank_assignment':'fixed strided across all variants'})
    write(OUT/'validation_plan.json',{'split':'validation','refs':vr,'total_validation_samples':len(val),'not_full_validation':True})
    source.update({'status':'prepared','generated_at':datetime.now(timezone.utc).isoformat(),
                   'config_sha256':sha(OUT/'config.json'),'sample_plan_sha256':sha(OUT/'sample_plan.json'),
                   'validation_plan_sha256':sha(OUT/'validation_plan.json'),
                   'benchmark_script_sha256':sha(Path(__file__)),'common_script_sha256':sha(Path(common.__file__)),
                   'source_run_never_resumed':True,'gpu_snapshot_before':snap,
                   'shared_host_other_gpus_may_be_active':True,
                   'topology':subprocess.check_output(['nvidia-smi','topo','-m'],text=True)})
    write(OUT/'PREPARED.json',source)
    print(json.dumps({'status':'prepared','selected_gpus':cfg['gpu_indices'],'variants':len(cfg['variants']),
                      'global_samples':count*32,'validation_samples':len(vr)}),flush=True)


def resolve_variant(cfg, variant_id):
    if variant_id=='winner_confirmation':
        selection=read(OUT/'selection.json')
        if selection['config_sha256']!=sha(OUT/'config.json'):
            raise ValueError('selection/config mismatch')
        return selection['confirmation_variant']
    return next(v for v in cfg['variants'] if v['id']==variant_id)


def validation_probe(model, prepared, device, rank, world, micro):
    import torch
    import torch.distributed as dist
    from small_agent.data import ShardedTokenDataset
    if sha(OUT/'validation_plan.json')!=prepared['validation_plan_sha256']:
        raise ValueError('validation plan changed')
    dataset=ShardedTokenDataset(prepared['training_config']['data_dir'],'validation',2048,verify_hashes=False)
    refs=read(OUT/'validation_plan.json')['refs'][rank::world]
    totals=torch.zeros(2,device=device,dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0,len(refs),micro):
            rows=[dataset[r] for r in refs[start:start+micro]]
            cpu={k:torch.stack([r[k] for r in rows]) for k in ('input_ids','labels','attention_mask')}
            n=int(torch.count_nonzero(cpu['labels'][:,1:]!=-100))
            inputs={k:t.pin_memory().to(device,non_blocking=True) for k,t in cpu.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model(**inputs,use_cache=False).loss
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite validation probe')
            totals[0]+=loss.detach().double()*n; totals[1]+=n
    dist.all_reduce(totals)
    model.train()
    loss=float(totals[0]/totals[1])
    return {'loss':loss,'perplexity':math.exp(loss),'predicted_tokens':int(totals[1]),
            'samples':len(refs)*world,'scope':'fixed heldout probe, not full validation or capability eval'}


def validate_against_baseline(cfg, variant_id, initial, final):
    if variant_id==cfg['baseline_id']:
        a=b=0.0
    else:
        base=read(OUT/cfg['baseline_id']/'result.json')
        a=initial['loss']-base['initial_validation']['loss']
        b=final['loss']-base['final_validation']['loss']
    return {'initial_loss_difference':a,'post40_step_loss_difference':b,
            'passed':abs(a)<=cfg['maximum_validation_loss_difference'] and abs(b)<=cfg['maximum_validation_loss_difference'],
            'tolerance':cfg['maximum_validation_loss_difference'],'convergence_certified':False}

def worker(variant_id: str) -> None:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from safetensors.torch import load_model
    from small_agent.data import ShardedTokenDataset
    from small_agent.models import build_model, load_model_spec

    rank, world, local_rank = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=3))
    cfg, prepared = read(OUT / 'config.json'), read(OUT / 'PREPARED.json')
    validate_config(cfg)
    assert world == cfg['world_size']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ','.join(map(str,cfg['gpu_indices']))
    if (sha(OUT / 'config.json') != prepared['config_sha256']
            or sha(OUT / 'sample_plan.json') != prepared['sample_plan_sha256']
            or sha(Path(common.__file__)) != prepared['common_script_sha256']
            or sha(Path(__file__)) != prepared['benchmark_script_sha256']):
        raise ValueError('benchmark code, config or sample plan changed')
    variant = resolve_variant(cfg, variant_id)
    micro, accum = variant['micro_batch'], variant['accumulation']
    report_dir = OUT / variant_id
    report_dir.mkdir(exist_ok=True)
    training, rm = prepared['training_config'], prepared['run_manifest']
    try:
        torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])
        torch.set_float32_matmul_precision('high')
        model = build_model(load_model_spec(Path(rm['model_spec']['path'])))
        missing, unexpected = load_model(model, str(CHECKPOINT / 'model.safetensors'), strict=True)
        assert not missing and not unexpected
        model = model.to(device)
        assert next(model.parameters()).dtype == torch.float32
        checkpointed_layers = configure_checkpointing(model, variant['checkpointing'])
        model.config.use_cache = False
        model.train()
        ddp = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False,
                  gradient_as_bucket_view=True, bucket_cap_mb=variant['bucket_mib'])
        if variant['communication'] == 'bf16':
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            ddp.register_comm_hook(dist.group.WORLD, default_hooks.bf16_compress_hook)
        opt_cfg = training['optimizer']
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=cfg['fixed_learning_rate'],
                                      betas=tuple(opt_cfg['betas']), weight_decay=opt_cfg['weight_decay'], fused=True)
        state = torch.load(CHECKPOINT / 'optimizer.pt', map_location='cpu', weights_only=True, mmap=True)
        optimizer.load_state_dict(state)
        del state
        for group in optimizer.param_groups:
            group['lr'] = cfg['fixed_learning_rate']
        gc.collect()
        torch.manual_seed(cfg['seed'] + rank)
        torch.cuda.manual_seed_all(cfg['seed'] + rank)
        dataset = ShardedTokenDataset(training['data_dir'], 'train', cfg['sequence_length'], verify_hashes=False)
        plan = read(OUT / 'sample_plan.json')['global_refs']

        def forward_backward(batch_refs: list[int], profile: bool = False):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            for k, refs in enumerate(rank_microbatches(batch_refs, rank, world, micro)):
                context = torch.profiler.record_function('bench_data') if profile else nullcontext()
                with context:
                    rows = [dataset[r] for r in refs]
                    inputs = {key: torch.stack([r[key] for r in rows]).pin_memory().to(device, non_blocking=True)
                              for key in ('input_ids', 'labels', 'attention_mask')}
                sync = nullcontext() if k == accum - 1 else ddp.no_sync()
                with sync:
                    context = torch.profiler.record_function('bench_forward') if profile else nullcontext()
                    with context, torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = ddp(**inputs, use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError('non-finite benchmark loss')
                    context = torch.profiler.record_function('bench_backward') if profile else nullcontext()
                    with context:
                        (loss / accum).backward()
                    loss_sum += loss.detach() / accum
            dist.all_reduce(loss_sum)
            loss_sum /= world
            return float(loss_sum)

        initial_validation = validation_probe(model, prepared, device, rank, world, micro)
        initial_loss = forward_backward(plan[0])
        gradient_report = None
        if rank == 0:
            reference_path = OUT / 'tmp/baseline_gradients.pt'
            reference_path.parent.mkdir(exist_ok=True)
            if variant_id == cfg['baseline_id']:
                reference = {name: p.grad.detach().cpu() for name, p in model.named_parameters() if p.grad is not None}
                torch.save(reference, reference_path)
                gradient_report = {'reference': True, 'relative_l2': 0.0, 'cosine': 1.0,
                                   'maximum_absolute_difference': 0.0, 'compared_parameters': sum(t.numel() for t in reference.values()),
                                   'baseline_loss': initial_loss, 'loss': initial_loss, 'passed': True}
                del reference
            else:
                reference = torch.load(reference_path, map_location='cpu', weights_only=True, mmap=True)
                aa = bb = ab = dd = maximum = 0.0
                count = 0
                names = {name for name, p in model.named_parameters() if p.grad is not None}
                if names != set(reference):
                    raise ValueError('gradient parameter coverage changed')
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    a, b = reference[name].to(device), p.grad.detach()
                    delta = a - b
                    aa += float(torch.sum(a.square()))
                    bb += float(torch.sum(b.square()))
                    ab += float(torch.sum(a * b))
                    dd += float(torch.sum(delta.square()))
                    maximum = max(maximum, float(delta.abs().max()))
                    count += a.numel()
                    del a, b, delta
                baseline_loss = read(OUT / cfg['baseline_id'] / 'gradient_check.json')['loss']
                gradient_report = {'reference': False, 'relative_l2': math.sqrt(dd / max(aa, 1e-30)),
                                   'cosine': ab / math.sqrt(max(aa * bb, 1e-60)),
                                   'maximum_absolute_difference': maximum, 'compared_parameters': count,
                                   'baseline_loss': baseline_loss, 'loss': initial_loss}
                gradient_report['passed'] = gradient_passed(gradient_report, baseline_loss, initial_loss, cfg['gradient_check'])
                del reference
            write(report_dir / 'gradient_check.json', gradient_report)
        dist.barrier(device_ids=[local_rank])
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        # The check did not update parameters or Adam state; every variant starts identically.
        rows = []
        for step in range(cfg['warmup_steps'] + cfg['measurement_steps']):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            loss = forward_backward(plan[1 + step])
            grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), opt_cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = torch.tensor(time.perf_counter() - started, device=device, dtype=torch.float64)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            row = {'step': step + 1, 'phase': 'warmup' if step < cfg['warmup_steps'] else 'measurement',
                   'seconds': float(elapsed), 'loss': loss, 'grad_norm': float(grad_norm),
                   'loss_tokens': cfg['global_batch'] * (cfg['sequence_length'] - 1)}
            rows.append(row)
            if rank == 0:
                with (report_dir / 'steps.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                if (step + 1) % 10 == 0:
                    print(json.dumps({'variant': variant_id, **row}), flush=True)
        peak = torch.tensor([torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)], device=device, dtype=torch.float64)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        measured = [r for r in rows if r['phase'] == 'measurement']
        final_validation = validation_probe(model, prepared, device, rank, world, micro)
        validation_screen = validate_against_baseline(cfg, variant_id, initial_validation, final_validation)
        report = {'status': 'measured', 'variant': variant, 'world_size': world, 'global_batch': 32,
                  'initial_loss': initial_loss, 'measured_steps': len(measured),
                  'initial_validation': initial_validation, 'final_validation': final_validation,
                  'validation_screen': validation_screen, 'checkpointed_layers': checkpointed_layers,
                  'selected_gpu_indices': cfg['gpu_indices'],
                  'selection_sha256': sha(OUT / 'selection.json') if variant_id == 'winner_confirmation' else None,
                  'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                                  'nccl': list(torch.cuda.nccl.version()),
                                  'attention_implementation': model.config._attn_implementation},
                  'loss_tokens_per_second': sum(r['loss_tokens'] for r in measured) / sum(r['seconds'] for r in measured),
                  'median_step_seconds': statistics.median(r['seconds'] for r in measured),
                  'maximum_step_seconds': max(r['seconds'] for r in measured),
                  'peak_allocated_gib': float(peak[0]) / 2**30,
                  'peak_reserved_gib': float(peak[1]) / 2**30,
                  'last_loss': rows[-1]['loss'], 'parameter_dtype': 'float32', 'compute_dtype': 'bfloat16',
                  'learning_rate': cfg['fixed_learning_rate'], 'saved_model_checkpoint': False,
                  'timing_note': 'Per-step synchronized wall time; load, gradient check, warmup and profiler excluded.'}
        if rank == 0:
            report['gradient_check'] = gradient_report
            write(report_dir / 'result.json', report)
            print(json.dumps(report), flush=True)

        profiler = None
        if rank == 0 and cfg['profile_steps']:
            try:
                profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                profiler.__enter__()
            except Exception as error:
                write(report_dir / 'profile.json', {'status': 'unavailable', 'error': repr(error)})
                profiler = None
        dist.barrier(device_ids=[local_rank])
        for k in range(cfg['profile_steps']):
            forward_backward(plan[1 + cfg['warmup_steps'] + cfg['measurement_steps'] + k], profile=profiler is not None)
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), opt_cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize(device)
        if profiler is not None:
            try:
                profiler.__exit__(None, None, None)
                activity_report = summarize_cuda_activity(profiler.events())
                activity_report['profile_steps'] = cfg['profile_steps']
                known_ranges = {'nccl:all_reduce', 'DistributedDataParallel.forward', 'Optimizer.step#AdamW.step'}
                if any(name in known_ranges for name, _ in activity_report['top_cuda_activities']):
                    activity_report['status'] = 'aggregation_invalid_annotation_range_remains'
                    activity_report['nccl_cuda_activity_duration_share'] = None
                activity_report['excluded_cuda_annotation_count'] = sum(
                    str(e.device_type).lower().endswith('cuda') and e.is_user_annotation for e in profiler.events())
                write(report_dir / 'profile.json', activity_report)
                (report_dir / 'profile_table.txt').write_text(profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=25))
            except Exception as error:
                write(report_dir / 'profile.json', {'status': 'unavailable', 'error': repr(error)})
        dist.barrier(device_ids=[local_rank])
    except BaseException as error:
        write(OUT / variant_id / f'error_rank_{rank:02d}.json', {'error': repr(error),
              'oom': 'out of memory' in str(error).lower(), 'rank': rank})
        raise
    finally:
        dist.destroy_process_group()


def collect_results(cfg, confirmation=False):
    variants=list(cfg['variants'])
    if confirmation:
        variants.append(resolve_variant(cfg,'winner_confirmation'))
    results=[]
    for variant in variants:
        directory=OUT/variant['id']
        code=int((directory/'exit_code.txt').read_text())
        if (directory/'result.json').exists():
            result=read(directory/'result.json')
            result['profile']=read(directory/'profile.json') if (directory/'profile.json').exists() else None
        else:
            errors=[read(p) for p in directory.glob('error_rank_*.json')]
            result={'status':'oom' if any(e['oom'] for e in errors) else 'failed',
                    'variant':variant,'errors':errors}
        result['worker_exit_code']=code
        results.append(result)
    return results


def eligible(result):
    return (result['status']=='measured' and result['worker_exit_code']==0
            and result['gradient_check']['passed'] and result['validation_screen']['passed'])


def select():
    cfg=read(OUT/'config.json'); validate_config(cfg)
    results=collect_results(cfg)
    candidates=[r for r in results if eligible(r) and not r['variant']['id'].endswith('_repeat')]
    if not candidates:
        raise RuntimeError('no acceptable variant for confirmation')
    best=max(candidates,key=lambda r:r['loss_tokens_per_second'])
    confirmation=dict(best['variant'],id='winner_confirmation',source_variant=best['variant']['id'])
    write(OUT/'selection.json',{'config_sha256':sha(OUT/'config.json'),
          'selected_variant':best['variant']['id'],'confirmation_variant':confirmation,
          'selection_rule':'fastest exit0 initial-gradient-screen and fixed-heldout-screen passing variant; not convergence certification'})
    print(json.dumps({'selected_for_repeat':best['variant']['id']}),flush=True)


def summarize():
    cfg=read(OUT/'config.json'); validate_config(cfg)
    snap=gpu_is_free()
    source=verify_source()
    prepared=read(OUT/'PREPARED.json')
    if source['identity']!=prepared['identity']:
        raise ValueError('production source changed')
    for path,key in ((Path(__file__),'benchmark_script_sha256'),(Path(common.__file__),'common_script_sha256'),
                     (OUT/'config.json','config_sha256'),(OUT/'sample_plan.json','sample_plan_sha256'),
                     (OUT/'validation_plan.json','validation_plan_sha256')):
        if sha(path)!=prepared[key]:
            raise ValueError('benchmark input changed: '+str(path))
    results=collect_results(cfg,confirmation=True)
    base=next(r for r in results if r['variant']['id']==cfg['baseline_id'])
    repeat=next(r for r in results if r['variant']['id']==cfg['baseline_id']+'_repeat')
    winner=results[-1]
    control=[r for r in (base,repeat) if eligible(r)]
    # Equal work in each control, so combine durations rather than taking arithmetic mean TPS.
    base_tps=len(control)/sum(1/r['loss_tokens_per_second'] for r in control) if control else None
    drift=repeat['loss_tokens_per_second']/base['loss_tokens_per_second']-1 if eligible(base) and eligible(repeat) else None
    confirmed=eligible(winner)
    ratio=winner['loss_tokens_per_second']/base_tps if confirmed and base_tps else None
    scratch=OUT/'tmp/baseline_gradients.pt'
    if scratch.exists():
        scratch.unlink()
    summary={'status':'four_card_trial_completed_production_checkpoint_unchanged',
             'generated_at':datetime.now(timezone.utc).isoformat(),'source_identity':source['identity'],
             'selected_gpu_indices':cfg['gpu_indices'],'gpu_snapshot_after':snap,'results':results,
             'selected_variant':read(OUT/'selection.json')['selected_variant'],
             'confirmation_passed':confirmed,'combined_control_tokens_per_second':base_tps,
             'baseline_repeat_speed_drift_fraction':drift,'shared_host_noise_warning':drift is None or abs(drift)>0.10,
             'confirmation_speedup_ratio':ratio,'temporary_full_gradient_deleted':not scratch.exists(),
             'automatic_production_config_change':False,'eight_card_speedup_verified':False,
             'full_validation_rerun':False,'model_capability_certified':False,
             'note':'Warm10/measure30, same checkpoint/Adam/1376 train refs; validation is a fixed128-sample probe before/after40 updates. Other GPUs run unrelated workloads; no automatic eight-card extrapolation.'}
    write(OUT/'COMPLETED.json',summary)
    lines=['# 四卡性能优化对照', '',
           f"物理卡 {cfg['gpu_indices']}，覆盖两侧 NUMA。全局 batch=32，每卡 micro=4、累积2，上下文2048。",
           '每组独立读取同一最终模型和 Adam 状态。FP32参数、BF16计算、固定LR=2e-5。正式 run 未恢复，不保存新模型。',
           '预热10步、计时30步；加载、梯度校验、128条验证探针与 profiler 不计入吞吐。', '',
           '| 配置 | 状态/退出码 | 合计 loss-token/s | 峰值分配/GiB | 梯度相对L2 | 更新后验证loss差 |',
           '|---|---|---:|---:|---:|---:|']
    for r in results:
        v=r['variant']
        label=f"{v['id']}：{v['communication']}，桶{v['bucket_mib']}MiB，GC={v['checkpointing']}"
        if r['status']=='measured':
            lines.append(f"| {label} | {r['status']}/{r['worker_exit_code']} | {r['loss_tokens_per_second']:.0f} | {r['peak_allocated_gib']:.2f} | {r['gradient_check']['relative_l2']:.6f} | {r['validation_screen']['post40_step_loss_difference']:+.6f} |")
        else:
            lines.append(f"| {label} | {r['status']}/{r['worker_exit_code']} | — | — | — | — |")
    lines+=['',f"候选胜者：{summary['selected_variant']}；独立复跑检查通过：{confirmed}。",
            f"复跑吞吐 / 两次基线合并吞吐：{ratio:.4f}。" if ratio else '复跑无有效提速结论。',
            f"两次基线吞吐漂移：{drift:+.2%}。" if drift is not None else '缺少有效基线复跑。',
            'GC full/half/none 分别为32/16/0层重计算；half 仅保留零起始偶数层的重计算。',
            '梯度容差和128条验证探针只是短期诊断，不证明长程收敛一致或能力改善；BF16通信存在舍入误差。',
            'CUDA profiler 排除用户标记范围；activity时长求和仍不等于墙钟占比，NCCL时长可包含等待。',
            '其他GPU上的任务未停止。共享CPU/PCIe可能引入噪声，四卡结果不能直接外推八卡。',
            '正式 checkpoint 全部文件再次校验，四卡 worker 已退出；临时完整梯度文件已删除。']
    (OUT/'report.zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','selected_variant','confirmation_passed','confirmation_speedup_ratio',
                                            'baseline_repeat_speed_drift_fraction','temporary_full_gradient_deleted')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','preflight','worker','select','summarize'])
    parser.add_argument('--variant')
    args=parser.parse_args()
    if args.mode=='prepare': prepare()
    elif args.mode=='preflight': print(json.dumps(gpu_is_free()),flush=True)
    elif args.mode=='worker':
        if not args.variant: parser.error('worker needs --variant')
        worker(args.variant)
    elif args.mode=='select': select()
    else: summarize()


if __name__=='__main__':
    main()
