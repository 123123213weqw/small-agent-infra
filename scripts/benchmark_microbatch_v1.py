#!/usr/bin/env python3
"""Isolated, read-only-checkpoint microbatch benchmark, never production resume."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path('/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1')
RUN = ROOT / 'runs/prior-wave2-full-1040m-8l40-v1'
CHECKPOINT = RUN / 'checkpoints/step_00015934'
CODE = ROOT / 'code'
OUT = ROOT / 'benchmarks/microbatch_v1'
BACKEND_SHA = 'e418e6c69bcf2f6724a8223325190a89f6799a72d75d6a9e0f0d1b283d23af41'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def validate_config(cfg: dict) -> None:
    if cfg.get('version') != 'microbatch_benchmark_v1':
        raise ValueError('wrong benchmark config version')
    if (cfg['world_size'], cfg['global_batch'], cfg['sequence_length']) != (8, 32, 2048):
        raise ValueError('benchmark must preserve eight ranks, batch32, context2048')
    if min(cfg['warmup_steps'], cfg['measurement_steps']) < 1 or cfg['profile_steps'] < 0:
        raise ValueError('invalid step counts')
    if len({v['id'] for v in cfg['variants']}) != len(cfg['variants']):
        raise ValueError('duplicate variant IDs')
    for v in cfg['variants']:
        if min(v['micro_batch'], v['accumulation']) < 1:
            raise ValueError('invalid microbatch')
        if v['micro_batch'] * v['accumulation'] * cfg['world_size'] != cfg['global_batch']:
            raise ValueError('global batch changed')


def rank_microbatches(refs: list[int], rank: int, world: int, micro: int) -> list[list[int]]:
    if world < 1 or not 0 <= rank < world or micro < 1 or len(refs) % (world * micro):
        raise ValueError('invalid sample partition')
    local = refs[rank::world]
    return [local[i:i + micro] for i in range(0, len(local), micro)]


def gradient_passed(metrics: dict, baseline_loss: float, loss: float, limits: dict) -> bool:
    return (all(math.isfinite(float(metrics[k])) for k in ('relative_l2', 'cosine'))
            and metrics['relative_l2'] <= limits['maximum_relative_l2']
            and metrics['cosine'] >= limits['minimum_cosine']
            and abs(loss - baseline_loss) <= limits['maximum_loss_difference'])


def summarize_cuda_activity(events) -> dict:
    """Exclude CUDA annotation ranges, which can duplicate actual kernel timings."""
    activities = defaultdict(float)
    for event in events:
        if not str(event.device_type).lower().endswith('cuda'):
            continue
        if not hasattr(event, 'is_user_annotation'):
            raise ValueError('profiler cannot distinguish CUDA annotations from device activities')
        if not event.is_user_annotation:
            activities[event.name] += event.time_range.elapsed_us() / 1000
    total = sum(activities.values())
    nccl = sum(v for name, v in activities.items() if 'nccl' in name.lower())
    return {'status': 'captured', 'total_cuda_activity_duration_ms': total,
            'nccl_cuda_activity_duration_ms': nccl,
            'nccl_cuda_activity_duration_share': nccl / total if total else None,
            'note': 'Sum of non-annotation CUDA activities on rank0 (kernels and copies). '
                    'Overlapping durations are not a wall-time percentage; NCCL duration can include waits.',
            'top_cuda_activities': sorted(activities.items(), key=lambda x: -x[1])[:20]}


def gpu_is_free() -> None:
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True).strip()
    if text:
        raise RuntimeError('GPU computation already present: ' + text)
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                      '--format=csv,noheader,nounits'], text=True)
    if len(memory.splitlines()) != 8 or any(int(x) > 100 for x in memory.splitlines()):
        raise RuntimeError('all eight GPUs must be free')


def verify_source() -> dict:
    sys.path.insert(0, str(CODE / 'scripts'))
    import trainctl
    if trainctl.sha256_source_tree(CODE) != BACKEND_SHA:
        raise ValueError('frozen training backend changed')
    rm = read(RUN / 'run_manifest.json')
    completed = read(ROOT / 'reports/COMPLETED.json')
    if read(RUN / 'status.json')['state'] != 'completed' or not completed['all_artifact_hashes_verified']:
        raise ValueError('source training not successfully complete')
    if sha(CHECKPOINT / 'manifest.json') != completed['final_checkpoint_manifest_sha256']:
        raise ValueError('final checkpoint identity changed')
    for item in ('model_spec', 'run_config', 'data_manifest'):
        if sha(Path(rm[item]['path'])) != rm[item]['sha256']:
            raise ValueError('frozen source binding changed: ' + item)
    cp = read(CHECKPOINT / 'manifest.json')
    for name, entry in cp['artifacts'].items():
        path = CHECKPOINT / name
        if Path(name).name != name or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError('source artifact changed: ' + name)
    cfg = read(Path(rm['run_config']['path']))
    identity = {'run_manifest_sha256': sha(RUN / 'run_manifest.json'),
                'checkpoint_manifest_sha256': sha(CHECKPOINT / 'manifest.json'),
                'checkpoint_model_sha256': cp['artifacts']['model.safetensors']['sha256'],
                'checkpoint_optimizer_sha256': cp['artifacts']['optimizer.pt']['sha256'],
                'data_manifest_sha256': rm['data_manifest']['sha256'],
                'model_spec_sha256': rm['model_spec']['sha256'],
                'run_config_sha256': rm['run_config']['sha256'],
                'backend_sha256': BACKEND_SHA}
    return {'identity': identity, 'training_config': cfg, 'run_manifest': rm}


def prepare() -> None:
    from small_agent.data import ShardedTokenDataset, DeterministicMixtureSchedule
    gpu_is_free()
    cfg = read(OUT / 'config.json')
    validate_config(cfg)
    if (OUT / 'PREPARED.json').exists() or (OUT / 'COMPLETED.json').exists():
        raise FileExistsError('benchmark already prepared; do not duplicate the run')
    source = verify_source()
    training = source['training_config']
    dataset = ShardedTokenDataset(training['data_dir'], 'train', cfg['sequence_length'], verify_hashes=True)
    schedule = DeterministicMixtureSchedule(dataset, seed=training['seed'],
                                            block_size=training['mixture_block_size'])
    count = 1 + cfg['warmup_steps'] + cfg['measurement_steps'] + cfg['profile_steps']
    refs = [[schedule.sample_ref_at(s * cfg['global_batch'] + k)
             for k in range(cfg['global_batch'])] for s in range(count)]
    if len({r for batch in refs for r in batch}) != count * cfg['global_batch']:
        raise ValueError('benchmark sample plan duplicates records')
    for batch in refs:
        for ref in batch:
            if int(dataset[ref]['valid_tokens']) != cfg['sequence_length']:
                raise ValueError('benchmark requires equal, full-length training samples')
    plan = {'split': 'train', 'global_batch': 32, 'global_refs': refs,
            'rank_assignment': 'strided rank ownership is fixed across all microbatch choices'}
    write(OUT / 'sample_plan.json', plan)
    source.update({'status': 'prepared', 'generated_at': datetime.now(timezone.utc).isoformat(),
                   'config_sha256': sha(OUT / 'config.json'),
                   'sample_plan_sha256': sha(OUT / 'sample_plan.json'),
                   'benchmark_script_sha256': sha(Path(__file__)),
                   'source_run_never_resumed': True})
    write(OUT / 'PREPARED.json', source)
    print(json.dumps({'status': 'prepared', 'steps_per_variant': cfg['warmup_steps'] + cfg['measurement_steps'],
                      'global_samples_checked': count * cfg['global_batch']}), flush=True)


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
    if (sha(OUT / 'config.json') != prepared['config_sha256']
            or sha(OUT / 'sample_plan.json') != prepared['sample_plan_sha256']
            or sha(Path(__file__)) != prepared['benchmark_script_sha256']):
        raise ValueError('benchmark code, config or sample plan changed')
    variant = next(v for v in cfg['variants'] if v['id'] == variant_id)
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
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False
        model.train()
        ddp = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False,
                  gradient_as_bucket_view=True, bucket_cap_mb=25.0)
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

        initial_loss = forward_backward(plan[0])
        gradient_report = None
        if rank == 0:
            reference_path = OUT / 'tmp/baseline_gradients.pt'
            reference_path.parent.mkdir(exist_ok=True)
            if variant_id == 'micro1_acc4':
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
                baseline_loss = read(OUT / 'micro1_acc4/gradient_check.json')['loss']
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
        report = {'status': 'measured', 'variant': variant, 'world_size': world, 'global_batch': 32,
                  'initial_loss': initial_loss, 'measured_steps': len(measured),
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


def summarize() -> None:
    gpu_is_free()
    source = verify_source()
    prepared = read(OUT / 'PREPARED.json')
    if source['identity'] != prepared['identity']:
        raise ValueError('source files changed during benchmark')
    cfg = read(OUT / 'config.json')
    results = []
    for variant in cfg['variants']:
        directory = OUT / variant['id']
        if (directory / 'result.json').exists():
            result = read(directory / 'result.json')
            result['profile'] = read(directory / 'profile.json') if (directory / 'profile.json').exists() else None
            result['worker_exit_code'] = int((directory / 'exit_code.txt').read_text())
        else:
            errors = [read(p) for p in directory.glob('error_rank_*.json')]
            result = {'status': 'oom' if any(e['oom'] for e in errors) else 'failed',
                      'variant': variant, 'errors': errors,
                      'worker_exit_code': int((directory / 'exit_code.txt').read_text())}
        results.append(result)
    candidates = [r for r in results if r['status'] == 'measured' and r['worker_exit_code'] == 0 and r['gradient_check']['passed']]
    best = max(candidates, key=lambda r: r['loss_tokens_per_second']) if candidates else None
    reference = next((r for r in candidates if r['variant']['id'] == 'micro1_acc4'), None)
    summary = {'status': 'benchmark_completed_original_checkpoint_unchanged_gpu_released',
               'generated_at': datetime.now(timezone.utc).isoformat(), 'source_identity': source['identity'],
               'results': results, 'best_measured_variant': best['variant']['id'] if best else None,
               'speedup_over_baseline': best['loss_tokens_per_second'] / reference['loss_tokens_per_second'] if best and reference else None,
               'automatic_production_config_change': False, 'formal_capability_benchmark': False,
               'note': 'Single bounded throughput trial per variant, with full-gradient comparison; small differences need repeat confirmation.'}
    scratch = OUT / 'tmp/baseline_gradients.pt'
    if scratch.exists():
        scratch.unlink()
    summary['temporary_full_gradient_deleted'] = not scratch.exists()
    write(OUT / 'COMPLETED.json', summary)
    lines = ['# 八卡 micro-batch 吞吐对照', '', '每组从同一最终模型与 Adam 状态开始；全局 batch=32、上下文2048、BF16计算、FP32参数。',
             '预热10步、计时30步；加载、完整梯度比较与 profiler 均不计入吞吐。原训练 run 未恢复，未保存新模型。', '',
             '| 配置 | 状态 | 合计 loss-token/s | 中位步耗时/s | 峰值分配/GiB | 梯度相对L2差异 |',
             '|---|---|---:|---:|---:|---:|']
    for r in results:
        v = r['variant']
        if r['status'] == 'measured':
            lines.append(f"| {v['micro_batch']}×{v['accumulation']} | {r['status']} | {r['loss_tokens_per_second']:.0f} | {r['median_step_seconds']:.3f} | {r['peak_allocated_gib']:.2f} | {r['gradient_check']['relative_l2']:.6f} |")
        else:
            lines.append(f"| {v['micro_batch']}×{v['accumulation']} | {r['status']} | — | — | — | — |")
    lines += ['', f"本次最快且通过项目梯度检查的配置：{summary['best_measured_variant']}。", 
              '梯度检查覆盖全部有梯度参数，容差不代表逐位一致。单次短跑不能代替长期稳定性测试，也不证明能力提高。',
              '原始 checkpoint 全部文件重新校验，GPU 已释放；临时完整梯度文件已删除。']
    (OUT / 'report.zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'worker', 'summarize'])
    parser.add_argument('--variant')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'worker':
        if not args.variant:
            parser.error('worker requires --variant')
        worker(args.variant)
    else:
        summarize()


if __name__ == '__main__':
    main()
