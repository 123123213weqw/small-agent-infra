#!/usr/bin/env python3
"""Exact eight-card A/N/NS repetitions in a new, immutable campaign directory."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_fusion8_v1 as family

PARENT = family.OUT
OUT = family.ROOT / 'benchmarks/fusion8_v1_confirm1'
UNIT = 'p0-wave2-fusion8-confirm1-v1.service'
GPU_INDICES = list(range(8))
EXPECTED_VARIANTS = [('torch_full', False, False), ('rmsnorm_only', True, False),
                     ('norm_swi', True, True), ('torch_full_repeat', False, False),
                     ('rmsnorm_only_repeat', True, False), ('norm_swi_repeat', True, True)]
family.OUT = OUT
family.UNIT = UNIT
family.engine.OUT = OUT
family.EXPECTED_VARIANTS = EXPECTED_VARIANTS
sha, read, write = family.sha, family.read, family.write
gpu_snapshot = family.gpu_snapshot


def scripts_identity():
    paths = [Path(__file__), Path(family.__file__), Path(family.backend.__file__),
             Path(family.engine.__file__), Path(family.engine.common.__file__),
             OUT / 'benchmark_fusion8_confirm_gpu_guard_v1.py', OUT / 'run.sh']
    return {p.name: sha(p) for p in paths}


family.scripts_identity = scripts_identity


def validate_config(cfg):
    if cfg['version'] != 'fusion8_confirm_benchmark_v1':
        raise ValueError('exact eight-card confirmation version required')
    if (cfg['world_size'], cfg['global_batch'], cfg['sequence_length']) != (8, 32, 2048) or cfg['gpu_indices'] != GPU_INDICES:
        raise ValueError('all eight ordered GPUs, global32 and context2048 required')
    if (cfg['warmup_steps'], cfg['measurement_steps'], cfg['profile_steps'], cfg['validation_samples']) != (10, 30, 2, 128):
        raise ValueError('fixed work and validation budget required')
    if cfg['baseline_id'] != 'torch_full' or [(v['id'], v['fuse_norm'], v['fuse_swiglu']) for v in cfg['variants']] != EXPECTED_VARIANTS:
        raise ValueError('two independent repeats for A, N and NS required')
    for v in cfg['variants']:
        if (v['micro_batch'], v['accumulation'], v['communication'], v['bucket_mib'], v['checkpointing'], v['loss_backend']) != (4, 1, 'bf16', 25, 'full', 'torch'):
            raise ValueError('same eight-card geometry and original loss required')
    if cfg['fixed_learning_rate'] != 2e-5 or cfg['maximum_validation_loss_difference'] != 0.01:
        raise ValueError('fixed learning rate and diagnostic tolerance required')
    if cfg['gradient_check'] != {'maximum_relative_l2': 0.02, 'minimum_cosine': 0.999, 'maximum_loss_difference': 0.01}:
        raise ValueError('fixed gradient thresholds required')
    if cfg.get('phase') != 'exact_eight_card_repetition':
        raise ValueError('explicit exact-eight phase required')


family.validate_config = validate_config
family.engine.validate_config = validate_config


def preflight():
    return family.preflight()


def parent_identity():
    p = read(PARENT / 'PREPARED.json')
    interrupted = read(PARENT / 'INTERRUPTED.json')
    if interrupted['status'] != 'interrupted_by_external_gpu_contention' or interrupted['completed_groups'] != 5:
        raise ValueError('expected archived interrupted parent required')
    if p['identity'] != interrupted['source_identity']:
        raise ValueError('parent source identity mismatch')
    for script in [Path(family.__file__), Path(family.backend.__file__), Path(family.engine.__file__), Path(family.engine.common.__file__)]:
        if sha(script) != p['fusion8_scripts'][script.name]:
            raise ValueError('numerical implementation changed across campaigns')
    for name, key in [('config.json', 'config_sha256'), ('sample_plan.json', 'sample_plan_sha256'),
                      ('validation_plan.json', 'validation_plan_sha256'), ('kernel_preflight.json', 'kernel_preflight_sha256')]:
        if sha(PARENT / name) != p[key]:
            raise ValueError('parent input changed: ' + name)
    return {'prepared_sha256': sha(PARENT / 'PREPARED.json'),
            'interrupted_sha256': sha(PARENT / 'INTERRUPTED.json'),
            'source_identity': p['identity'], 'sample_plan_sha256': p['sample_plan_sha256'],
            'validation_plan_sha256': p['validation_plan_sha256']}


def prepare():
    parent = parent_identity()
    family.prepare()
    p = read(OUT / 'PREPARED.json')
    if p['identity'] != parent['source_identity'] or p['sample_plan_sha256'] != parent['sample_plan_sha256'] or p['validation_plan_sha256'] != parent['validation_plan_sha256']:
        raise ValueError('checkpoint, Adam or sample plans changed')
    p['confirmation_parent'] = parent
    p['fusion8_design'] = 'A, N, NS, A-repeat, N-repeat, NS-repeat; eight ranks, micro4 accumulation1 global32'
    write(OUT / 'PREPARED.json', p)


def repeat_stats(rows):
    tps = family.combined_tps(rows)
    drift = rows[1]['loss_tokens_per_second'] / rows[0]['loss_tokens_per_second'] - 1 if tps else None
    return {'combined_tokens_per_second': tps, 'repeat_drift_fraction': drift,
            'stable_within_10pct': drift is not None and abs(drift) <= 0.1}


def summarize():
    prepared = family.verify_experiment()
    parent = parent_identity()
    source = family.engine.verify_source()
    if prepared['confirmation_parent'] != parent or source['identity'] != prepared['identity']:
        raise ValueError('production source or parent changed')
    results = family.engine.collect_results(read(OUT / 'config.json'))
    by = {r['variant']['id']: r for r in results}
    groups = {name: repeat_stats([by[name], by[name + '_repeat']])
              for name in ['torch_full', 'rmsnorm_only', 'norm_swi']}
    control = groups['torch_full']['combined_tokens_per_second']
    candidates = {}
    for name in ['rmsnorm_only', 'norm_swi']:
        metrics = groups[name]
        ratio = metrics['combined_tokens_per_second'] / control if metrics['combined_tokens_per_second'] and control else None
        candidates[name] = dict(metrics, speedup_ratio=ratio,
            speedup_over_2pct_repeated=ratio is not None and ratio > 1.02 and metrics['stable_within_10pct'] and groups['torch_full']['stable_within_10pct'])
    norm = groups['rmsnorm_only']['combined_tokens_per_second']
    joint = groups['norm_swi']['combined_tokens_per_second']
    scratch = OUT / 'tmp/baseline_gradients.pt'
    scratch.unlink(missing_ok=True)
    passed = all(family.engine.eligible(r) for r in results)
    report = {'status': 'fusion8_exact_repetition_completed' if passed else 'fusion8_exact_repetition_failed_diagnostic_gate',
              'generated_at': datetime.now(timezone.utc).isoformat(), 'source_identity': source['identity'],
              'confirmation_parent': parent, 'results': results, 'groups': groups, 'candidates': candidates,
              'joint_over_norm_speedup_ratio': joint / norm if joint and norm else None,
              'all_diagnostic_gates_passed': passed, 'eight_card_repeat_completed': passed,
              'gpu_snapshot_after': gpu_snapshot(), 'temporary_full_gradient_deleted': not scratch.exists(),
              'world_size': 8, 'global_batch': 32, 'micro_batch': 4, 'accumulation': 1,
              'formal_training_resumed': False, 'production_config_changed': False,
              'capability_or_long_convergence_certified': False, 'full_validation_rerun': False}
    write(OUT / ('COMPLETED.json' if passed else 'FAILED.json'), report)
    lines = ['# RMSNorm／SwiGLU：八卡精确重复对照', '',
             '八卡micro4×累积1、global32、2048上下文；A、N、NS各独立重复两次。',
             '每组从相同step15934模型及Adam重新开始；原始CE、相同训练/验证样本、固定LR=2e-5。',
             '每组10步预热、30步计时、2步额外profiler；128条heldout仅是短期诊断。', '',
             '|组|状态/退出码|loss-token/s|峰值allocated GiB|梯度相对L2|40步后验证loss差|',
             '|---|---|---:|---:|---:|---:|']
    for r in results:
        if r['status'] == 'measured':
            lines.append(f"|{r['variant']['id']}|{r['status']}/{r['worker_exit_code']}|{r['loss_tokens_per_second']:.0f}|{r['peak_allocated_gib']:.4f}|{r['gradient_check']['relative_l2']:.6f}|{r['validation_screen']['post40_step_loss_difference']:+.7f}|")
        else:
            lines.append(f"|{r['variant']['id']}|{r['status']}/{r['worker_exit_code']}|—|—|—|—|")
    lines += ['', f'两次原始对照合并：{control:.2f} loss-token/s。' if control else '无完整有效原始对照。']
    for name, metrics in candidates.items():
        ratio = metrics['speedup_ratio']
        lines.append(f'{name}：两次合并相对原始 {ratio:.4f}，重复提速>2%门槛：{metrics["speedup_over_2pct_repeated"]}。' if ratio else f'{name}：无完整有效结论。')
    lines += ['相同八卡几何内部比较，不混合四卡或历史八卡绝对吞吐。',
              '梯度及40步验证通过不等于长期训练质量无损；没有切换生产配置或恢复正式训练。']
    (OUT / 'report.zh.md').write_text('\n'.join(lines) + '\n')
    print(report['status'], flush=True)
    if not passed:
        raise RuntimeError('one or more diagnostic gates failed')


def finish():
    """ExecStopPost receipt after our unit's workers have been reaped."""
    completed = read(OUT / 'COMPLETED.json') if (OUT / 'COMPLETED.json').exists() else None
    snap = gpu_snapshot()
    def unit_state(unit):
        text = subprocess.check_output(['systemctl', '--user', 'show', unit,
            '--property=ActiveState,SubState,MainPID,ExecMainStatus'], text=True)
        return dict(line.split('=', 1) for line in text.splitlines() if '=' in line)
    own_processes = []
    for process in snap['processes']:
        try: group = (Path('/proc') / str(process['pid']) / 'cgroup').read_text()
        except FileNotFoundError: continue
        if any(Path(line.split(':', 2)[-1]).name == UNIT for line in group.splitlines()):
            own_processes.append(process)
    parent = parent_identity()
    source = family.engine.verify_source()
    unchanged = source['identity'] == parent['source_identity']
    verified = not (OUT / 'PREPARED.json').exists() or family.verify_experiment()['confirmation_parent'] == parent
    service_ok = os.environ.get('SERVICE_RESULT') == 'success' and os.environ.get('EXIT_STATUS') == '0'
    status = ('passed' if completed and completed['all_diagnostic_gates_passed'] and unchanged and verified and not own_processes and service_ok
              else 'interrupted_by_external_gpu_contention' if (OUT / 'CONTENTION.json').exists()
              else 'waiting_expired' if (OUT / 'WAIT_EXPIRED.json').exists() else 'failed_or_interrupted')
    receipt = {'status': status, 'checked_at': datetime.now(timezone.utc).isoformat(),
               'service_result': os.environ.get('SERVICE_RESULT'), 'service_exit_status': os.environ.get('EXIT_STATUS'),
               'own_cuda_processes': own_processes, 'own_cuda_workers_released': not own_processes,
               'gpu_snapshot': snap, 'production_identity': source['identity'],
               'production_identity_unchanged': unchanged, 'experiment_and_parent_input_hashes_verified': verified,
               'temporary_full_gradients_deleted': not (OUT / 'tmp/baseline_gradients.pt').exists(),
               'production_unit': unit_state('p0-wave2-base1b-full8-v1.service'),
               'tensorboard_unit': unit_state('small-agent-tensorboard.service'),
               'eight_card_repeat_completed': bool(completed and completed['eight_card_repeat_completed']),
               'production_config_changed': False, 'formal_training_resumed': False}
    write(OUT / 'POSTCHECK.json', receipt)
    print(status, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preflight', 'kernel', 'prepare', 'worker', 'summarize', 'finish'])
    p.add_argument('--variant')
    args = p.parse_args()
    if args.mode == 'preflight':
        import json
        print(json.dumps(preflight()), flush=True)
    elif args.mode == 'kernel': family.kernel_check()
    elif args.mode == 'prepare': prepare()
    elif args.mode == 'worker':
        if not args.variant: p.error('worker requires --variant')
        validate_config(read(OUT / 'config.json'))
        family.worker(args.variant)
    elif args.mode == 'summarize': summarize()
    else: finish()


if __name__ == '__main__': main()
