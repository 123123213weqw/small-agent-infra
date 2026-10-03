#!/usr/bin/env python3
"""Useful eight-card norm-fusion/GC/bucket trials after exact fusion repetitions."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_fusion8_confirm_v1 as support

PREREQUISITE = support.OUT
OUT = support.family.ROOT / 'benchmarks/fusion8_tune_v1'
UNIT = 'p0-wave2-fusion8-tune-v1.service'
GPU_INDICES = list(range(8))
support.OUT = OUT
support.UNIT = UNIT
support.family.OUT = OUT
support.family.UNIT = UNIT
support.family.engine.OUT = OUT
sha, read, write = support.sha, support.read, support.write
gpu_snapshot = support.gpu_snapshot
engine = support.family.engine
DESIGN = [('norm_full25', 'full', 25), ('norm_half25', 'half', 25),
          ('norm_full100', 'full', 100), ('norm_half100', 'half', 100),
          ('norm_full25_repeat', 'full', 25)]


def scripts_identity():
    files = [Path(__file__), Path(support.__file__), Path(support.family.__file__),
             Path(support.family.backend.__file__), Path(engine.__file__), Path(engine.common.__file__),
             OUT / 'benchmark_fusion8_tune_gpu_guard_v1.py', OUT / 'run.sh', OUT / 'after.sh']
    return {p.name: sha(p) for p in files}


support.family.scripts_identity = scripts_identity


def validate_config(cfg):
    if cfg['version'] != 'fusion8_tune_benchmark_v1' or cfg.get('phase') != 'after_exact_fusion8_confirmation':
        raise ValueError('bounded follow-up tuning phase required')
    if (cfg['world_size'], cfg['global_batch'], cfg['sequence_length']) != (8, 32, 2048) or cfg['gpu_indices'] != GPU_INDICES:
        raise ValueError('fixed eight ranks, global32, context2048 required')
    if (cfg['warmup_steps'], cfg['measurement_steps'], cfg['profile_steps'], cfg['validation_samples']) != (10, 30, 2, 128):
        raise ValueError('fixed work budget required')
    if cfg['baseline_id'] != 'norm_full25' or [(v['id'], v['checkpointing'], v['bucket_mib']) for v in cfg['variants']] != DESIGN:
        raise ValueError('predeclared norm/full-half/25-100 scan and baseline repeat required')
    for v in cfg['variants']:
        if (v['micro_batch'], v['accumulation'], v['communication'], v['loss_backend'], v['fuse_norm'], v['fuse_swiglu']) != (4, 1, 'bf16', 'torch', True, False):
            raise ValueError('only checkpointing and communication bucket may vary')
    if cfg['seed'] != 20261002 or cfg['fixed_learning_rate'] != 2e-5 or cfg['maximum_validation_loss_difference'] != 0.01:
        raise ValueError('fixed RNG, LR and diagnostic tolerance required')
    if cfg['gradient_check'] != {'maximum_relative_l2': 0.02, 'minimum_cosine': 0.999, 'maximum_loss_difference': 0.01}:
        raise ValueError('fixed full-gradient thresholds required')


support.validate_config = validate_config
support.family.validate_config = validate_config
engine.validate_config = validate_config


def preflight(): return support.preflight()


def prerequisite_identity():
    p = read(PREREQUISITE / 'PREPARED.json')
    completed = read(PREREQUISITE / 'COMPLETED.json')
    post = read(PREREQUISITE / 'POSTCHECK.json')
    if completed['status'] != 'fusion8_exact_repetition_completed' or not completed['all_diagnostic_gates_passed'] or not completed['eight_card_repeat_completed']:
        raise ValueError('exact-eight parent must have passed all gates')
    if post['status'] != 'passed' or not post['own_cuda_workers_released'] or not post['production_identity_unchanged']:
        raise ValueError('parent cleanup and source checks required')
    if completed['source_identity'] != p['identity'] or post['production_identity'] != p['identity']:
        raise ValueError('prerequisite source identities differ')
    for script in [Path(support.__file__), Path(support.family.__file__), Path(support.family.backend.__file__), Path(engine.__file__), Path(engine.common.__file__)]:
        if sha(script) != p['fusion8_scripts'][script.name]:
            raise ValueError('parent numerical code changed')
    return {'prepared_sha256': sha(PREREQUISITE / 'PREPARED.json'),
            'completed_sha256': sha(PREREQUISITE / 'COMPLETED.json'), 'postcheck_sha256': sha(PREREQUISITE / 'POSTCHECK.json'),
            'source_identity': p['identity']}


def prepare():
    prerequisite = prerequisite_identity()
    support.prepare()
    p = read(OUT / 'PREPARED.json')
    if p['identity'] != prerequisite['source_identity']:
        raise ValueError('source changed between sequential campaigns')
    p['optimization_prerequisite'] = prerequisite
    p['fusion8_design'] = 'norm-only: full25, half25, full100, half100, full25-repeat, eligible-winner-repeat'
    write(OUT / 'PREPARED.json', p)


def worker(name):
    cfg = read(OUT / 'config.json'); validate_config(cfg)
    if name == 'winner_confirmation':
        selection = read(OUT / 'selection.json')
        selected = next(v for v in cfg['variants'] if v['id'] == selection['selected_variant'])
        expected = dict(selected, id=name, source_variant=selected['id'])
        if selection['config_sha256'] != sha(OUT / 'config.json') or selection['confirmation_variant'] != expected:
            raise ValueError('winner must be an exact predeclared configuration')
    support.family.worker(name)


def summarize():
    prepared = support.family.verify_experiment()
    if prepared['optimization_prerequisite'] != prerequisite_identity():
        raise ValueError('completed prerequisite changed')
    source = engine.verify_source()
    if source['identity'] != prepared['identity']:
        raise ValueError('production source changed')
    cfg = read(OUT / 'config.json')
    results = engine.collect_results(cfg, confirmation=True)
    by = {r['variant']['id']: r for r in results}
    valid = [r for r in results[:-1] if engine.eligible(r) and not r['variant']['id'].endswith('_repeat')]
    expected = max(valid, key=lambda r: r['loss_tokens_per_second'])['variant']['id']
    if read(OUT / 'selection.json')['selected_variant'] != expected:
        raise ValueError('selection does not match eligible measured results')
    base = support.repeat_stats([by['norm_full25'], by['norm_full25_repeat']])
    winner = support.repeat_stats([by[expected], by['winner_confirmation']])
    bt, wt = base['combined_tokens_per_second'], winner['combined_tokens_per_second']
    ratio = wt / bt if wt and bt else None
    passed = all(engine.eligible(r) for r in results)
    scratch = OUT / 'tmp/baseline_gradients.pt'; scratch.unlink(missing_ok=True)
    report = {'status': 'fusion8_tuning_completed' if passed else 'fusion8_tuning_failed_gate',
              'generated_at': datetime.now(timezone.utc).isoformat(), 'source_identity': source['identity'],
              'results': results, 'selected_variant': expected, 'baseline_repetition': base, 'winner_repetition': winner,
              'winner_speedup_ratio_over_norm_full25': ratio,
              'speedup_over_2pct_repeated': ratio is not None and ratio > 1.02 and base['stable_within_10pct'] and winner['stable_within_10pct'],
              'all_diagnostic_gates_passed': passed, 'optimization_prerequisite': prepared['optimization_prerequisite'],
              'optimization_repeat_completed': passed, 'eight_card_repeat_completed': False,
              'world_size': 8, 'global_batch': 32, 'micro_batch': 4, 'accumulation': 1,
              'formal_training_resumed': False, 'production_config_changed': False,
              'capability_or_long_convergence_certified': False, 'full_validation_rerun': False,
              'temporary_full_gradient_deleted': not scratch.exists(), 'gpu_snapshot_after': gpu_snapshot()}
    write(OUT / ('COMPLETED.json' if passed else 'FAILED.json'), report)
    lines = ['# 八卡RMSNorm候选：重计算与通信桶优化', '',
             '前置八卡精确重复及收尾门禁通过后才运行；本轮不是正式训练。',
             '只变32层完整/16层隔层重计算、25/100MiB通信桶；每组仍有相同checkpoint/Adam、global32、训练/验证样本。',
             'RMSNorm训练期融合、原始CE、BF16通信，SwiGLU不融合。', '',
             '|组|重计算|桶MiB|loss-token/s|峰值allocated GiB|梯度相对L2|验证loss差|',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        v = r['variant']
        if engine.eligible(r):
            lines.append(f"|{v['id']}|{v['checkpointing']}|{v['bucket_mib']}|{r['loss_tokens_per_second']:.0f}|{r['peak_allocated_gib']:.3f}|{r['gradient_check']['relative_l2']:.6f}|{r['validation_screen']['post40_step_loss_difference']:+.7f}|")
        else: lines.append(f"|{v['id']}|{v['checkpointing']}|{v['bucket_mib']}|失败或未通过门禁|—|—|—|")
    lines += ['', f'扫描候选：{expected}。', f'两次候选/两次基线合并吞吐：{ratio:.4f}。' if ratio else '没有完整有效重复对照。',
              '选择扫描最高吞吐只是工程线索；一轮独立复测不能排除选择偏差和共享主机噪声。',
              '半重计算通常占更多显存，必须同时检查余量，不只看tps。',
              '40步探针不保证长期质量，正式配置未改，退出自动释放自身worker。']
    (OUT / 'report.zh.md').write_text('\n'.join(lines) + '\n')
    print(report['status'], flush=True)
    if not passed: raise RuntimeError('diagnostic gate failed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preflight', 'prerequisite', 'kernel', 'prepare', 'worker', 'select', 'summarize', 'finish'])
    p.add_argument('--variant'); a = p.parse_args()
    if a.mode == 'preflight':
        import json
        print(json.dumps(preflight()), flush=True)
    elif a.mode == 'prerequisite': prerequisite_identity()
    elif a.mode == 'kernel': support.family.kernel_check()
    elif a.mode == 'prepare': prepare()
    elif a.mode == 'worker':
        if not a.variant: p.error('worker requires --variant')
        worker(a.variant)
    elif a.mode == 'select': engine.select()
    elif a.mode == 'summarize': summarize()
    else:
        if (OUT / 'PREPARED.json').exists() and read(OUT / 'PREPARED.json')['optimization_prerequisite'] != prerequisite_identity():
            raise ValueError('prerequisite changed before cleanup receipt')
        support.finish()


if __name__ == '__main__': main()
