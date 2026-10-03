#!/usr/bin/env python3
"""CPU-only new attempt: reuse immutable v5 generations, rescore all code.

Never retries a failed v5 directory, never loads a model, never trains. A
finished measurement is separate from a passing gate; zero-floor measures
remain explicitly inconclusive.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from training_eval_common_v1 import sha, write_json


def read(path):
    return json.loads(Path(path).read_text())


def capability_pair(seed, left, right):
    if left.keys() != right.keys():
        raise ValueError('paired score keys differ')
    for key in left:
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (left[key], right[key])):
            raise ValueError('invalid/nonfinite capability score: ' + key)
    gaps = {key: right[key] - left[key] for key in left}
    generation = 'generation/repeated_4gram_fraction'
    if generation not in gaps:
        raise ValueError('missing generation comparison')
    floor = [key for key in left if ('pass@1' in key or key.startswith('gsm8k/'))
             and left[key] == 0 and right[key] == 0]
    passed = all(value >= -.03 for key, value in gaps.items()
                 if key != generation and key not in floor) and gaps[generation] <= .05
    return {'seed': seed, 'baseline_scores': left, 'candidate_scores': right,
            'score_gaps': gaps, 'diagnostic_gate_passed': passed,
            'floor_metrics_inconclusive': floor, 'floor_metrics_certified': False}


def scores(directory, protocol):
    out = {}
    for task in ('arc_easy', 'piqa', 'hellaswag', 'gsm8k'):
        report = read(directory / f'{task}_result.json')
        if len(report['results']) != 1:
            raise ValueError('unexpected task result count')
        row = next(iter(report['results'].values()))
        n = protocol['pilot']['gsm8k_limit' if task == 'gsm8k' else 'multiple_choice_limit']
        if row.get('sample_len') != n:
            raise ValueError('fixed capability sample count changed: ' + task)
        expected = ('exact_match,strict-match', 'exact_match,flexible-extract') if task == 'gsm8k' else ('acc,none', 'acc_norm,none')
        for metric in expected:
            out[f'{task}/{metric}'] = float(row[metric])
    code = read(directory / 'code_execution/result.json')
    if not code['complete'] or code['generation_identity_sha256'] != sha(directory / 'identity.json'):
        raise ValueError('code result incomplete/unbound')
    for task in ('humaneval', 'mbpp'):
        row = code['results'][task]
        generations = read(directory / f'{task}_generations.json')['records']
        if row['n'] != protocol['pilot']['code_limit_per_dataset'] or row['n'] != len(generations):
            raise ValueError('code sample count changed')
        if [r['task_id'] for r in row['records']] != [r['task_id'] for r in generations]:
            raise ValueError('code task order changed or samples discarded')
        if not all(r['canonical_sanity_pass'] for r in row['records']):
            raise ValueError('canonical sanity failed')
        for key, field in [('base_pass_at_1', 'base_pass'), ('plus_pass_at_1', 'plus_pass')]:
            if row[key] != sum(r[field] for r in row['records']) / row['n']:
                raise ValueError('incorrect code aggregate')
        out[f'{task}/base_pass@1'] = row['base_pass_at_1']
        out[f'{task}/plus_pass@1'] = row['plus_pass_at_1']
    records = read(directory / 'generation16.json')['records']
    if len(records) != 16:
        raise ValueError('generation panel changed')
    rates = [float(r['uncached']['metrics']['repeated_4gram_rate']) for r in records]
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in rates):
        raise ValueError('invalid generation rate')
    out['generation/repeated_4gram_fraction'] = sum(rates) / len(rates)
    return out


def prepare(a):
    root, parent = a.root, a.parent
    if (root / 'PREPARED.json').exists():
        raise FileExistsError('attempt already frozen')
    cfg = read(parent / 'config.json')
    from training_quality_campaign_v2 import production_identity
    identity = production_identity(cfg)
    if identity != read(parent / 'PREPARED.json')['production_identity']:
        raise ValueError('formal identity drift')
    if not read(parent / 'reports/paired_numeric_acceptance.json')['all_numeric_gates_passed']:
        raise ValueError('v5 numerical gates did not pass')
    if read(parent / 'reports/FINAL.json')['status'] != 'evaluation_incomplete':
        raise ValueError('parent does not have expected preserved failure receipt')
    config = {'parent': str(parent), 'base_root': str(a.base_root), 'seeds': cfg['seeds'],
              'purpose': 'cpu_only_evalplus_accounting_repair_reuse_identical_generations',
              'formal_training_started': False, 'automatic_adoption_allowed': False}
    write_json(root / 'config.json', config)
    parents, copies = {}, {}

    def track(path):
        parents[str(path)] = sha(path)

    for name in ('config.json', 'PREPARED.json', 'reports/conv_preflight.json',
                 'reports/resume_acceptance.json', 'reports/paired_numeric_acceptance.json',
                 'reports/FINAL.json'):
        track(parent / name)
    for name, digest in read(parent / 'PREPARED.json')['experiment_source_manifest'].items():
        path = parent / 'code' / name
        if sha(path) != digest:
            raise ValueError('v5 frozen source drift')
        parents[str(path)] = digest
    eval_config = parent / 'code/configs/eval/training_quality_v2.json'
    shutil.copy2(eval_config, root / 'eval_config.json'); track(eval_config)
    protocol = read(eval_config)
    if protocol['pilot'] != {'multiple_choice_limit': 256, 'gsm8k_limit': 64, 'code_limit_per_dataset': 64}:
        raise ValueError('unexpected parent pilot panel')
    for seed in cfg['seeds']:
        for variant in ('baseline', 'candidate'):
            case = f'seed{seed}_{variant}'; run = parent / 'runs' / case
            cp = Path(read(run / 'RESULT.json')['checkpoint'])
            track(run / 'RESULT.json'); track(run / 'run_manifest.json'); track(cp / 'manifest.json')
            source = run / 'evaluation/runs' / f'{case}_queue_step_{read(cp / "manifest.json")["step"]:08d}_scheduled'
            track(source / 'eval_config.json'); track(source / 'prompts.json')
            origin = source / 'abilities'; identity_row = read(origin / 'identity.json')
            if identity_row['config_sha256'] != sha(eval_config) or identity_row['prompt_sha256'] != sha(source / 'prompts.json'):
                raise ValueError('parent generation protocol drift')
            if identity_row['identity']['checkpoint_manifest_sha256'] != sha(cp / 'manifest.json'):
                raise ValueError('parent generation checkpoint unbound')
            if identity_row['identity']['checkpoint_model_sha256'] != read(run / 'RESULT.json')['model_sha256']:
                raise ValueError('endpoint model identity mismatch')
            if not read(origin / 'model_eval_complete.json')['complete']:
                raise ValueError('model generation incomplete')
            dest = root / 'runs' / case / 'abilities'; dest.mkdir(parents=True, exist_ok=False)
            for path in sorted(origin.glob('*.json')):
                track(path); shutil.copy2(path, dest / path.name)
                copies[str((dest / path.name).relative_to(root))] = sha(dest / path.name)
            for task in ('humaneval', 'mbpp'):
                generated = read(dest / f'{task}_generations.json')
                if not generated['complete'] or len(generated['records']) != 64:
                    raise ValueError('incomplete fixed code panel')
    for path in (root / 'data/manifest.json', root / 'data/evalplus/HumanEvalPlus-v0.1.10.jsonl',
                 root / 'data/evalplus/MbppPlus-v0.2.0.jsonl', root / 'deps/evalplus/eval/__init__.py',
                 root / 'sandbox/bwrap-package/usr/bin/bwrap'):
        track(path)
    for path in (root / 'config.json', root / 'eval_config.json', root / 'reports/cpu_tests.log',
                 root / 'reports/sandbox_regression.json'):
        copies[str(path.relative_to(root))] = sha(path)
    if not read(root / 'reports/sandbox_regression.json')['regression']['passed']:
        raise ValueError('real sandbox regression did not pass')
    source_hashes = {str(p.relative_to(root)): sha(p) for p in sorted((root / 'code').rglob('*.py'))
                     if '__pycache__' not in p.parts}
    source_hashes.update({str(p.relative_to(root)): sha(p) for p in sorted((root / 'deploy').glob('*')) if p.is_file()})
    write_json(root / 'PREPARED.json', {'production_identity': identity, 'parent_files': parents,
        'copied_inputs': copies, 'source_manifest': source_hashes, 'gpu_model_regeneration': False,
        'reused_v5_numeric_receipt': str(parent / 'reports/paired_numeric_acceptance.json'),
        'old_failure_preserved': True, 'formal_training_started': False, 'automatic_adoption_allowed': False})


def verify(root, prepared):
    for name, expected in prepared['parent_files'].items():
        if sha(name) != expected:
            raise ValueError('frozen parent changed: ' + name)
    for key in ('copied_inputs', 'source_manifest'):
        for name, expected in prepared[key].items():
            if sha(root / name) != expected:
                raise ValueError('frozen recovery input changed: ' + name)


def run(a):
    root = a.root; prepared = read(root / 'PREPARED.json'); verify(root, prepared)
    cfg = read(root / 'config.json'); parent = Path(cfg['parent']); protocol = read(root / 'eval_config.json')
    cases = [f'seed{seed}_{variant}' for seed in cfg['seeds'] for variant in ('baseline', 'candidate')]
    errors, completed = [], []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def evaluate(case):
        dest = root / 'runs' / case
        scope = f'p0-quality-v6-code-{case}.scope'
        command = ['systemd-run', '--user', '--scope', '--quiet', '--unit=' + scope, '--property=TasksMax=64',
                   '--property=MemoryMax=4G', '--property=CPUQuota=200%', sys.executable, '-u',
                   str(root / 'code/scripts/eval_training_code_sandbox_v6.py'),
                   '--eval-root', str(root), '--base-deps', str(Path(cfg['base_root']) / 'deps'),
                   '--data', str(root / 'data'), '--generations', str(dest / 'abilities'),
                   '--config', str(root / 'eval_config.json')]
        try:
            with (dest / 'code_execution.log').open('x') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200)
        except BaseException:
            # This version's named scopes only; never enumerate or kill others.
            subprocess.run(['systemctl', '--user', 'stop', scope], check=False)
            raise
        return case

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(evaluate, case): case for case in cases}
        for future in as_completed(futures):
            try:
                completed.append(future.result())
            except Exception as error:
                errors.append({'case': futures[future], 'error': repr(error)})
            write_json(root / 'reports/abilities_progress.json', {'completed': completed, 'errors': errors, 'total': 4})
    pairs = []
    if not errors:
        for seed in cfg['seeds']:
            left, right = [root / 'runs' / f'seed{seed}_{v}' / 'abilities' for v in ('baseline', 'candidate')]
            for task in ('humaneval', 'mbpp'):
                l = read(left / f'{task}_generations.json')['records']
                r = read(right / f'{task}_generations.json')['records']
                if [x['task_id'] for x in l] != [x['task_id'] for x in r]:
                    raise ValueError('paired generation task order differs')
            pairs.append(capability_pair(seed, scores(left, protocol), scores(right, protocol)))
    verify(root, prepared)
    from training_quality_campaign_v2 import production_identity
    unchanged = production_identity(read(parent / 'config.json')) == prepared['production_identity']
    numeric = read(parent / 'reports/paired_numeric_acceptance.json')
    complete = len(completed) == 4 and len(pairs) == 2 and not errors
    passed = complete and unchanged and numeric['all_numeric_gates_passed'] and all(p['diagnostic_gate_passed'] for p in pairs)
    write_json(root / 'reports/FINAL.json', {
        'status': 'all_measurements_completed' if complete else 'evaluation_incomplete',
        'all_decidable_gates_passed': passed, 'numeric': numeric, 'capability_pairs': pairs, 'errors': errors,
        'floor_metrics_inconclusive': any(p['floor_metrics_inconclusive'] for p in pairs),
        'floor_metrics_certified': False, 'production_identity_unchanged': unchanged,
        'parent_generation_files_exact': True, 'old_failure_preserved': True,
        'original_fast_convolution_bitwise_equivalence_certified': False,
        'original_dynamic_reducer_bitwise_equivalence_certified': False,
        'from_scratch_quality_certified': False, 'automatic_adoption_allowed': False,
        'formal_training_started': False, 'gpu_used_by_recovery': False})
    if not complete:
        raise RuntimeError('recovery evaluation incomplete; inspect new receipts, never retry in place')


def main():
    p = argparse.ArgumentParser()
    for name in ('root', 'parent', 'base-root'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--prepare', action='store_true'); a = p.parse_args()
    if a.prepare:
        prepare(a)
    else:
        run(a)


if __name__ == '__main__':
    main()
