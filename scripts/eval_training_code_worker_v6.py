#!/usr/bin/env python3
"""CPU worker: ONLY run in the namespace sandbox, never on the host."""
import json
import os
from pathlib import Path


def main():
    if os.environ.get('TRAINING_EVAL_SANDBOX') != '1':
        raise RuntimeError('sandbox required')
    mode = os.environ.get('SANDBOX_MODE')
    if mode == 'probe':
        from eval_training_code_worker_v1 import main as original_probe
        return original_probe()
    import importlib.metadata
    if importlib.metadata.version('evalplus') != '0.3.1':
        raise RuntimeError('unlocked evalplus')
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    from evalplus.evaluate import check_correctness
    from evalplus.gen.util import trusted_exec
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.sanitize import sanitize
    from training_eval_evalplus_compat_v6 import install

    def groundtruth(problem, special):
        code = problem['prompt'] + problem['canonical_solution']
        truth = {}
        for part in ('base', 'plus'):
            truth[part], truth[part + '_time'] = trusted_exec(
                code, problem[part + '_input'], problem['entry_point'],
                record_time=True, output_not_none=problem['entry_point'] in special)
        return code, truth

    def summary(row):
        return {p: {'status': row[p][0], 'details': len(row[p][1]),
                    'passed_cases': sum(row[p][1])} for p in ('base', 'plus')}

    if mode == 'regression':
        problem = get_human_eval_plus()['HumanEval/32']
        canonical, truth = groundtruth(problem, [])
        legacy = check_correctness('humaneval', 0, problem, canonical, truth, fast_check=False)
        if any(legacy[p][0] != 'fail' or len(legacy[p][1]) != 0 for p in ('base', 'plus')):
            raise RuntimeError('legacy accounting failure was not reproduced')
        policy = install()
        fixed = check_correctness('humaneval', 0, problem, canonical, truth, fast_check=False)
        if any(fixed[p][0] != 'pass' or len(fixed[p][1]) != len(problem[p + '_input'])
               or not all(fixed[p][1]) for p in ('base', 'plus')):
            raise RuntimeError('full official HumanEval/32 canonical regression failed')
        # Distinct legitimate roots must pass the special oracle, not be forced
        # to equal the canonical root. A wrong root must still fail.
        toy = dict(problem, base_input=[[[-6, 11, -6, 1]]], plus_input=[[[-6, 11, -6, 1]]])
        toy_truth = {'base': [1.0], 'plus': [1.0], 'base_time': [0.001], 'plus_time': [0.001]}
        valid = check_correctness('humaneval', 0, toy, 'def find_zero(xs): return 3.0', toy_truth, fast_check=False)
        invalid = check_correctness('humaneval', 0, toy, 'def find_zero(xs): return 0.0', toy_truth, fast_check=False)
        generic = dict(toy, entry_point='identity', atol=0, base_input=[[2]], plus_input=[[3]])
        generic_truth = dict(toy_truth, base=[2], plus=[3])
        generic_good = check_correctness('humaneval', 0, generic, 'def identity(x): return x', generic_truth, fast_check=False)
        generic_bad = check_correctness('humaneval', 0, generic, 'def identity(x): return x+1', generic_truth, fast_check=False)
        for result, expected in [(valid, 'pass'), (invalid, 'fail'),
                                 (generic_good, 'pass'), (generic_bad, 'fail')]:
            if any(result[p][0] != expected or len(result[p][1]) != 1 for p in ('base', 'plus')):
                raise RuntimeError('special-oracle or generic regression failed')
        Path('/output/result.json').write_text(json.dumps({
            'passed': True, 'legacy': summary(legacy), 'fixed': summary(fixed),
            'alternative_valid_root': summary(valid), 'invalid_root': summary(invalid),
            'generic_valid': summary(generic_good), 'generic_invalid': summary(generic_bad),
            'policy': policy}))
        return
    if mode != 'score':
        raise ValueError('unsupported sandbox mode')
    policy = install()
    req = json.loads(Path('/input/request.json').read_text())
    dataset, tid = req['dataset'], req['task_id']
    if dataset not in ('humaneval', 'mbpp'):
        raise ValueError('unknown dataset')
    problems = get_human_eval_plus() if dataset == 'humaneval' else get_mbpp_plus()
    problem = problems[tid]
    special = [] if dataset == 'humaneval' else MBPP_OUTPUT_NOT_NONE_TASKS
    canonical, truth = groundtruth(problem, special)
    sanity = check_correctness(dataset, 0, problem, canonical, truth, fast_check=False)
    if any(sanity[p][0] != 'pass' for p in ('base', 'plus')):
        Path('/output/canonical_failure.json').write_text(json.dumps(
            {'dataset': dataset, 'task_id': tid, 'sanity': summary(sanity)}))
        raise RuntimeError('canonical solution did not pass official tests')
    solution = sanitize(req['solution'], entrypoint=problem['entry_point'])
    result = check_correctness(dataset, 0, problem, solution, truth, fast_check=False)
    report = {'dataset': dataset, 'task_id': tid, 'canonical_sanity_pass': True,
              'base_status': result['base'][0], 'plus_status': result['plus'][0],
              'base_pass': result['base'][0] == 'pass',
              'plus_pass': result['base'][0] == 'pass' and result['plus'][0] == 'pass',
              'sanitized_changed': solution != req['solution'],
              'base_cases': len(problem['base_input']), 'plus_cases': len(problem['plus_input']),
              'evaluator_compatibility': policy}
    Path('/output/result.json').write_text(json.dumps(report))


if __name__ == '__main__':
    main()
