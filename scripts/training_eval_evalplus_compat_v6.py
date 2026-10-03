"""Fail-closed, worker-local accounting repair for the frozen EvalPlus 0.3.1.

Its find_zero special oracle returns via `continue` without marking a passing
input or advancing progress. No polynomial oracle, tolerance, input, timeout,
or generic-task comparison is changed. Installed packages stay untouched.
"""
import hashlib
import importlib.metadata
from pathlib import Path

ORIGINAL_SHA256 = '76857b678cddca08dcaf54d7927b9a77826715cf657654ca5baf6c2b267f4c34'
OLD = '''                        if "find_zero" == entry_point:
                            assert abs(_poly(*inp, out)) <= atol
                            continue'''
NEW = '''                        if "find_zero" == entry_point:
                            assert abs(_poly(*inp, out)) <= atol
                            details[i] = True
                            progress.value += 1
                            continue'''


def patched_source(raw, expected_sha256=ORIGINAL_SHA256):
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('EvalPlus evaluator identity changed; refuse compatibility patch')
    source = raw.decode('utf-8').replace('\r\n', '\n')
    if source.count(OLD) != 1:
        raise ValueError('expected find_zero accounting branch not unique')
    return source.replace(OLD, NEW, 1)


def install():
    import evalplus.eval as module
    if importlib.metadata.version('evalplus') != '0.3.1':
        raise ValueError('compatibility repair is only for frozen EvalPlus 0.3.1')
    if getattr(module, '_training_quality_v6_accounting_patch', False):
        raise RuntimeError('accounting patch already installed')
    source_path = Path(module.__file__)
    source = patched_source(source_path.read_bytes())
    # Compile just unsafe_execute, in the official module's global namespace.
    # untrusted_check and check_correctness, including all limits, remain native.
    import ast
    tree = ast.parse(source, filename=str(source_path))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'unsafe_execute']
    if len(functions) != 1:
        raise ValueError('unsafe_execute definition not unique')
    exec(compile(ast.Module(body=functions, type_ignores=[]),
                 str(source_path) + ':v6_accounting', 'exec'), module.__dict__)
    module._training_quality_v6_accounting_patch = True
    return {'original_sha256': ORIGINAL_SHA256,
            'patched_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'scope': 'find_zero_pass_detail_and_progress_accounting_only',
            'installed_dependency_modified': False,
            'oracle_tolerance_inputs_limits_unchanged': True}
