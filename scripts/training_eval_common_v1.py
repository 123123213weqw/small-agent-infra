"""Pure, CPU-testable protocol helpers for training quality eval v1."""
from __future__ import annotations
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else (x.tolist() if hasattr(x, 'tolist') else str(x)))
        f.write('\n')
    os.replace(f.name, path)


def fixed_subset(refs, maximum):
    """Evenly spaced, immutable refs; all records when the domain is tiny."""
    refs = list(refs)
    if maximum is None or len(refs) <= maximum:
        return refs
    if maximum < 1:
        raise ValueError('maximum must be positive')
    return [refs[(i * len(refs)) // maximum] for i in range(maximum)]


def wilson(successes, total):
    if not 0 <= successes <= total or total <= 0:
        raise ValueError('invalid count')
    p, z = successes / total, 1.959963984540054
    d = 1 + z*z / total
    c = (p + z*z/(2*total)) / d
    h = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / d
    return [max(0., c-h), min(1., c+h)]


def weighted_loss(domains):
    n = sum(x['predicted_tokens'] for x in domains.values())
    if n <= 0:
        raise ValueError('no validation targets')
    l = sum(x['loss'] * x['predicted_tokens'] for x in domains.values()) / n
    if not math.isfinite(l):
        raise ValueError('nonfinite validation')
    return {'loss': l, 'ppl': math.exp(l), 'predicted_tokens': n}


def due_checks(step, final_step, config):
    if step <= 0:
        raise ValueError('checkpoint step must be positive')
    out = ['generation']
    if step == final_step:
        return out + ['validation_full', 'abilities_full', 'code_full']
    if step % config['fast_interval_steps'] == 0:
        out.append('validation_quick')
    if step % config['ability_interval_steps'] == 0:
        out += ['abilities_pilot', 'code_pilot']
    return out


def verify_inputs(run, checkpoint, tokenizer_dir):
    run, checkpoint, tokenizer_dir = map(Path, (run, checkpoint, tokenizer_dir))
    rm = json.loads((run / 'run_manifest.json').read_text())
    cm = json.loads((checkpoint / 'manifest.json').read_text())
    for key in ('run_config', 'model_spec', 'data_manifest'):
        if sha(rm[key]['path']) != rm[key]['sha256']:
            raise ValueError(f'{key} changed')
        if cm['compatibility'][key + '_sha256'] != rm[key]['sha256']:
            raise ValueError(f'checkpoint {key} mismatch')
    mp = checkpoint / 'model.safetensors'
    art = cm['artifacts'][mp.name]
    if mp.stat().st_size != art['bytes'] or sha(mp) != art['sha256']:
        raise ValueError('checkpoint weights changed')
    tm = json.loads((tokenizer_dir / 'manifest.json').read_text())
    dm = json.loads(Path(rm['data_manifest']['path']).read_text())
    ts = sha(tokenizer_dir / 'tokenizer.json')
    if tm['stage'] != 'frozen_tokenizer' or ts != tm['artifacts']['tokenizer.json']['sha256'] or ts != dm['tokenizer_sha256']:
        raise ValueError('tokenizer identity differs from training')
    return rm, cm, {
        'run_manifest_sha256': sha(run / 'run_manifest.json'),
        'checkpoint_manifest_sha256': sha(checkpoint / 'manifest.json'),
        'checkpoint_model_sha256': art['sha256'], 'tokenizer_sha256': ts,
        'data_manifest_sha256': rm['data_manifest']['sha256'],
        'model_spec_sha256': rm['model_spec']['sha256'],
        'run_config_sha256': rm['run_config']['sha256'],
    }
