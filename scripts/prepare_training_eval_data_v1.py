#!/usr/bin/env python3
"""Remote-only, checksum-bound benchmark cache; no training data writes."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import time
from training_eval_common_v1 import sha, write_json

PIQA_REV = '142f6d7367fd9877f0fb3b5734ea6a545f54cdd1'
GSM_REV = '740312add88f781978c0658806c59bc2815b9866'


def download(url, dest, headers=None):
    import requests
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            with requests.get(url, timeout=(15, 120), stream=True, headers=headers) as r:
                r.raise_for_status()
                with dest.with_suffix('.partial').open('wb') as f:
                    for block in r.iter_content(1024*1024):
                        f.write(block)
            dest.with_suffix('.partial').rename(dest)
            return
        except Exception:
            dest.with_suffix('.partial').unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--existing', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / 'manifest.json').exists():
        m = json.loads((a.output / 'manifest.json').read_text())
        for item in m['files']:
            if sha(a.output / item['file']) != item['sha256']:
                raise ValueError('existing benchmark data changed')
        print('verified existing benchmark cache', flush=True)
        return
    provenance = []
    for name, revision in [('ai2_arc', '210d026faf9955653af8916fad021475a3f00453'), ('hellaswag', '218ec52e09a7e7462a5400043bb9a69a41d06b76')]:
        src = a.existing / name
        if not src.is_dir():
            raise FileNotFoundError(src)
        shutil.copytree(src, a.output/name, dirs_exist_ok=True)
        provenance.append({'dataset':name, 'revision': revision, 'method':'reuse_existing_pinned_parquet'})
    for repo, rev, mapping in [
        ('baber/piqa', PIQA_REV, {f'piqa_{s}.parquet':f'piqa/{s}.parquet' for s in ('train','validation')}),
        ('openai/gsm8k', GSM_REV, {f'main/{s}-00000-of-00001.parquet':f'gsm8k/{s}.parquet' for s in ('train','test')})]:
        for rel, dest in mapping.items():
            download(f'https://hf-mirror.com/datasets/{repo}/resolve/{rev}/{rel}', a.output/dest)
        provenance.append({'dataset':repo, 'revision':rev, 'transport':'hf-mirror.com'})
    for name, version in [('HumanEvalPlus','v0.1.10'), ('MbppPlus','v0.2.0')]:
        url = f'https://github.com/evalplus/{name.lower()}_release/releases/download/{version}/{name}.jsonl.gz'
        gz = a.output/'evalplus'/f'{name}-{version}.jsonl.gz'
        try:
            download(url, gz)
        except Exception:
            asset = 165460216 if name=='HumanEvalPlus' else 162592602
            download(f'https://api.github.com/repos/evalplus/{name.lower()}_release/releases/assets/{asset}?download=1', gz, headers={'Accept':'application/octet-stream'})
        dest = gz.with_suffix('')
        dest.write_bytes(gzip.decompress(gz.read_bytes()))
        for line in dest.read_text().splitlines():
            json.loads(line)
        provenance.append({'dataset':name, 'version':version, 'url':url})
    write_json(a.output/'manifest.json', {'version':'training_eval_benchmark_data_v1', 'training_eligible':False, 'provenance':provenance, 'files':[{'file':str(x.relative_to(a.output)), 'bytes':x.stat().st_size, 'sha256':sha(x)} for x in sorted(a.output.rglob('*')) if x.is_file()]})
    print('benchmark cache ready', flush=True)


if __name__ == '__main__': main()
