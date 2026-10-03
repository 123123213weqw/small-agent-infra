#!/usr/bin/env python3
"""Freeze committed READY packs into one directly readable training dataset.

Same-domain packs are supported. Storage is hardlinked, not recopied. A training
run must pin the immutable generation path, never follow CURRENT while running.
Does not start training, change domain recipes, or include inherited legacy data.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from p0_ready_catalog_v5 import catalog
from small_agent.curation.supply_control import digest, file_sha
from small_agent.curation.release_gates_v5 import write
from small_agent.data.sharded_format import load_sharded_manifest

COMPATIBLE = ('version', 'dtype', 'index_dtype', 'sequence_length', 'stride',
              'vocab_size', 'eos_token_id', 'tokenizer_sha256')


def check_documents(entries):
    seen = set(); families = {}; totals = Counter(); by_domain = Counter()
    for e in entries:
        r = json.loads(Path(e['ready_receipt']).read_text())
        index = Path(r['document_index']['path'])
        if file_sha(index) != r['document_index']['sha256']:
            raise ValueError('Released document index changed')
        train = 0
        for line in index.read_text().splitlines():
            d = json.loads(line); h = d['body_sha256']; split = d['split']
            if h in seen:
                raise ValueError('Cross-pack exact duplicate, including held-out')
            seen.add(h)
            if families.setdefault(d['family_id'], split) != split:
                raise ValueError('Cross-pack family split conflict')
            totals[split] += d['body_tokens']
            if split == 'train':
                train += d['body_tokens']; by_domain[d['domain']] += d['body_tokens']
        if train != e['train_body_tokens']:
            raise ValueError('READY train counter differs from body index')
    return dict(totals), dict(by_domain), len(seen)


def verify_reader(root):
    import numpy as np
    from small_agent.data.sharded_dataset import ShardedTokenDataset
    from small_agent.data.mixture_sampler import DeterministicMixtureSchedule, MixtureDistributedSampler
    m, shards = load_sharded_manifest(root, verify_hashes=True)
    for s in shards:
        ids = np.memmap(s.bin_path, dtype='<u2', mode='r')
        for offset in range(0, s.tokens, 1000000):
            if int(ids[offset:offset + 1000000].max()) >= m['vocab_size']:
                raise ValueError('Out-of-vocabulary packed token')
    checks = {}; train = None
    for split in ('train', 'validation', 'test'):
        ds = ShardedTokenDataset(root, split, m['sequence_length'], verify_hashes=False)
        if not len(ds):
            raise ValueError('Missing held-out/train samples')
        for i in range(len(ds.shards)):
            for sample in {ds._prefix[i], ds._prefix[i + 1] - 1}:
                r = ds[sample]; n = int(r['valid_tokens'])
                if not (r['labels'][n:] == -100).all() or not (r['attention_mask'][:n] == 1).all():
                    raise ValueError('Merged loader masking differs')
        checks[split] = ds.coverage_report()
        if split == 'train': train = ds
    schedule = DeterministicMixtureSchedule(train, seed=20261002, block_size=len(train))
    for rank in (0, 1):
        sampler = MixtureDistributedSampler(schedule, num_replicas=2, rank=rank,
                    micro_batch_size=1, gradient_accumulation_steps=2)
        it = iter(sampler); prefetched = [next(it) for _ in range(6)]
        sampler.commit_step()
        restored = MixtureDistributedSampler(schedule, num_replicas=2, rank=rank,
                    micro_batch_size=1, gradient_accumulation_steps=2)
        restored.load_state_dict(sampler.state_dict())
        if next(iter(restored)) != prefetched[2]:
            raise ValueError('Merged sampler committed-state recovery differs')
    return dict(status='passed', physical_hashes_checked=True, all_tokens_in_vocabulary=True,
                first_and_last_sample_per_shard_checked=True,
                two_rank_committed_sampler_recovery=True, coverage=checks)


def build(database, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'bundle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = catalog(database)
        if not c['packs']:
            raise ValueError('No READY data; never fall back to candidates or legacy')
        identity = dict(catalog=c, assembler_sha256=file_sha(Path(__file__)))
        generation = output / digest(identity)
        if generation.exists():
            if json.loads((generation / 'CATALOG.json').read_text()) != c:
                raise ValueError('Frozen bundle identity differs')
            m, _ = load_sharded_manifest(generation, verify_hashes=True)
            receipt = json.loads((generation / 'TRAINING_READY.json').read_text())
            if receipt['manifest_sha256'] != file_sha(generation / 'manifest.json') or receipt['verify_sha256'] != file_sha(generation / 'VERIFY.json'):
                raise ValueError('Training bundle evidence changed')
            if (receipt['catalog_sha256'] != file_sha(generation / 'CATALOG.json')
                    or receipt['unique_train_body_tokens'] != c['unique_train_body_tokens']
                    or receipt['status'] != 'TRAINING_READY_INTERNAL'
                    or receipt['data_dir'] != str(generation.resolve())
                    or receipt['context_length'] != m['sequence_length']
                    or receipt['tokenizer_sha256'] != m['tokenizer_sha256']
                    or receipt['distribution_eligible'] is not False
                    or receipt['training_started'] is not False
                    or json.loads((generation / 'VERIFY.json').read_text())['status'] != 'passed'):
                raise ValueError('Training receipt binding differs')
        else:
            totals, domains, documents = check_documents(c['packs'])
            temp = Path(tempfile.mkdtemp(prefix='.building-', dir=output))
            try:
                shards = []; first = None; weights = Counter()
                for e in c['packs']:
                    root = Path(e['manifest']).parent
                    m, _ = load_sharded_manifest(root, verify_hashes=True)
                    if first is None: first = m
                    if any(m[k] != first[k] for k in COMPATIBLE):
                        raise ValueError('Incompatible tokenizer/context/disk format; cannot merge')
                    for old in m['shards']:
                        s = dict(old); sid = len(shards)
                        for kind in ('bin', 'idx'):
                            name = f"{s['split']}-{sid:05d}.{kind}"
                            os.link(root / old[kind], temp / name); s[kind] = name
                        s.update(shard_id=sid, source_batch_id=e['batch_id']); shards.append(s)
                        if s['split'] == 'train': weights[s['domain']] += s['samples']
                m = {k: first[k] for k in COMPATIBLE}
                m.update(stage='approved_sharded_token_stream',
                         license_gate=dict(training_eligible=True, distribution_eligible=False,
                                           scope='source_audited_internal_pretraining_only'),
                         source_manifest_sha256=digest(c), source_packs=c['packs'],
                         mixture=dict(weights), mixture_basis='train_sample_counts_one_pass_not_final_20b_recipe',
                         storage='hardlinks_to_immutable_ready_packs', shards=shards,
                         unique_body_tokens_by_split=totals, train_body_tokens_by_domain=domains)
                write(temp / 'manifest.json', m); write(temp / 'CATALOG.json', c)
                verification = verify_reader(temp); write(temp / 'VERIFY.json', verification)
                # Single final generation path is stable before trainer consumption.
                receipt = dict(status='TRAINING_READY_INTERNAL', data_dir=str(generation.resolve()),
                    manifest_sha256=file_sha(temp / 'manifest.json'), verify_sha256=file_sha(temp / 'VERIFY.json'),
                    catalog_sha256=file_sha(temp / 'CATALOG.json'), unique_train_body_tokens=c['unique_train_body_tokens'],
                    body_tokens_by_split=totals, train_body_tokens_by_domain=domains, documents=documents,
                    context_length=m['sequence_length'], data_reader='sharded_v1', tokenizer_sha256=m['tokenizer_sha256'],
                    mixture_block_size=sum(weights.values()), distribution_eligible=False,
                    scope='only committed v5 READY; no legacy imports, raw candidates, or failed batches',
                    trainer_binding={'data_dir': str(generation.resolve()), 'data_reader': 'sharded_v1',
                                     'context_length': m['sequence_length'], 'mixture_block_size': sum(weights.values())},
                    model_environment_gpu_preflight_required_before_launch=True, training_started=False)
                write(temp / 'TRAINING_READY.json', receipt)
                temp.rename(generation)
            except BaseException:
                shutil.rmtree(temp, ignore_errors=True); raise
        # Future queue updates replace the pointer, not this frozen training snapshot.
        pointer = output / '.CURRENT.tmp'; pointer.write_text(str(generation.resolve()) + '\n')
        pointer.replace(output / 'CURRENT')
        return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); print(json.dumps(build(a.database, a.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
