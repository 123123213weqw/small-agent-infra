#!/usr/bin/env python3
"""Frozen source snapshot -> deterministic bounded Python release jobs (not READY).

No API calls or training. Existing jobs, all ledger splits, known audit failures,
and undeclared licenses are excluded before paying. Publication is replayable.
The registered v5 release worker independently checks every release gate again.
"""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from small_agent.curation.supply_control import file_sha, digest
from small_agent.curation.release_gates_v5 import sha, write
from p0_source_audit_v4 import snapshot_files
from p0_release_queue_v5 import validate_job, validate_source_cohort


def eligible_reason(row, policy, scope, known, families):
    if sha(row['text']) != row['normalized_sha256']:
        raise ValueError('Candidate body hash differs')
    if row['source_id'] != scope['source_id']:
        return 'uncalibrated_source_hold'
    n = row['own_text_tokens']
    layer = ('short' if n <= 512 else 'medium' if n <= 4096 else 'long') + '/python'
    if row['language'].lower() != 'python' or layer not in scope['approved_strata']:
        return 'uncalibrated_language_or_length_hold'
    if row['policy_bucket'] != 'record_license_whitelisted' or row['record_license'].lower() not in policy['licenses']:
        return 'license_hold'
    if row['split'] not in ('train', 'validation', 'test'):
        return 'unassigned_split_hold'
    if row['normalized_sha256'] in known:
        return 'known_audit_bad_hold'
    if row['family_id'] in families and families[row['family_id']] != row['split']:
        return 'released_family_split_hold'
    return None


def partition(train, validation, test, target):
    """All records keep their original split; no held-out examples are promoted."""
    if target < 4096:
        raise ValueError('Batch target too small')
    chunks = []; current = []; tokens = 0
    for r in train:
        current.append(r); tokens += r['own_text_tokens']
        if tokens >= target:
            chunks.append(current); current = []; tokens = 0
    if current:
        # Merge a small final tail, so its strata are not tested with tiny samples.
        if chunks and tokens < target / 2:
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    if not chunks:
        return []
    if len(validation) < len(chunks) or len(test) < len(chunks):
        raise ValueError('Insufficient independent held-out documents; cannot manufacture validation')
    for records in (validation, test):
        for i, r in enumerate(records):
            chunks[i % len(chunks)].append(r)
    return chunks


def publish_plan(plan, output, queue, release_root):
    for item in plan['jobs']:
        candidate = output / item['candidate']
        if file_sha(candidate) != item['candidate_sha256']:
            raise ValueError('Frozen queued candidate changed')
        job = dict(version=5, processor='python_internal_pretraining_v5',
                   source_id=plan['source_id'], candidates=str(candidate.resolve()),
                   candidate_sha256=item['candidate_sha256'], source_audit=plan['source_audit'],
                   output=str((release_root / item['release_name']).resolve()))
        if validate_job(job, release_root) or validate_source_cohort(job):
            raise ValueError('Candidate is outside registered calibration')
        write(queue / (item['release_name'] + '.json'), job)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('census-root', 'download-plan', 'database', 'source-audit', 'scope-certificate', 'output', 'queue', 'release-root'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--policy', type=Path, default=ROOT / 'configs/curation/p0_release_gates_v5.json')
    p.add_argument('--batch-train-tokens', type=int, default=8000000)
    p.add_argument('--seed', type=int, default=20261002)
    a = p.parse_args()
    for path in (a.output, a.queue, a.release_root):
        path.mkdir(parents=True, exist_ok=True)
    with (a.output / 'producer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = dict(source_audit_plan_sha256=file_sha(a.source_audit / 'SAMPLE_PLAN.json'),
                       scope_sha256=file_sha(a.scope_certificate), policy_sha256=file_sha(a.policy),
                       download_plan_sha256=file_sha(a.download_plan), seed=a.seed,
                       batch_train_tokens=a.batch_train_tokens, producer_sha256=file_sha(Path(__file__)))
        plan_path = a.output / 'PLAN.json'
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if plan['binding'] != binding:
                raise ValueError('Producer replay configuration changed')
            publish_plan(plan, a.output, a.queue, a.release_root)
            print(json.dumps(dict(stage='existing_plan_reverified', jobs=len(plan['jobs']), paid_calls=0)))
            return
        scope = json.loads(a.scope_certificate.read_text())
        if scope['status'] != 'passed' or scope['source_id'] != 'github_code_clean_supply_v4' or scope['approved_language'] != 'python':
            raise ValueError('Missing registered Python scope evidence')
        policy = json.loads(a.policy.read_text())
        files, _ = snapshot_files(a.census_root, json.loads(a.download_plan.read_text()), scope['source_id'], False)
        binding['normalized_inputs'] = [{k: r[k] for k in ('source_file', 'source_sha256', 'output_relative_path', 'output_sha256')} for r in files]
        db = sqlite3.connect('file:' + str(a.database.resolve()) + '?mode=ro', uri=True)
        try:
            db.execute('BEGIN')
            seen = {r[0] for r in db.execute('SELECT body_sha FROM documents')}
            families = dict(db.execute('SELECT family_id,split FROM legacy_families'))
        finally:
            db.close()
        import pyarrow as pa
        import pyarrow.parquet as pq
        queued = set()
        # Even held/failed paid batches are excluded; never re-bill them by a new ID.
        for path in sorted(a.queue.glob('*.json')):
            j = json.loads(path.read_text()); candidate = Path(j['candidates'])
            if file_sha(candidate) != j['candidate_sha256']:
                raise ValueError('Existing queue input changed')
            for b in pq.ParquetFile(candidate).iter_batches(columns=['normalized_sha256']):
                queued.update(b.column(0).to_pylist())
        known = set(scope['known_audit_bad_body_sha256']); rejected = Counter(); unique = {}
        priority = {'test': 0, 'validation': 1, 'train': 2}
        for rec in files:
            path = a.census_root / rec['output_relative_path']
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                for r in batch.to_pylist():
                    reason = eligible_reason(r, policy, scope, known, families)
                    h = r['normalized_sha256']
                    if not reason and h in seen: reason = 'exact_released_overlap'
                    if not reason and h in queued: reason = 'already_queued_no_paid_replay'
                    if reason: rejected[reason] += 1; continue
                    old = unique.get(h)
                    if old:
                        rejected['candidate_exact_duplicate'] += 1
                        # Prefer held-out and deterministic locator tie break.
                        if (priority[r['split']], r['source_locator']) >= (priority[old['split']], old['source_locator']):
                            continue
                    unique[h] = r
            if file_sha(path) != rec['output_sha256']:
                raise ValueError('Candidate shard changed during selection')
            print('SCANNED_SOURCE', rec['source_file'], len(unique), flush=True)
        family_splits = {}
        bad_families = set()
        for r in unique.values():
            if family_splits.setdefault(r['family_id'], r['split']) != r['split']:
                bad_families.add(r['family_id'])
        records = [r for r in unique.values() if r['family_id'] not in bad_families]
        rejected['candidate_family_conflict_hold'] = len(unique) - len(records)
        records.sort(key=lambda r: sha(str(a.seed) + '|' + r['normalized_sha256']))
        chunks = partition(*[[r for r in records if r['split'] == s] for s in ('train', 'validation', 'test')], a.batch_train_tokens)
        jobs = []; counts = Counter()
        for i, rows in enumerate(chunks):
            train_strata = Counter(('short' if r['own_text_tokens'] <= 512 else 'medium') for r in rows if r['split'] == 'train')
            if any(n < 8 for n in train_strata.values()):
                raise ValueError('Too few train examples for the fixed rolling audit')
            candidate = a.output / f'part-{i:05d}.parquet'
            if candidate.exists():
                # Previous crash before PLAN: reproduce bytes, compare, do not overwrite.
                temp = candidate.with_suffix('.replay.parquet')
                pq.write_table(pa.Table.from_pylist(rows), temp, compression='zstd')
                if file_sha(temp) != file_sha(candidate):
                    temp.unlink(); raise ValueError('Interrupted candidate output differs')
                temp.unlink()
            else:
                temp = candidate.with_suffix('.tmp')
                pq.write_table(pa.Table.from_pylist(rows), temp, compression='zstd'); temp.replace(candidate)
            totals = Counter()
            for r in rows: totals[r['split']] += r['own_text_tokens']
            h = file_sha(candidate)
            jobs.append(dict(candidate=candidate.name, candidate_sha256=h, release_name=f'python_supply_{h[:20]}_v5',
                             candidate_documents=len(rows), candidate_body_tokens=dict(totals)))
            counts.update(totals)
        # Separate fixed call configuration from input snapshot for idempotent replay.
        source_inputs = binding.pop('normalized_inputs')
        plan = dict(version=5, stage='candidate_jobs_not_ready', source_id=scope['source_id'],
                    source_audit=str(a.source_audit.resolve()), binding=binding, normalized_inputs=source_inputs,
                    jobs=jobs, candidate_body_tokens=dict(counts), exclusions=dict(rejected),
                    registered_scope='Python short/medium only; not all code, mathematics or downloaded inventory',
                    paid_calls=0, new_ready_train_tokens=0, training_started=False)
        write(plan_path, plan)
        publish_plan(plan, a.output, a.queue, a.release_root)
        print(json.dumps(dict(stage=plan['stage'], jobs=len(jobs), candidate_body_tokens=dict(counts), paid_calls=0)))


if __name__ == '__main__':
    main()
