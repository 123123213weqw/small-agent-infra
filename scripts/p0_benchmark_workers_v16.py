#!/usr/bin/env python3
"""Read-only 8/16-worker AB on one immutable production-ledger snapshot.

Uses the unchanged v13 near engine deployed by the v15 publisher. No API or
production ledger writes. A concurrent production commit cannot invalidate this
benchmark once its SQLite backup matches the bound index manifest.
"""
import argparse
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from p0_parallel_near_v13 import ParallelNearIndex
from small_agent.curation.release_gates_v5 import ledger_snapshot, write
from small_agent.curation.supply_control import digest, file_sha


def main():
    import pyarrow.parquet as pq
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('candidates', 'inventory-index', 'database', 'policy', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--documents', type=int, default=1024)
    a = p.parse_args()
    if a.documents < 512:
        raise ValueError('At least 16 chunks required for 16 workers')
    if a.output.exists():
        raise ValueError('Benchmark receipt already exists')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = a.output.with_suffix('.ledger.sqlite')
    if snapshot.exists():
        raise ValueError('Snapshot already exists; inspect rather than overwrite')
    ref = json.loads(a.inventory_index.read_text())
    source = sqlite3.connect('file:' + str(a.database.resolve()) + '?mode=ro', uri=True)
    destination = sqlite3.connect(snapshot)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    if ledger_snapshot(snapshot) != ref['ledger']:
        raise ValueError('Backup does not match index; no benchmark acceptance')
    rows = []
    for batch in pq.ParquetFile(a.candidates).iter_batches(batch_size=64, use_threads=False):
        rows.extend(batch.to_pylist())
        if len(rows) >= a.documents:
            break
    rows = rows[:a.documents]
    if len(rows) < 512:
        raise ValueError('Insufficient candidate rows')
    policy = json.loads(a.policy.read_text())['near']
    candidate_sha = file_sha(a.candidates)
    manifest_sha = file_sha(a.inventory_index)
    results = []
    baseline = None
    for workers in (8, 16, 16, 8):
        index = ParallelNearIndex(Path(ref['database']).parent)
        started = time.monotonic()
        try:
            value = index.filter(rows, a.inventory_index, policy, snapshot, workers)
        finally:
            index.close()
        elapsed = time.monotonic() - started
        if baseline is None:
            baseline = value
        if baseline != value:
            raise ValueError('Kept rows or complete near report changed')
        item = dict(workers=workers, seconds=elapsed, retained_documents=len(value[0]),
                    excluded_documents=value[1]['excluded_documents'],
                    exact_jaccard_pairs=value[1]['exact_jaccard_pairs'],
                    identical_full_report=True, full_result_sha256=digest(value))
        results.append(item)
        print(json.dumps(item), flush=True)
    if file_sha(a.candidates) != candidate_sha or file_sha(a.inventory_index) != manifest_sha:
        raise ValueError('Frozen benchmark input changed')
    medians = {str(w): statistics.median(r['seconds'] for r in results if r['workers'] == w)
               for w in (8, 16)}
    write(a.output, dict(version=16, stage='READ_ONLY_8_16_WORKERS_AB',
                        documents=len(rows), results=results, median_seconds=medians,
                        speedup_16_over_8=medians['8'] / medians['16'],
                        inventory_snapshot=ref['ledger'], candidate_sha256=candidate_sha,
                        index_manifest_sha256=manifest_sha, policy_sha256=file_sha(a.policy),
                        benchmark_ledger_sha256=file_sha(snapshot),
                        production_ledger_written=False, api_requests=0,
                        engine_sha256=file_sha(ROOT / 'scripts/p0_parallel_near_v13.py'),
                        quality_thresholds_changed=False))


if __name__ == '__main__':
    main()
