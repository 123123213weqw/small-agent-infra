#!/usr/bin/env python3
"""Parallel sketch preparation; unchanged v6 coverage checks and single index writer."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from p0_parallel_cpu_v13 import build_sketch
from small_agent.curation.release_gates_v5 import inventory_files
from p0_batched_index_v18 import BatchedDiskNearIndex as DiskNearIndex


def main():
    p = argparse.ArgumentParser()
    for name in ('database', 'legacy-receipt', 'sketch-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(Path(os.environ['P0_PARALLEL_CONFIG']).read_text())
    with (a.database.parent / 'production-release.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Only new registered immutable text files; never recompute the historical corpus.
        indexed = {}
        if (a.output / 'postings.sqlite').exists():
            with sqlite3.connect('file:' + str(a.output / 'postings.sqlite') + '?mode=ro', uri=True) as db:
                indexed = {r[0]: r[1:] for r in db.execute('SELECT path,sha,complete FROM sources')}
        for entry in inventory_files(a.database, a.legacy_receipt):
            old = indexed.get(entry['path'])
            if old and old != (entry['sha256'], 1):
                if old[0] != entry['sha256']: raise ValueError('Append-only reference source changed')
            if not old or not old[1]:
                build_sketch(Path(entry['path']), entry, a.sketch_root, cfg['release_cpu_workers'])
        idx = DiskNearIndex(a.output)
        try: path = idx.update(a.database, a.legacy_receipt, a.sketch_root)
        finally: idx.close()
    print(json.dumps(dict(stage='complete_registered_disk_index', manifest=str(path),
                          parallel_sketch_engine='v13_batched_index_v18', training_started=False)))


if __name__ == '__main__': main()
