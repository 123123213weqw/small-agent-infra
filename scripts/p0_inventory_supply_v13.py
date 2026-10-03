#!/usr/bin/env python3
"""Opt-in CPU acceleration adapter. Frozen v11 QA, allocation and budget remain unchanged."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import p0_inventory_supply_v11 as original


def publisher_for(arguments):
    output = Path(arguments[arguments.index('--output') + 1])
    binding = output / 'INPUT_BINDING.json'
    if binding.exists():
        # Never migrate an in-flight/committed v11 release to another engine.
        prior = json.loads(binding.read_text())
        names = {Path(p).name for p in prior['engine_sha256']}
        return 'p0_release_inventory_batch_v13.py' if 'p0_release_inventory_batch_v13.py' in names else 'p0_release_inventory_batch_v11.py'
    return 'p0_release_inventory_batch_v13.py'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--performance-config', type=Path, required=True)
    a = p.parse_args()
    perf = json.loads(a.performance_config.read_text())
    if type(perf['release_cpu_workers']) is not int or not 1 <= perf['release_cpu_workers'] <= 16: raise ValueError('Invalid release workers')
    os.environ['P0_PARALLEL_CONFIG'] = str(a.performance_config.resolve())
    cfg = json.loads(a.config.read_text()); cfg['config_path'] = str(a.config.resolve())
    # No changes to target tokens, audit density, quality policy or API worker.
    component = original.component
    def dispatch(script, args):
        if script == 'p0_build_inventory_index_v6.py': script = 'p0_build_inventory_index_v13.py'
        elif script == 'p0_release_inventory_batch_v11.py': script = publisher_for(args)
        return component(script, args)
    original.component = dispatch
    Path(cfg['root']).mkdir(parents=True, exist_ok=True)
    with (Path(cfg['root']) / 'queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            original.run(cfg)
        except BlockingIOError: print('EXISTING_QUEUE_NO_DUPLICATE_START')


if __name__ == '__main__': main()
