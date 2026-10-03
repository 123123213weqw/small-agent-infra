#!/usr/bin/env python3
"""Observe GPU contention; stop only the direct parent in our benchmark cgroup."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import time
import benchmark_perf8_v1 as bench


def cgroup(pid):
    return (Path('/proc') / str(pid) / 'cgroup').read_text()


def validate_parent(parent, actual_parent, own_group, parent_group):
    if parent <= 1 or parent != actual_parent or own_group != parent_group:
        raise ValueError('guard can only signal its direct parent in the same cgroup')
    if 'p0-wave2-perf8-v1' not in own_group:
        raise ValueError('guard must run in the dedicated benchmark systemd unit')


def outside_processes(processes, own_group, get_group=cgroup):
    outside = []
    for process in processes:
        try:
            group = get_group(process['pid'])
        except FileNotFoundError:
            continue  # Process exited between snapshot and lookup.
        if group != own_group:
            outside.append(dict(process, cgroup=group))
    return outside


def run(parent):
    own_group = cgroup(os.getpid())
    validate_parent(parent, os.getppid(), own_group, cgroup(parent))
    bench.write(bench.OUT / 'guard.started.json', {
        'parent_pid': parent, 'guard_pid': os.getpid(), 'cgroup': own_group,
        'interval_seconds': 2, 'started_at': datetime.now(timezone.utc).isoformat(),
        'note': 'Observation is not a resource reservation; never signals another job.'})
    checks = 0
    while os.getppid() == parent:
        snapshot = bench.gpu_snapshot()
        outside = outside_processes(snapshot['processes'], own_group)
        checks += 1
        bench.write(bench.OUT / 'guard.latest.json', {
            'checks': checks, 'captured_at': snapshot['captured_at'],
            'outside_processes': outside, 'gpu_snapshot': snapshot})
        if outside:
            bench.write(bench.OUT / 'CONTENTION.json', {
                'status': 'external_gpu_contention', 'captured_at': snapshot['captured_at'],
                'outside_processes': outside, 'action': 'signal only own runner parent with SIGTERM'})
            # Revalidate immediately; do not signal a reused or unrelated PID.
            validate_parent(parent, os.getppid(), own_group, cgroup(parent))
            os.kill(parent, signal.SIGTERM)
            return
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    try:
        run(args.parent_pid)
    except Exception as error:
        bench.write(bench.OUT / 'GUARD_ERROR.json', {'error': repr(error), 'status': 'guard_failed_closed'})
        own_group = cgroup(os.getpid())
        validate_parent(args.parent_pid, os.getppid(), own_group, cgroup(args.parent_pid))
        os.kill(args.parent_pid, signal.SIGTERM)
        raise


if __name__ == '__main__':
    main()
