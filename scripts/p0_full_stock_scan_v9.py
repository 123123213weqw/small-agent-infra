#!/usr/bin/env python3
"""Resume a CPU-only ALL-source triage into metadata sidecars, no corpus body copy."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from small_agent.curation.full_stock_v9 import route, length_layer
from small_agent.curation.release_gates_v5 import sha, write
from small_agent.curation.supply_control import file_sha, digest
from p0_stock_plan_v6 import open_plan
from p0_curation_worker_v3 import atomic_status


def scan(original, output, config_path):
    import pyarrow.parquet as pq
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(config_path.read_text()); registration = original/'STOCK_REGISTERED.json'
    identity = dict(version=9, original_registration_sha256=file_sha(registration), config_sha256=file_sha(config_path),
                    scanner_sha256=file_sha(Path(__file__)), router_sha256=file_sha(ROOT/'src/small_agent/curation/full_stock_v9.py'),
                    existing_v8_filter_sha256=file_sha(ROOT/'src/small_agent/curation/math_filter_v7.py'))
    canonical = sqlite3.connect('file:'+str((original/'stock.sqlite').resolve())+'?mode=ro', uri=True)
    sources = canonical.execute('SELECT id,path,sha,source_id,revision FROM sources WHERE complete=1 ORDER BY id').fetchall()
    expected = json.loads(registration.read_text())
    if len(sources) != expected['sources']:
        raise ValueError('Full original stock registration required')
    # Independent metadata DB, NOT a path alias for an existing live WAL database.
    db = open_plan(output)
    db.executescript('CREATE TABLE IF NOT EXISTS routing(h TEXT PRIMARY KEY,state TEXT,reason TEXT,bucket TEXT,stratum TEXT);'
                     'CREATE INDEX IF NOT EXISTS routing_bucket ON routing(bucket,state);'
                     'CREATE TABLE IF NOT EXISTS scan_progress(sid INTEGER PRIMARY KEY,next_pos INTEGER,stats TEXT);')
    old = db.execute("SELECT v FROM meta WHERE k='full_stock_identity'").fetchone()
    if old and json.loads(old[0]) != identity:
        raise ValueError('Full-stock version changed; explicit new output required')
    with db: db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', ('full_stock_identity', json.dumps(identity, sort_keys=True)))
    try:
        for sid, path, h, source_id, revision in sources:
            path = Path(path); prev = db.execute('SELECT sha,complete FROM sources WHERE id=?', (sid,)).fetchone()
            if prev and prev != (h, 1) and prev[0] != h:
                raise ValueError('Registered source changed')
            if prev and prev[1]: continue
            if file_sha(path) != h: raise ValueError('Frozen source bytes changed')
            saved = db.execute('SELECT next_pos,stats FROM scan_progress WHERE sid=?', (sid,)).fetchone()
            next_pos, counts = saved[0] if saved else 0, Counter(json.loads(saved[1])) if saved else Counter()
            with db: db.execute('INSERT OR IGNORE INTO sources VALUES (?,?,?,?,?,0)', (sid, str(path), h, source_id, revision))
            pos = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
                rows, routes = [], []
                for r in batch.to_pylist():
                    at = pos; pos += 1
                    if at < next_pos: continue
                    if r['source_id'] != source_id or r['source_revision'] != revision or sha(r['text']) != r['normalized_sha256']:
                        raise ValueError('Source/body provenance changed')
                    state, reason, bucket = route(r, cfg)
                    rows.append((r['normalized_sha256'], sid, at, r['split'], r['language'].lower(), r['domain'], r['own_text_tokens'], r['family_id'], r['record_license'], r['policy_bucket'], bytes.fromhex(sha('20261002|'+r['normalized_sha256'])), state, None))
                    routes.append((r['normalized_sha256'], state, reason, bucket, length_layer(r['own_text_tokens'], r['language'])))
                    counts['documents'] += 1; counts[state+'_documents'] += 1
                    counts['body_tokens/'+r['split']] += r['own_text_tokens']
                # Rows, routing and restart cursor advance in the SAME transaction.
                with db:
                    db.executemany('INSERT OR IGNORE INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)
                    db.executemany('INSERT OR IGNORE INTO routing VALUES (?,?,?,?,?)', routes)
                    db.execute('INSERT OR REPLACE INTO scan_progress VALUES (?,?,?)', (sid, pos, json.dumps(dict(counts), sort_keys=True)))
                if rows and pos % 8192 < 512:
                    atomic_status(output/'SCAN_STATUS.json', dict(stage='SCANNING_ALL_STOCK_NOT_READY', source_id=sid, next_pos=pos, completed_sources=db.execute('SELECT COUNT(*) FROM sources WHERE complete=1').fetchone()[0], total_sources=len(sources), training_started=False))
            if file_sha(path) != h: raise ValueError('Source bytes changed during scan')
            with db: db.execute('UPDATE sources SET complete=1 WHERE id=?', (sid,))
            print('FULL_STOCK_SOURCE_TRIAGED', sid, dict(counts), flush=True)
        actual = db.execute('SELECT COUNT(*),SUM(CASE WHEN split=\'train\' THEN n ELSE 0 END) FROM records').fetchone()
        if actual != (expected['documents'], expected['candidate_train_body_tokens']):
            raise ValueError('ALL-stock coverage differs, no READY or calibration allowed')
        summary = [dict(domain=d, language=l, bucket=b, state=s, split=sp, documents=c, body_tokens=n) for d,l,b,s,sp,c,n in db.execute('SELECT r.domain,r.language,x.bucket,x.state,r.split,COUNT(*),SUM(r.n) FROM records r JOIN routing x USING(h) GROUP BY r.domain,r.language,x.bucket,x.state,r.split')]
        receipt = dict(version=9, stage='ALL_STOCK_RULE_TRIAGE_COMPLETE_NOT_READY', identity=identity,
                       sources=len(sources), documents=actual[0], candidate_train_body_tokens=actual[1], routes=summary,
                       original_stock=str(original.resolve()), rule_filter_config=str(config_path.resolve()),
                       evidence_files=[dict(path=str(config_path.resolve()),sha256=identity['config_sha256']),dict(path=str((ROOT/'src/small_agent/curation/full_stock_v9.py').resolve()),sha256=identity['router_sha256']),dict(path=str((ROOT/'src/small_agent/curation/math_filter_v7.py').resolve()),sha256=identity['existing_v8_filter_sha256'])],
                       training_eligible=False,distribution_eligible=False)
        write(output/'SCAN_COMPLETE.json', receipt)
        write(output/'STOCK_REGISTERED.json', {**expected, 'full_stock_v9_scan_receipt_sha256':file_sha(output/'SCAN_COMPLETE.json')})
        atomic_status(output/'SCAN_STATUS.json', dict(stage=receipt['stage'], completed_sources=len(sources), total_sources=len(sources), candidate_train_body_tokens=actual[1],new_ready_tokens=0,training_started=False))
        return receipt
    finally:
        canonical.close(); db.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('original','output','config'): p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'scan.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        r=scan(a.original,a.output,a.config)
        print(json.dumps({k:v for k,v in r.items() if k!='routes'},ensure_ascii=False))
if __name__=='__main__':main()
