#!/usr/bin/env python3
"""Resumable full-shard candidate normalization/exact dedup/token census.

Never modifies the supply ledger, approves data, packs tokens, calls a model or
starts training. Splits are provisional until near-dedup/eval isolation passes.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time
from urllib.parse import urlsplit

from p0_data_pipeline import build_family_id, assign_split, normalize_text, sha256_file

VERSION = 'candidate_token_census_v4_20261002_1'
FROZEN_SHA = '9d0b99ab9cf20d337217c96a5b39e9ff87590eb462abcbcc8f1119c3d43911a7'
SOURCES = {'finemath_4plus_supply_v4': ('math', 'text'),
           'github_code_clean_supply_v4': ('code', 'code')}
LICENSES = {'mit','apache-2.0','bsd-2-clause','bsd-3-clause','isc','0bsd','unlicense','cc0-1.0'}
LANGUAGES = {'Python','JavaScript','TypeScript','Shell','Go','Rust','C','C++','Java'}
VENDOR = {'vendor','node_modules','third_party','third-party','site-packages','dist-packages','.venv','venv','build','dist'}


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.census-')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def prepare(source, row, max_chars):
    domain, field = SOURCES[source]
    raw = row.get(field)
    if not isinstance(raw, str): return None, ['missing_text']
    # Code is NOT stripped, NFC-normalized, dedented or whitespace-collapsed.
    text = raw.replace('\r\n','\n').replace('\r','\n')
    if domain != 'code': text = text.strip()
    flags = []
    if not text.strip(): flags.append('empty_text')
    if len(text) > max_chars: flags.append('oversize_hold')
    if '\x00' in text: flags.append('nul_character')
    if text.count('\ufffd') / max(1,len(text)) > .001: flags.append('broken_encoding')
    if domain == 'math':
        score = row.get('int_score')
        if not isinstance(score,(int,float)) or score < 4 or row.get('language') != 'en':
            flags.append('upstream_math_score_or_language')
    else:
        if row.get('language') not in LANGUAGES: flags.append('language_outside_recipe')
        path = str(row.get('path') or '').replace('\\','/')
        if VENDOR & {p.casefold() for p in path.split('/')[:-1]}: flags.append('vendor_or_build_path')
        lines = [l for l in text.splitlines() if l.strip()]
        if len(lines) < 2 or max(map(len,lines),default=0) > 5000:
            flags.append('flattened_or_minified')
    if flags: return None, flags
    license_value = str(row.get('license') or '').strip().lower()
    policy = ('record_license_whitelisted' if license_value in LICENSES else 'record_license_hold') if domain == 'code' else 'source_terms_hold'
    parse_status = 'not_python'
    if domain == 'code' and row.get('language') == 'Python':
        try: ast.parse(text); parse_status = 'parsed'
        except (SyntaxError,ValueError,RecursionError,MemoryError): parse_status = 'parse_failed_not_an_automatic_base_rejection'
    field_name = 'repo_name' if domain == 'code' else 'url'
    family_value = row.get(field_name)
    valid_family = isinstance(family_value,str) and bool(family_value.strip())
    if valid_family and field_name == 'url':
        try:
            u = urlsplit(family_value); valid_family = u.scheme in {'http','https'} and bool(u.hostname)
        except ValueError: valid_family = False
    if valid_family and field_name == 'repo_name':
        valid_family = bool(re.fullmatch(r'[^/\s]+/[^/\s]+',family_value))
    family = build_family_id(row,[field_name],source,'') if valid_family else None
    body = sha_text(text)
    # Check legacy normalization as an alias ONLY for old-inventory overlap.
    # Do not change stored code using that normalization.
    aliases = sorted({body,sha_text(raw),sha_text(normalize_text(raw))})
    locator = (str(row.get('repo_name') or '') + '/' + str(row.get('path') or '')) if domain == 'code' else str(row.get('url') or '')
    return {'domain':domain,'text':text,'normalized_sha256':body,'raw_sha256':sha_text(raw),
            'family_id':family,'source_locator':locator,'language':str(row.get('language') or ''),
            'record_license':license_value,'policy_bucket':policy,'python_parse_status':parse_status,
            'upstream_score':str(row.get('score') or ''),'legacy_aliases':aliases}, []


def split_for(db, family, seed):
    if family is None: return 'unassigned', 'missing_family_hold'
    key = bytes.fromhex(family)
    row = db.execute('SELECT split FROM legacy_families WHERE family=?',(key,)).fetchone()
    if row: return row[0], 'inherited_legacy_family_split'
    row = db.execute('SELECT split FROM families WHERE family=?',(key,)).fetchone()
    if row: return row[0], 'current_census_family_split'
    split = assign_split(family,seed,.98,.01)
    db.execute('INSERT INTO families VALUES (?,?)',(key,split))
    return split, 'hash_family_98_1_1'


def initialize(db, identity):
    db.executescript('''
      CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS legacy_bodies(body BLOB PRIMARY KEY);
      CREATE TABLE IF NOT EXISTS legacy_families(family BLOB PRIMARY KEY, split TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS families(family BLOB PRIMARY KEY, split TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS documents(body BLOB PRIMARY KEY, shard TEXT NOT NULL,
          domain TEXT NOT NULL, split TEXT NOT NULL, tokens INTEGER NOT NULL);
      CREATE TABLE IF NOT EXISTS shards(key TEXT PRIMARY KEY, receipt TEXT NOT NULL);
    ''')
    serialized = json.dumps(identity,sort_keys=True,separators=(',',':'))
    old = db.execute("SELECT v FROM meta WHERE k='identity'").fetchone()
    if old and old[0] != serialized: raise ValueError('Census identity changed; use a new output root')
    with db: db.execute("INSERT OR IGNORE INTO meta VALUES ('identity',?)",(serialized,))


def import_legacy(db, receipt_path):
    if db.execute("SELECT v FROM meta WHERE k='legacy_imported'").fetchone(): return
    receipt = json.loads(receipt_path.read_text())
    if receipt['stage'] != 'legacy_inventory_audit_internal_only' or receipt['tokenizer_sha256'] != FROZEN_SHA:
        raise ValueError('Legacy audit/tokenizer mismatch')
    index = Path(receipt['document_index_path'])
    item = next(x for x in receipt['evidence_files'] if x['path'] == str(index))
    if sha256_file(index) != item['sha256']: raise ValueError('Legacy index checksum mismatch')
    expected_docs = sum(receipt['documents_by_raw_domain_split'].values())
    count = 0
    with db:
        with index.open() as f:
            for line in f:
                r = json.loads(line); split = r['split']
                if split not in {'train','validation','test'}: raise ValueError('Bad legacy split')
                db.execute('INSERT INTO legacy_bodies VALUES (?)',(bytes.fromhex(r['body_sha256']),))
                if r.get('family_id'):
                    fam = bytes.fromhex(r['family_id'])
                    old = db.execute('SELECT split FROM legacy_families WHERE family=?',(fam,)).fetchone()
                    if old and old[0] != split: raise ValueError('Legacy family split conflict')
                    db.execute('INSERT OR IGNORE INTO legacy_families VALUES (?,?)',(fam,split))
                count += 1
        if count != expected_docs: raise ValueError('Legacy index row count mismatch')
        db.execute("INSERT INTO meta VALUES ('legacy_imported',?)",(str(count),))
    print('LEGACY_EXACT_INDEX',count,flush=True)


def reserve_body(db, record, shard, seed):
    for alias in record['legacy_aliases']:
        if db.execute('SELECT 1 FROM legacy_bodies WHERE body=?',(bytes.fromhex(alias),)).fetchone():
            return 'exact_overlap_legacy'
    body = bytes.fromhex(record['normalized_sha256'])
    if db.execute('SELECT 1 FROM documents WHERE body=?',(body,)).fetchone(): return 'exact_duplicate_census'
    split, strategy = split_for(db,record['family_id'],seed)
    record.update(split=split,split_strategy=strategy)
    db.execute('INSERT INTO documents VALUES (?,?,?,?,0)',(body,shard,record['domain'],split))
    return None


def schema(pa):
    strings = ['document_id','source_id','source_revision','source_file','source_locator','domain','language',
               'text','normalized_sha256','raw_sha256','family_id','split','split_strategy','record_license',
               'policy_bucket','python_parse_status','upstream_score']
    return pa.schema([(k,pa.string()) for k in strings] + [('raw_row',pa.int64()),('own_text_tokens',pa.int64()),
          ('training_approved',pa.bool_())])


def collect_totals(db):
    totals = Counter(); by_domain = {}; by_policy = {}; files = []
    for (text,) in db.execute('SELECT receipt FROM shards ORDER BY key'):
        r = json.loads(text); files.append(r); totals.update(r['counts'])
        by_domain.setdefault(r['domain'],Counter()).update(r['counts'])
        for k,v in r['policy_split_body_tokens'].items(): by_policy[k] = by_policy.get(k,0)+v
    return {'completed_shards':len(files),'counts':dict(totals),
            'by_domain':{k:dict(v) for k,v in by_domain.items()},'policy_split_body_tokens':by_policy,'files':files}


def validate_items(plan):
    items = plan['items']; seen = set(); root = Path(plan['allowed_root']).resolve()
    for r in items:
        if r['source_id'] not in SOURCES: raise ValueError('Unsupported source')
        if not re.fullmatch('[0-9a-f]{40}',r['revision']): raise ValueError('Unpinned revision')
        if not re.fullmatch('[0-9a-f]{64}',r['sha256']): raise ValueError('Invalid file SHA')
        p = Path(r['destination']).resolve()
        if p.suffix != '.parquet' or root not in p.parents: raise ValueError('Unsafe source path')
        if p in seen: raise ValueError('Duplicate physical source file')
        seen.add(p)
    # Code first for an early complete-domain count. No first-N source selection.
    return sorted(items,key=lambda r:(0 if SOURCES[r['source_id']][0]=='code' else 1,r['path']))


def process_shard(db, item, tokenizer, output, cfg, identity_sha, progress):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(item['destination']); start = time.monotonic()
    if path.stat().st_size != item['bytes'] or sha256_file(path) != item['sha256']:
        raise ValueError(f'Source file mismatch: {path}')
    before = path.stat(); key = sha_text(item['source_id']+'\0'+item['path']+'\0'+item['sha256'])
    final = output/'normalized'/item['source_id']/key
    completed = db.execute('SELECT receipt FROM shards WHERE key=?',(key,)).fetchone()
    if completed:
        r = json.loads(completed[0]); out = final/'part-00000.parquet'
        if sha256_file(out) != r['output_sha256'] or sha256_file(final/'excluded.jsonl') != r['excluded_sha256'] or json.loads((final/'receipt.json').read_text()) != r:
            raise ValueError('Completed shard output/receipt changed')
        print('REUSE_COUNTED',item['path'],flush=True); return
    # Rename-before-SQL-commit crash: remove ONLY this worker's identity-bound orphan.
    if final.exists():
        r = json.loads((final/'receipt.json').read_text())
        if r['identity_sha256'] != identity_sha or r['key'] != key:
            raise ValueError('Unowned orphan output; refusing overwrite')
        shutil.rmtree(final)
    final.parent.mkdir(parents=True,exist_ok=True)
    work = final.parent/('.'+key+'.partial')
    if work.exists():
        marker = json.loads((work/'owner.json').read_text())
        if marker != {'identity_sha256':identity_sha,'key':key}: raise ValueError('Unowned partial')
        shutil.rmtree(work)
    work.mkdir(); atomic_json(work/'owner.json',{'identity_sha256':identity_sha,'key':key})
    stats = Counter(); policy_counts = Counter(); pf = pq.ParquetFile(path)
    columns = [c for c in ('text','code','repo_name','path','url','license','language','score','int_score') if c in pf.schema_arrow.names]
    if SOURCES[item['source_id']][1] not in columns: raise ValueError('Adapter text field absent')
    writer = pq.ParquetWriter(work/'part-00000.parquet',schema(pa),compression='zstd')
    rejected = (work/'excluded.jsonl').open('w',encoding='utf-8')
    db.execute('BEGIN IMMEDIATE'); committed = False; closed = False
    try:
        row_idx = 0; last_log = 0; reserved_tokens = set(cfg['reserved_tokens'])
        for batch in pf.iter_batches(batch_size=cfg['batch_size'],columns=columns):
            records = []
            for row in batch.to_pylist():
                raw_row = row_idx; row_idx += 1; stats['raw_documents_scanned'] += 1
                record, flags = prepare(item['source_id'],row,cfg['max_chars'])
                if record and any(t in record['text'] for t in reserved_tokens):
                    record = None; flags = ['literal_reserved_token_hold']
                if flags:
                    stats['rule_excluded_documents'] += 1
                    stats.update('reason:'+f for f in flags)
                    raw = row.get(SOURCES[item['source_id']][1])
                    rejected.write(json.dumps({'raw_row':raw_row,'reason_codes':flags,
                        'raw_sha256':sha_text(raw) if isinstance(raw,str) else None,
                        'locator':str(row.get('url') or str(row.get('repo_name') or '')+'/'+str(row.get('path') or '')),
                        'record_license':str(row.get('license') or '')},ensure_ascii=False)+'\n')
                    continue
                stats['rule_pass_documents'] += 1
                reason = reserve_body(db,record,key,cfg['split_seed'])
                if reason:
                    stats[reason] += 1
                    rejected.write(json.dumps({'raw_row':raw_row,'reason_codes':[reason],
                        'normalized_sha256':record['normalized_sha256'],
                        'locator':record['source_locator']},ensure_ascii=False)+'\n')
                    continue
                record.pop('legacy_aliases')
                record.update(document_id=sha_text('\0'.join((item['source_id'],item['revision'],record['source_locator'],record['normalized_sha256']))),
                    source_id=item['source_id'],source_revision=item['revision'],source_file=item['path'],raw_row=raw_row,
                    training_approved=False)
                records.append(record)
            if records:
                enc = tokenizer.encode_batch([r['text'] for r in records],add_special_tokens=False)
                for i,(r,e) in enumerate(zip(records,enc)):
                    ids = e.ids; n = len(ids)
                    if n <= 0 or min(ids) < 15 or max(ids) >= 32768: raise ValueError('Unexpected body token IDs')
                    # Decode sampled records per batch, and every record in --full-roundtrip pilot.
                    if cfg['full_roundtrip'] or i < 2:
                        if tokenizer.decode(ids,skip_special_tokens=False) != r['text']: raise ValueError('Roundtrip mismatch')
                        stats['roundtrip_checked_documents'] += 1
                    r['own_text_tokens'] = n
                    db.execute('UPDATE documents SET tokens=? WHERE body=?',(n,bytes.fromhex(r['normalized_sha256'])))
                    stats['candidate_documents'] += 1; stats['candidate_body_tokens'] += n
                    stats['documents:'+r['split']] += 1; stats['body_tokens:'+r['split']] += n
                    stats['python:'+r['python_parse_status']] += 1
                    policy_counts[r['policy_bucket']+'/'+r['split']] += n
                writer.write_table(pa.Table.from_pylist(records,schema=schema(pa)))
            if time.monotonic()-last_log >= 30:
                progress(item,key,dict(stats),time.monotonic()-start)
                print('COUNTING',item['source_id'],item['path'],json.dumps(dict(stats)),flush=True)
                last_log = time.monotonic()
        writer.close(); closed = True
        rejected.flush(); os.fsync(rejected.fileno()); rejected.close()
        after = path.stat()
        if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns) or sha256_file(path) != item['sha256']:
            raise ValueError('Source changed during counting')
        if row_idx != pf.metadata.num_rows: raise ValueError('Source footer/scan count mismatch')
        if pq.ParquetFile(work/'part-00000.parquet').metadata.num_rows != stats['candidate_documents']:
            raise ValueError('Output row count mismatch')
        if stats['raw_documents_scanned'] != stats['rule_excluded_documents']+stats['candidate_documents']+stats['exact_overlap_legacy']+stats['exact_duplicate_census']:
            raise ValueError('Document accounting mismatch')
        r = {'key':key,'identity_sha256':identity_sha,'domain':SOURCES[item['source_id']][0],
            'source_id':item['source_id'],'source_file':item['path'],'source_sha256':item['sha256'],
            'output_relative_path':str(final.relative_to(output)/'part-00000.parquet'),
            'output_sha256':sha256_file(work/'part-00000.parquet'),
            'excluded_relative_path':str(final.relative_to(output)/'excluded.jsonl'),
            'excluded_sha256':sha256_file(work/'excluded.jsonl'),'counts':dict(stats),
            'policy_split_body_tokens':dict(policy_counts),'elapsed_s':time.monotonic()-start,
            'training_approved':False}
        atomic_json(work/'receipt.json',r); work.rename(final)
        db.execute('INSERT INTO shards VALUES (?,?)',(key,json.dumps(r,sort_keys=True)))
        db.commit(); committed = True
        print('COUNTED',item['source_id'],item['path'],json.dumps(r['counts']),flush=True)
    finally:
        if not closed: writer.close()
        if not rejected.closed: rejected.close()
        if not committed: db.rollback()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True); p.add_argument('--config',type=Path,required=True)
    p.add_argument('--legacy-receipt',type=Path,required=True); p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-shards',type=int,help='Bounded pilot; same identity, resumes into full run')
    a = p.parse_args()
    cfg = json.loads(a.config.read_text()); plan = json.loads(a.plan.read_text()); items = validate_items(plan)
    if a.max_shards is not None and a.max_shards <= 0: raise ValueError('Positive max shards required')
    if cfg['version'] != VERSION or cfg['tokenizer_sha256'] != FROZEN_SHA or cfg['vocab_size'] != 32768 or cfg['eos_token_id'] != 2:
        raise ValueError('Frozen tokenizer/config mismatch')
    if sha256_file(a.tokenizer) != FROZEN_SHA: raise ValueError('Tokenizer file mismatch')
    os.environ['RAYON_NUM_THREADS'] = str(cfg['tokenizer_threads'])
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(a.tokenizer)); tokenizer.no_padding(); tokenizer.no_truncation()
    if tokenizer.get_vocab_size() != 32768 or tokenizer.token_to_id('<eos>') != 2: raise ValueError('Token ID mismatch')
    identity = {'version':VERSION,'script_sha256':sha256_file(Path(__file__)),
        'family_utility_sha256':sha256_file(Path(__file__).with_name('p0_data_pipeline.py')),
        'config_sha256':sha256_file(a.config),'plan_sha256':sha256_file(a.plan),
        'legacy_receipt_sha256':sha256_file(a.legacy_receipt),'tokenizer_sha256':FROZEN_SHA}
    identity_sha = sha_text(json.dumps(identity,sort_keys=True,separators=(',',':')))
    a.output.mkdir(parents=True,exist_ok=True)
    lock = (a.output/'worker.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    if shutil.disk_usage(a.output).free < cfg['minimum_disk_free_bytes']: raise ValueError('Low free disk')
    db = sqlite3.connect(a.output/'census.sqlite',timeout=60)
    db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL'); db.execute('PRAGMA cache_size=-131072')
    initialize(db,identity); start = time.monotonic()
    def status(item=None,key=None,current=None,elapsed=0,stage='counting'):
        totals = collect_totals(db); totals.pop('files')
        atomic_json(a.output/'status.json',{'stage':stage,'identity':identity,'identity_sha256':identity_sha,
            'planned_shards':len(items),**totals,'current_shard':None if not item else {
                'source_id':item['source_id'],'source_file':item['path'],'counts_uncommitted':current,
                'elapsed_s':elapsed},'training_approved':False,'new_ready_train_tokens':0})
    try:
        status(stage='building_legacy_exact_index'); import_legacy(db,a.legacy_receipt)
        processed = 0
        for item in items:
            key = sha_text(item['source_id']+'\0'+item['path']+'\0'+item['sha256'])
            was_complete = db.execute('SELECT 1 FROM shards WHERE key=?',(key,)).fetchone()
            if not was_complete and a.max_shards is not None and processed >= a.max_shards: break
            if shutil.disk_usage(a.output).free < cfg['minimum_disk_free_bytes']: raise ValueError('Low free disk')
            process_shard(db,item,tokenizer,a.output,cfg,identity_sha,status)
            if not was_complete: processed += 1
            status()
        totals = collect_totals(db); complete = totals['completed_shards'] == len(items)
        report = {'stage':'candidate_token_census_complete_not_ready' if complete else 'candidate_token_census_partial',
            'identity':identity,'identity_sha256':identity_sha,'planned_shards':len(items),**totals,
            'complete':complete,'training_approved':False,'new_ready_train_tokens':0,
            'eos_padding_added':False,'train_split_provisional':True,'split_policy':'98/1/1 family hash; existing legacy families inherit their old split; missing family held unassigned',
            'exact_dedup_scope':'all this census plus audited old inventory (raw, LF-normalized and legacy-normalized aliases)',
            'near_dedup_completed':False,'benchmark_screen_completed':False,
            'new_api_calls':0,'new_gpu_tasks':0,'elapsed_this_invocation_s':time.monotonic()-start,
            'pending':['source_terms_and_license_policy','source_quality_calibration','global_near_duplicate_clustering',
                'benchmark_contamination_check','final_family_cluster_split','pack_and_loader_acceptance'],
            'limits':['No tokenizer counts for rule-excluded or duplicate bodies; report is retained unique candidate body census, not raw total.',
                'Only 8 specified new code shards; not all 880 source shards or 6 older code shards.',
                '98/1/1 applies to families, NOT exact token proportions; heldout may still share near-duplicates or question variants pending clustering.',
                'Python AST failures are flags, not automatic deletion of useful Base snippets.',
                'Token roundtrip sampling is recorded, not full-corpus roundtrip proof.']}
        atomic_json(a.output/('report.json' if complete else 'pilot_report.json'),report)
        status(stage=report['stage']); db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        print('CENSUS_STATUS',json.dumps({k:report[k] for k in ('stage','completed_shards','counts','by_domain','new_ready_train_tokens')}),flush=True)
    except BaseException as exc:
        atomic_json(a.output/'error.json',{'stage':'failed_resume_same_identity','error_type':type(exc).__name__,
            'error':str(exc),'identity_sha256':identity_sha,'new_ready_train_tokens':0})
        raise
    finally:
        db.close(); lock.close()

if __name__ == '__main__': main()
