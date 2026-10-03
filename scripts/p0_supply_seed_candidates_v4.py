#!/usr/bin/env python3
"""Bounded offline replenishment probe from existing files, not a READY release.

Full selected shards are scanned before hash-ranked sampling. Source approval,
near dedup, benchmark isolation and packed-loader checks remain pending.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import heapq
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from small_agent.curation.supply_control import file_sha,digest

TEXT_FIELDS={'finemath_4plus_new':'text','github_code_clean_new':'code',
             'ultradata_code_l2_py':'content','ultrafineweb_zh':'content','mdn_en_full_direct_v1':'text'}
DOMAINS={'finemath_4plus_new':'math','github_code_clean_new':'code',
         'ultradata_code_l2_py':'code','ultrafineweb_zh':'general_zh','mdn_en_full_direct_v1':'technical'}
LICENSES={'mit','apache-2.0','bsd-2-clause','bsd-3-clause','isc','0bsd','unlicense','cc0-1.0'}
VENDOR={'vendor','node_modules','third_party','third-party','site-packages','dist-packages','.venv','venv','build','dist'}

def normalize(text):return text.replace('\r\n','\n').replace('\r','\n').strip()

def reasons(source,row,text):
    flags=[];code=DOMAINS[source]=='code'
    if not isinstance(text,str) or not text: return ['empty_text']
    if not (160 if code else 300) <= len(text) <= 100000: flags.append('length_outside_probe_scope')
    if text.count('\ufffd')/max(1,len(text)) > .001:flags.append('broken_encoding')
    if '\x00' in text:flags.append('nul_character')
    if source=='github_code_clean_new':
        if str(row.get('license','')).lower() not in LICENSES:flags.append('license_not_whitelisted')
        if row.get('language') not in {'Python','JavaScript','TypeScript','Shell','Go','Rust','C','C++','Java'}:
            flags.append('language_outside_probe_scope')
    if code:
        path=str(row.get('path',row.get('relative_path',''))).replace('\\','/')
        if VENDOR & {p.casefold() for p in path.split('/')[:-1]}:flags.append('vendor_or_build_path')
        lines=[l for l in text.splitlines() if l.strip()]
        if len(lines)<4 or max(map(len,lines),default=0)>5000:flags.append('flattened_or_minified')
        if (source=='ultradata_code_l2_py' or row.get('language')=='Python') and len(text)<=100000:
            try:ast.parse(text)
            except (SyntaxError,ValueError,RecursionError,MemoryError):flags.append('python_parse_failure')
    if source=='finemath_4plus_new' and (int(row.get('int_score',0))<4 or row.get('language')!='en'):
        flags.append('upstream_math_score_or_language')
    return flags

def offer(heap,rank,record,k):
    # Identity is a deterministic tie-breaker; do not compare dicts on rank ties.
    item=(-int(rank,16),record['document_id'],record)
    if len(heap)<k:heapq.heappush(heap,item)
    elif item[:2]>heap[0][:2]:heapq.heapreplace(heap,item)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inventory',type=Path,required=True);ap.add_argument('--database',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-documents-per-source',type=int,default=2048)
    ap.add_argument('--token-cap-per-source',type=int,default=5000000);ap.add_argument('--seed',default='supply-v4-20261002')
    a=ap.parse_args()
    if a.max_documents_per_source<=0 or a.token_cap_per_source<=0:raise ValueError('positive probe bounds required')
    if a.output.exists():raise FileExistsError(a.output)
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    x=json.loads(a.inventory.read_text());root=Path(x['root'])
    toksha=file_sha(a.tokenizer)
    db=sqlite3.connect('file:'+str(a.database.resolve())+'?mode=ro',uri=True)
    try:
        if db.execute('SELECT COUNT(*) FROM legacy_imports').fetchone()[0]!=1:
            raise ValueError('audited legacy inventory must be imported before replenishment')
        old={r[0] for r in db.execute('SELECT body_sha FROM documents')}
    finally:db.close()
    expected='9d0b99ab9cf20d337217c96a5b39e9ff87590eb462abcbcc8f1119c3d43911a7'
    if toksha!=expected:raise ValueError('frozen tokenizer mismatch')
    tokenizer=Tokenizer.from_file(str(a.tokenizer));tokenizer.no_padding();tokenizer.no_truncation()
    sources={g['source']:g for g in x['raw_groups'] if g['source'] in TEXT_FIELDS}
    mdn=root/'mdn_full_wave4_v1/normalized/mdn_en_full_candidates_v1.jsonl'
    mp=root/'mdn_full_wave4_v1/audit/mdn_en_full_candidates_v1.manifest.json'
    m=json.loads(mp.read_text())
    if file_sha(mdn)!=m['output_sha256']:raise ValueError('MDN candidate binding changed')
    sources['mdn_en_full_direct_v1']={'files':[{'path':str(mdn),'bytes':mdn.stat().st_size,
        'revision':m['source_revision'],'expected_sha256':m['output_sha256']}],'root':'MDN'}
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.seed-',dir=a.output.parent))
    summaries={};input_files=[];seen=set()
    try:
        for source in sorted(sources):
            stats=Counter();heap=[];group=sources[source];revisions=set()
            for f in sorted(group['files'],key=lambda f:f['path']):
                path=Path(f['path']);path=path if path.is_absolute() else root/path
                if path.stat().st_size!=f['bytes']:raise ValueError('raw file size changed')
                before=file_sha(path)
                if not f.get('revision'):raise ValueError('source revision missing')
                if f.get('expected_sha256') and before!=f['expected_sha256']:raise ValueError('raw source hash changed')
                revisions.add(f['revision']);input_files.append({'path':str(path),'sha256':before,'source_id':source,'revision':f['revision']})
                if path.suffix=='.jsonl':
                    def rows():
                        with path.open() as handle:
                            for line in handle:yield json.loads(line)
                else:
                    pf=pq.ParquetFile(path)
                    # Read only adapter fields, excluding embeddings and large unrelated metadata.
                    needed={TEXT_FIELDS[source],'url','repo_name','path','relative_path','language','license','score',
                            'int_score','source','quality_score','algo_rel_score','category','source_revision'}
                    columns=[c for c in pf.schema_arrow.names if c in needed]
                    def rows():
                        for b in pf.iter_batches(batch_size=512,columns=columns):yield from b.to_pylist()
                for i,row in enumerate(rows()):
                    stats['raw_documents_scanned']+=1
                    raw=row.get(TEXT_FIELDS[source]);text=normalize(raw) if isinstance(raw,str) else ''
                    flags=reasons(source,row,text)
                    if flags:
                        stats['rule_dropped_documents']+=1;stats.update('drop:'+r for r in flags);continue
                    sha=hashlib.sha256(text.encode()).hexdigest()
                    rawsha=hashlib.sha256(raw.encode()).hexdigest()
                    if sha in old or rawsha in old:stats['exact_overlap_legacy']+=1;continue
                    if sha in seen:stats['exact_duplicate_in_probe']+=1;continue
                    seen.add(sha);stats['rule_pass_unique_candidates']+=1
                    docid=digest({'source':source,'revision':f['revision'],'body_sha256':sha})
                    url=row.get('url');repo=row.get('repo_name');relative=row.get('path',row.get('relative_path'))
                    locator=(url if url else f'repo={repo}|path={relative}' if repo else f'shard={path}|row={i}')
                    # Provisional family only; explicitly NOT claimed compatible with old URL keys.
                    family_basis=f'repo={repo}' if repo else str(url) if url else None
                    if family_basis: family=digest({'family_basis':family_basis})
                    else:family=None
                    record={'document_id':docid,'source_id':source,'source_revision':f['revision'],
                        'source_file':str(path),'raw_row':i,'source_locator':locator,
                        'domain':DOMAINS[source],'text':text,'normalized_sha256':sha,'family_id':family,
                        'family_status':'provisional_needs_legacy_compatible_mapping' if family else 'missing_original_family_hold',
                        'split':'candidate_unassigned','training_approved':False,
                        'policy_status':'await_source_calibration_and_policy','upstream_score':str(row.get('quality_score',row.get('score','')))}
                    rank=hashlib.sha256((a.seed+docid).encode()).hexdigest();record['sampling_rank']=rank
                    offer(heap,rank,record,a.max_documents_per_source)
                if file_sha(path)!=before:raise ValueError('raw file changed during scan')
            selected=sorted((h[2] for h in heap),key=lambda r:r['sampling_rank']);kept=[];tokens=0
            for start in range(0,len(selected),128):
                chunk=selected[start:start+128];encoded=tokenizer.encode_batch([r['text'] for r in chunk],add_special_tokens=False)
                hit_cap=False
                for row,enc in zip(chunk,encoded):
                    n=len(enc.ids)
                    if tokens+n>a.token_cap_per_source:hit_cap=True;break
                    row['own_text_tokens']=n;kept.append(row);tokens+=n
                if hit_cap:break
            if not kept:raise ValueError(f'no candidate records: {source}')
            out=tmp/(source+'.parquet');pq.write_table(pa.Table.from_pylist(kept),out,compression='zstd')
            summaries[source]={'domain':DOMAINS[source],'revisions':sorted(revisions),'scan':dict(stats),
                'sampled_documents':len(kept),'sampled_body_tokens':tokens,'output_file':out.name,'sha256':file_sha(out),
                'sampling':'top K SHA256(seed||document_id) after full selected-shard scan; token-cap prefix',
                'scope':'selected_existing_shards_only_not_entire_upstream_dataset'}
            print('PREPARED',source,json.dumps(summaries[source],ensure_ascii=False),flush=True)
        report={'stage':'offline_seed_candidates_not_training_approved','tokenizer_sha256':toksha,
            'inventory_sha256':file_sha(a.inventory),'sources':summaries,'input_files':input_files,'seed':a.seed,
            'new_ready_train_tokens':0,'new_api_calls':0,'new_gpu_tasks':0,'new_downloads':0,
            'general_en_replenishment_enabled':False,
            'pending':['source_policy','quality_calibration','legacy_compatible_family_split',
                       'cross_source_near_dedup','benchmark_isolation','pack_loader_acceptance'],
            'limits':['Tokenizer counts concern only the sampled candidate records, not the full scanned pool.',
                      'Rows are unassigned candidates, not train-only quota progress.']}
        (tmp/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');tmp.rename(a.output)
    except BaseException:
        import shutil
        shutil.rmtree(tmp,ignore_errors=True);raise

if __name__=='__main__':main()
