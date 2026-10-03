#!/usr/bin/env python3
"""No API, no production ledger writes: same real source, v11 vs v13 byte equivalence."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.supply_control import file_sha
from small_agent.curation.release_gates_v5 import write
from p0_parallel_cpu_v13 import encoded_documents,build_sketch


def main():
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from small_agent.data.sharded_pack import TokenizedDocument,pack_sharded
    from p0_verify_release_pack_v11 import verify as verify11
    from p0_verify_release_pack_v13 import verify as verify13
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path)
    p.add_argument('--tokenizer',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--workers',type=int,nargs='+',default=[1,8]);a=p.parse_args()
    if a.output.exists():raise ValueError('Benchmark output already exists; do not overwrite receipts')
    a.output.mkdir(parents=True)
    source=a.run/'deduplicated.parquet';source_sha=file_sha(source);tokenizer_sha=file_sha(a.tokenizer)
    original=json.loads((a.run/'tokenized/manifest.json').read_text());tok=Tokenizer.from_file(str(a.tokenizer))
    # Test exact physical equivalence against an EMPTY disposable ledger, never against production.
    database=a.output/'shadow.sqlite'
    with sqlite3.connect(database) as db:
        db.executescript('CREATE TABLE documents(body_sha TEXT,split TEXT);CREATE TABLE legacy_families(family_id TEXT,split TEXT);')
    results=[];baseline_manifest=None;baseline_verify=None;baseline_sketch=None
    def docs():
        for b in pq.ParquetFile(source).iter_batches(batch_size=64):
            for r in b.to_pylist():
                ids=tok.encode(r['text'],add_special_tokens=False).ids
                if len(ids)!=r['own_text_tokens'] or tok.decode(ids,skip_special_tokens=False)!=r['text']:raise ValueError('Original roundtrip failed')
                yield TokenizedDocument(r['normalized_sha256'],r['split'],r['domain'],ids)
    for workers in a.workers:
        run=a.output/('workers-'+str(workers));run.mkdir()
        for name in ('deduplicated.parquet','documents.jsonl','FROZEN.json'):os.link(a.run/name,run/name)
        t=time.monotonic()
        manifest=pack_sharded(docs() if workers==1 else encoded_documents(source,a.tokenizer,workers),run/'tokenized',
            sequence_length=original['sequence_length'],target_shard_tokens=8000000,vocab_size=32768,
            eos_token_id=2,tokenizer_sha256=tokenizer_sha,mixture=original['mixture'],
            license_gate=original['license_gate'],source_manifest_sha256=file_sha(run/'FROZEN.json'))
        pack_seconds=time.monotonic()-t
        if baseline_manifest is None:baseline_manifest=manifest
        if manifest!=baseline_manifest:raise ValueError('Parallel manifest/bin/idx hashes differ')
        t=time.monotonic();v=verify11(run,a.tokenizer,database) if workers==1 else verify13(run,a.tokenizer,database,workers)
        verify_seconds=time.monotonic()-t
        semantic={k:x for k,x in v.items() if k not in ('checker_sha256','parallel_engine_sha256')}
        if baseline_verify is None:baseline_verify=semantic
        if semantic!=baseline_verify:raise ValueError('Full source/loader/sampler validation differs')
        t=time.monotonic();m=build_sketch(source,dict(path=str(source),sha256=source_sha),run/'sketch',workers)
        sketch_seconds=time.monotonic()-t
        sketch=pq.read_table(m['path']).to_pylist()
        if baseline_sketch is None:baseline_sketch=sketch
        if sketch!=baseline_sketch:raise ValueError('Parallel sketch rows differ')
        tokens=sum(x['body_tokens'] for x in [json.loads(l) for l in (run/'documents.jsonl').read_text().splitlines()])
        measured=dict(workers=workers,pack_seconds=pack_seconds,verify_seconds=verify_seconds,
                      sketch_seconds=sketch_seconds,total_seconds=pack_seconds+verify_seconds+sketch_seconds,
                      body_tokens=tokens,documents=sum(v['documents_by_split'].values()),
                      manifest_bin_idx_identical=True,all_document_checks_identical=True,all_sketches_identical=True)
        results.append(measured);print(json.dumps(measured),flush=True)
    if file_sha(source)!=source_sha or file_sha(a.tokenizer)!=tokenizer_sha:raise ValueError('Benchmark input changed')
    for r in results:r['combined_speedup_vs_serial']=results[0]['total_seconds']/r['total_seconds']
    write(a.output/'RESULT.json',dict(version=13,stage='MEASURED_CPU_EQUIVALENCE_NOT_CORPUS_APPROVAL',
        source_sha256=source_sha,tokenizer_sha256=tokenizer_sha,results=results,api_requests=0,
        production_ledger_written=False,benchmark_script_sha256=file_sha(Path(__file__))))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
