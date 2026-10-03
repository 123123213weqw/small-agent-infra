#!/usr/bin/env python3
"""Physical/source/loader verifier for eligible v5 packs; never certifies facts."""
from __future__ import annotations
import argparse
import os
from p0_parallel_cpu_v13 import CodecChecks
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from small_agent.curation.supply_control import file_sha


def verify(run,tokenizer,database,workers=8):
    with CodecChecks(tokenizer,workers) as checks:
        return _verify(run,tokenizer,database,checks)

def _verify(run,tokenizer,database,checks):
    import numpy as np
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from small_agent.data.sharded_format import load_sharded_manifest
    from small_agent.data.sharded_dataset import ShardedTokenDataset
    from small_agent.data.mixture_sampler import DeterministicMixtureSchedule,MixtureDistributedSampler
    frozen=json.loads((run/'FROZEN.json').read_text())
    if (frozen['parquet_sha256']!=file_sha(run/'deduplicated.parquet') or frozen['index_sha256']!=file_sha(run/'documents.jsonl')
            or frozen['tokenizer_sha256']!=file_sha(tokenizer) or frozen['training_eligible'] is not True):raise ValueError('Frozen identity changed')
    manifest,shards=load_sharded_manifest(run/'tokenized',verify_hashes=True)
    if (manifest['source_manifest_sha256']!=file_sha(run/'FROZEN.json') or manifest['tokenizer_sha256']!=file_sha(tokenizer)
            or manifest['license_gate']['training_eligible'] is not True or manifest['eos_token_id']!=2):raise ValueError('Pack/source gate mismatch')
    streams={s:np.concatenate([np.memmap(x.bin_path,dtype='<u2',mode='r') for x in shards if x.split==s]) for s in ('train','validation','test')}
    positions=Counter();docs=Counter();bodies=Counter();seen=set();families={}
    indices=[json.loads(l) for l in (run/'documents.jsonl').read_text().splitlines()];indexiter=iter(indices)
    ledger=sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True);ledger.execute('BEGIN')
    try:
        for batch in pq.ParquetFile(run/'deduplicated.parquet').iter_batches(batch_size=64):
            for r in batch.to_pylist():
                d=next(indexiter);sha=hashlib.sha256(r['text'].encode('utf-8')).hexdigest();split=r['split']
                if sha in seen or sha!=r['normalized_sha256'] or sha!=d['body_sha256']:raise ValueError('Duplicate or corrupt source')
                seen.add(sha)
                if d['split']!=split or d['family_id']!=r['family_id'] or d['domain']!=r['domain'] or d['body_tokens']!=r['own_text_tokens']:raise ValueError('Source/index disagreement')
                if ledger.execute('SELECT 1 FROM documents WHERE body_sha=?',(sha,)).fetchone():raise ValueError('Exact overlap with released inventory')
                old=ledger.execute('SELECT split FROM legacy_families WHERE family_id=?',(r['family_id'],)).fetchone()
                if old and old[0]!=split:raise ValueError('Legacy family split overlap')
                if families.setdefault(r['family_id'],split)!=split:raise ValueError('Local family split conflict')
                n=d['body_tokens'];start=positions[split];ids=streams[split][start:start+n].tolist()
                if start+n>=len(streams[split]) or int(streams[split][start+n])!=2 or 2 in ids:raise ValueError('EOS/document boundary mismatch')
                checks.add(r['text'],streams[split][start:start+n].tobytes())
                positions[split]+=n+1;docs[split]+=1;bodies[split]+=n
        if next(indexiter,None) is not None or len(indices)!=frozen['documents'] or dict(bodies)!=frozen['body_tokens']:raise ValueError('Frozen document counts differ')
        for split in streams:
            if positions[split]!=len(streams[split]) or sum(x['tokens'] for x in manifest['shards'] if x['split']==split)!=positions[split]:raise ValueError('Pack tail/count differs')
            if sum(x['documents'] for x in manifest['shards'] if x['split']==split)!=docs[split]:raise ValueError('Pack document counter differs')
    finally:ledger.close()
    if checks.finish()!=len(indices):raise ValueError('Full source verification coverage differs')
    samples=Counter()
    for split in streams:
        dataset=ShardedTokenDataset(run/'tokenized',split,manifest['sequence_length'],verify_hashes=True)
        for i in range(len(dataset)):
            r=dataset[i];valid=int(r['valid_tokens'])
            if not (r['attention_mask'][:valid]==1).all() or not (r['attention_mask'][valid:]==0).all() or not (r['labels'][valid:]==-100).all():raise ValueError('Loader padding/labels mismatch')
            samples[split]+=1
    train=ShardedTokenDataset(run/'tokenized','train',manifest['sequence_length'])
    schedule=DeterministicMixtureSchedule(train,seed=20261002)
    for rank in (0,1):
        s=MixtureDistributedSampler(schedule,num_replicas=2,rank=rank,micro_batch_size=1,gradient_accumulation_steps=2)
        it=iter(s);prefetched=[next(it) for _ in range(6)];s.commit_step();checkpoint=s.state_dict()
        restored=MixtureDistributedSampler(schedule,num_replicas=2,rank=rank,micro_batch_size=1,gradient_accumulation_steps=2);restored.load_state_dict(checkpoint)
        if [next(iter(restored))]!=[prefetched[2]]:raise ValueError('Committed sampler restart mismatch')
    return {'stage':'eligible_pack_physical_source_loader_verified','checker_sha256':file_sha(Path(__file__)),
        'parallel_engine_sha256':file_sha(ROOT/'scripts/p0_parallel_cpu_v13.py'),
        'pack_manifest_sha256':file_sha(run/'tokenized/manifest.json'),'source_sha256':file_sha(run/'deduplicated.parquet'),
        'tokenizer_sha256':file_sha(tokenizer),'documents_by_split':dict(docs),'body_tokens_by_split':dict(bodies),
        'all_document_decode_reencode_and_eos_verified':True,'all_index_and_loader_samples_verified':dict(samples),
        'committed_two_rank_sampler_restore_verified':True,'global_exact_overlap_documents':0,'selected_family_split_conflicts':0,
        'global_near_approved':False,'independent_source_quality_approved':False,'benchmark_contamination_approved':False,'training_approved':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('run','tokenizer','database'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();config=json.loads(Path(os.environ['P0_PARALLEL_CONFIG']).read_text());r=verify(a.run,a.tokenizer,a.database,config['release_cpu_workers']);out=a.run/'VERIFY.json'
    if out.exists():
        if json.loads(out.read_text())!=r:raise ValueError('Immutable verifier output differs')
    else:
        tmp=out.with_name('.VERIFY.tmp');tmp.write_text(json.dumps(r,ensure_ascii=False,sort_keys=True,indent=2)+'\n');tmp.replace(out)
    print(json.dumps(r,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
