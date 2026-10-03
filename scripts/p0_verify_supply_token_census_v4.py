#!/usr/bin/env python3
"""Independent streaming check of a completed candidate token census, not approval."""
import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import sqlite3
import time

from p0_data_pipeline import sha256_file

FROZEN_SHA='9d0b99ab9cf20d337217c96a5b39e9ff87590eb462abcbcc8f1119c3d43911a7'

def verify(root,tokenizer_path,sample_size=16):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    start=time.monotonic();report_path=root/'report.json';report_sha=sha256_file(report_path)
    r=json.loads(report_path.read_text())
    if r['stage']!='candidate_token_census_complete_not_ready' or not r['complete'] or r['training_approved'] or r['new_ready_train_tokens']!=0:
        raise ValueError('Not a completed unapproved candidate census')
    if r['completed_shards']!=r['planned_shards'] or len(r['files'])!=r['planned_shards']:
        raise ValueError('Missing counted shards')
    if sha256_file(tokenizer_path)!=FROZEN_SHA or r['identity']['tokenizer_sha256']!=FROZEN_SHA:
        raise ValueError('Frozen tokenizer mismatch')
    tok=Tokenizer.from_file(str(tokenizer_path));tok.no_padding();tok.no_truncation()
    overall=Counter();by_domain={};policy=Counter();verified=[];samples_checked=0
    for shard in r['files']:
        p=(root/shard['output_relative_path']).resolve()
        if root.resolve() not in p.parents or sha256_file(p)!=shard['output_sha256']:
            raise ValueError('Output SHA/path mismatch')
        receipt=json.loads((p.parent/'receipt.json').read_text())
        if receipt!=shard or receipt['identity_sha256']!=r['identity_sha256']:
            raise ValueError('Receipt identity mismatch')
        excluded=(root/shard['excluded_relative_path']).resolve()
        if root.resolve() not in excluded.parents or sha256_file(excluded)!=shard['excluded_sha256']:
            raise ValueError('Exclusion index mismatch')
        counts=Counter();policy_shard=Counter();heap=[]
        pf=pq.ParquetFile(p)
        cols=['document_id','text','normalized_sha256','own_text_tokens','domain','split','policy_bucket','training_approved']
        for batch in pf.iter_batches(batch_size=128,columns=cols):
            for row in batch.to_pylist():
                n=row['own_text_tokens'];split=row['split']
                if type(n)is not int or n<=0 or split not in {'train','validation','test','unassigned'}:
                    raise ValueError('Invalid counter or split')
                if row['domain']!=shard['domain'] or row['training_approved']:
                    raise ValueError('Unexpected domain/approval flag')
                if hashlib.sha256(row['text'].encode('utf-8')).hexdigest()!=row['normalized_sha256']:
                    raise ValueError('Stored body SHA mismatch')
                counts['candidate_documents']+=1;counts['candidate_body_tokens']+=n
                counts['documents:'+split]+=1;counts['body_tokens:'+split]+=n
                policy_shard[row['policy_bucket']+'/'+split]+=n
                # Fixed hash-ranked sample across the whole output, not first rows.
                rank=int(hashlib.sha256(row['document_id'].encode()).hexdigest(),16)
                item=(-rank,row['document_id'],row['text'],n)
                if len(heap)<sample_size:heapq.heappush(heap,item)
                elif item[:2]>heap[0][:2]:heapq.heapreplace(heap,item)
        for key in ['candidate_documents','candidate_body_tokens']+[prefix+split for prefix in ['documents:','body_tokens:'] for split in ['train','validation','test','unassigned']]:
            if counts[key]!=shard['counts'].get(key,0):raise ValueError('Parquet/receipt count mismatch')
        if dict(policy_shard)!=shard['policy_split_body_tokens']:raise ValueError('Policy bucket mismatch')
        if pf.metadata.num_rows!=counts['candidate_documents']:raise ValueError('Footer mismatch')
        with excluded.open() as f:
            exclusions=sum(1 for line in f if json.loads(line)['reason_codes'])
        expected=shard['counts']['raw_documents_scanned']-counts['candidate_documents']
        if exclusions!=expected:raise ValueError('Exclusion document accounting mismatch')
        entries=[x for x in heap];enc=tok.encode_batch([x[2] for x in entries],add_special_tokens=False)
        for entry,e in zip(entries,enc):
            if len(e.ids)!=entry[3] or tok.decode(e.ids,skip_special_tokens=False)!=entry[2]:
                raise ValueError('Independent re-encode/roundtrip mismatch')
        samples_checked+=len(entries);overall.update(counts)
        by_domain.setdefault(shard['domain'],Counter()).update(counts);policy.update(policy_shard)
        verified.append({'key':shard['key'],'counts':dict(counts),'reencoded_documents':len(entries)})
        print('VERIFIED_COUNT',shard['source_id'],shard['source_file'],dict(counts),flush=True)
    for k,v in overall.items():
        if r['counts'].get(k,0)!=v:raise ValueError('Aggregate mismatch')
    if dict(policy)!=r['policy_split_body_tokens']:raise ValueError('Overall policy mismatch')
    db=sqlite3.connect('file:'+str((root/'census.sqlite').resolve())+'?mode=ro',uri=True)
    try:
        size,tokens=db.execute('SELECT COUNT(*),SUM(tokens) FROM documents').fetchone()
        if (size,tokens)!=(overall['candidate_documents'],overall['candidate_body_tokens']):raise ValueError('Unique index/Parquet mismatch')
        for domain,split,size,tokens in db.execute('SELECT domain,split,COUNT(*),SUM(tokens) FROM documents GROUP BY domain,split'):
            if size!=by_domain[domain]['documents:'+split] or tokens!=by_domain[domain]['body_tokens:'+split]:raise ValueError('Domain split index mismatch')
    finally:db.close()
    if sha256_file(report_path)!=report_sha:raise ValueError('Report changed during verification')
    return {'stage':'candidate_token_census_verified_not_training_approved','report_sha256':report_sha,
        'identity_sha256':r['identity_sha256'],'counts':dict(overall),
        'by_domain':{d:dict(c) for d,c in by_domain.items()},'policy_split_body_tokens':dict(policy),
        'verified_shards':len(verified),'reencoded_documents':samples_checked,'full_body_hash_checks':True,
        'full_corpus_reencoding':False,'training_approved':False,'new_ready_train_tokens':0,
        'pending':r['pending'],'elapsed_s':time.monotonic()-start,'files':verified}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True)
    a=p.parse_args();v=verify(a.root,a.tokenizer)
    # This report is a new artefact, never a mutation of the main census receipt.
    path=a.root/'verification.json';tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
    print('VERIFIED_CENSUS',json.dumps({k:v[k] for k in ('stage','counts','by_domain','verified_shards','reencoded_documents')}),flush=True)

if __name__=='__main__':main()
