#!/usr/bin/env python3
"""Full-document math rolling audits, immutable inputs, eight per length layer."""
import argparse,json,sys
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.release_gates_v5 import sha,write
from small_agent.curation.stock_v6 import layer
from small_agent.curation.supply_control import file_sha
from small_agent.curation.store import Batch,prepare
from p0_source_audit_v4 import registry_for
from p0_curation_deepseek_env_v3 import load_profile

def prepare_audit(parquet,profile_path,output,source_audit):
    if (output/'SAMPLE_PLAN.json').exists():
        p=json.loads((output/'SAMPLE_PLAN.json').read_text())
        if p['parquet_sha256']!=file_sha(parquet):raise ValueError('Frozen batch audit differs')
        return
    import pyarrow.parquet as pq
    src=Batch(source_audit/'batch')
    try:state=src.state_fingerprint();manifest=src.manifest_hash;registry=src.registry
    finally:src.close()
    strata=defaultdict(list);excluded=Counter()
    for b in pq.ParquetFile(parquet).iter_batches(batch_size=128):
        for r in b.to_pylist():
            if r['split']!='train' or r['language'].lower()!='en':continue
            if len(r['text'].encode())>100000:excluded['uncalibrated_body_size_hold']+=1;continue
            strata[layer(r)].append(r)
    selected=[(h,r) for h,rows in sorted(strata.items()) for r in sorted(rows,key=lambda r:sha('rolling-v6|'+r['normalized_sha256']))[:8]]
    if not selected:raise ValueError('No math audit population')
    output.mkdir(parents=True,exist_ok=True);docs=[]
    for h,r in selected:
        docs.append(dict(document_id=r['normalized_sha256'],document_text=r['text'],source_id=r['source_id'],source_locator=r['source_locator'],source_revision=r['source_revision'],family_id=r['family_id'],language='en',content_kind='text',token_count=r['own_text_tokens'],tokenizer_sha256='9d0b99ab9cf20d337217c96a5b39e9ff87590eb462abcbcc8f1119c3d43911a7',license_status='source_level_internal_scope_authorized',sampling_stratum=h,current_bucket='math_en'))
    (output/'input.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs))
    profile,_,_=load_profile(profile_path);write(output/'judges.json',registry)
    prepare(output/'input.jsonl',output/'batch',ROOT/'configs/curation/p0_source_audit_policy_parallel_v5.json',ROOT/'schemas/p0_teacher_quality_v3.json',source_audit/'batch/prompt.txt',output/'judges.json')
    write(output/'SAMPLE_PLAN.json',dict(version=6,parquet_path=str(parquet.resolve()),parquet_sha256=file_sha(parquet),input_sha256=file_sha(output/'input.jsonl'),source_manifest_sha256=manifest,review_state_sha256=state,selected=len(docs),population={h:len(r) for h,r in strata.items()},exclusions=dict(excluded),per_stratum=8,scope='full-body math source-scoped rolling spotcheck; not individual factual certification'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('parquet','profile','output','source-audit'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();prepare_audit(a.parquet,a.profile,a.output,a.source_audit)
    print('FROZEN_MATH_BATCH_AUDIT_READY')
if __name__=='__main__':main()
