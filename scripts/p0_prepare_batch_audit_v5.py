#!/usr/bin/env python3
"""Deterministic whole-document batch spot check, not per-document approval."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.release_gates_v5 import sha,write
from small_agent.curation.supply_control import file_sha
from p0_source_audit_v4 import registry_for
from p0_curation_deepseek_env_v3 import load_profile
from small_agent.curation.store import prepare
p=argparse.ArgumentParser(description=__doc__)
for n in ('parquet','profile','output','source-audit'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args()
if (a.output/'SAMPLE_PLAN.json').exists():print('EXISTING_BATCH_AUDIT_REUSED');raise SystemExit(0)
import pyarrow.parquet as pq
from collections import defaultdict
from small_agent.curation.store import Batch
source=Batch(a.source_audit/'batch')
try:state=source.state_fingerprint();manifest=source.manifest_hash
finally:source.close()
strata=defaultdict(list)
for b in pq.ParquetFile(a.parquet).iter_batches(batch_size=128):
 for r in b.to_pylist():
  if r['split']!='train' or r['language'].lower()!='python':continue
  n=r['own_text_tokens'];h=('short' if n<=512 else 'medium' if n<=4096 else 'long')+'/python'
  strata[h].append(r)
selected=[(h,r) for h,rows in sorted(strata.items()) for r in sorted(rows,key=lambda r:sha('rolling-v5|'+r['normalized_sha256']))[:8]]
if not selected:raise ValueError('No reference-calibrated Python batch strata')
a.output.mkdir(parents=True,exist_ok=True);docs=[]
for h,r in selected:
 docs.append(dict(document_id=r['normalized_sha256'],document_text=r['text'],source_id=r['source_id'],source_locator=r['source_locator'],source_revision=r['source_revision'],family_id=r['family_id'],language='python',content_kind='code_file_or_fragment',token_count=r['own_text_tokens'],tokenizer_sha256='9d0b99ab9cf20d337217c96a5b39e9ff87590eb462abcbcc8f1119c3d43911a7',license_status='record_license_whitelisted',sampling_stratum=h,current_bucket='code_python'))
(a.output/'input.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs));profile,_,_=load_profile(a.profile);write(a.output/'judges.json',registry_for(profile))
prepare(a.output/'input.jsonl',a.output/'batch',ROOT/'configs/curation/p0_source_audit_policy_parallel_v5.json',ROOT/'schemas/p0_teacher_quality_v3.json',ROOT/'configs/curation/p0_source_audit_prompt_v4_r2.txt',a.output/'judges.json')
write(a.output/'SAMPLE_PLAN.json',dict(stage='batch_spotcheck_prepared',parquet_path=str(a.parquet.resolve()),parquet_sha256=file_sha(a.parquet),input_sha256=file_sha(a.output/'input.jsonl'),source_manifest_sha256=manifest,review_state_sha256=state,selected=len(docs),population={h:len(r) for h,r in strata.items()},per_stratum=8,scope='Python source-calibrated language only; hash sampling within batch strata'))
print(json.dumps(dict(stage='batch_spotcheck_prepared',documents=len(docs),paid_api_calls=0)))
