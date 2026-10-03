#!/usr/bin/env python3
"""Source calibration and rolling spot checks via the existing v3 review kernel.

No training/source approval. Prepare/report are offline; score requires both
explicit API authorization and the private profile's positive shared budget.
"""
from __future__ import annotations
import argparse
from collections import Counter
import heapq
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from small_agent.curation.core import canonical,fingerprint,normalize_document,sha
from small_agent.curation.store import Batch,prepare,file_sha,read_json,write_json
from p0_curation_deepseek_env_v3 import load_profile


def stratum(row,config):
    n=row['own_text_tokens'];bounds=config['length_bins_own_tokens']
    size='short' if n<=bounds[0] else 'medium' if n<=bounds[1] else 'long'
    lang='python' if str(row['language']).lower()=='python' else 'other_code' if row['domain']=='code' else 'en'
    if len(row['text'].encode('utf-8'))>config['maximum_sample_body_bytes']:size='over_budget_body_size'
    return size+'/'+lang


def offer(heap,rank,row,k):
    item=(-int(rank,16),row['normalized_sha256'],row)
    if len(heap)<k:heapq.heappush(heap,item)
    elif item[:2]>heap[0][:2]:heapq.heapreplace(heap,item)


def registry_for(profile):
    judge={'family':'deepseek','model':profile['model'],
        'model_revision':profile['model_display_name']+':model_metadata_sha256:'+profile['model_metadata_sha256'],
        'provider_weight_revision_not_pinnable':True,'served_models':[profile['model']],
        'generation':{'temperature':0,'thinking':'disabled','response_format':'json_schema',
            'max_tokens':profile['max_output_tokens'],'require_document_id_match':True,'chat_template_kwargs':{}},
        'endpoint':{'kind':'responses','base_url':profile['api_base'],'key_env':'DEEPSEEK_API_KEY',
            'maximum_input_tokens':profile['maximum_input_tokens'],'pricing_cny_per_million':profile['pricing_cny_per_million']}}
    return {'version':3,'deployment_status':'verified','verification_scope':'account/models metadata plus previously tested transport; new sample not live tested',
        'judges':{'deepseek':judge}}


def worker_module():
    p=ROOT/'scripts/p0_curation_worker_v3.py'
    spec=importlib.util.spec_from_file_location('source_audit_existing_worker',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def snapshot_files(census_root,plan,source,allow_partial):
    planned={i['path']:i for i in plan['items'] if i['source_id']==source}
    if not planned:raise ValueError('Source absent from pinned download plan')
    d=sqlite3.connect('file:'+str((census_root/'census.sqlite').resolve())+'?mode=ro',uri=True)
    try:
        receipts=[json.loads(r[0]) for r in d.execute('SELECT receipt FROM shards ORDER BY key')]
    finally:d.close()
    files=[r for r in receipts if r['source_id']==source]
    if not files or (len(files)!=len(planned) and not allow_partial):
        raise ValueError('Source census not complete; wait or explicitly declare a partial-source snapshot')
    for r in files:
        p=planned.get(r['source_file'])
        if not p or p['sha256']!=r['source_sha256']:raise ValueError('Census/source plan mismatch')
        path=(census_root/r['output_relative_path']).resolve()
        if census_root.resolve() not in path.parents or file_sha(path)!=r['output_sha256']:
            raise ValueError('Normalized candidate source SHA/path mismatch')
        if read_json(path.parent/'receipt.json')!=r:raise ValueError('Committed source receipt changed')
    return sorted(files,key=lambda r:r['source_file']),len(planned)


def prepare_audit(args):
    import pyarrow.parquet as pq
    profile,_,_=load_profile(args.profile)
    cfg=read_json(args.sampling);policy=read_json(args.policy)
    if cfg['version']!=4 or cfg['review_purpose']!='source_quality_audit' or policy.get('review_purpose')!='source_quality_audit':
        raise ValueError('Expected source-audit-only configuration')
    if args.output.exists():
        manifest=read_json(args.output/'SAMPLE_PLAN.json')
        expected={'mode':args.mode,'round_id':args.round_id,'source_id':args.source,
            'plan_sha256':file_sha(args.download_plan),'sampling_sha256':file_sha(args.sampling),
            'policy_sha256':file_sha(args.policy),'prompt_sha256':file_sha(args.prompt),'schema_sha256':file_sha(args.schema),
            'model_metadata_sha256':profile['model_metadata_sha256'],'adapter_sha256':file_sha(Path(__file__))}
        if manifest['request_identity']!=expected:raise ValueError('Existing audit run has different configuration')
        for name,expected_sha in manifest['frozen_files'].items():
            if file_sha(args.output/name)!=expected_sha:raise ValueError('Frozen audit input changed')
        batch=Batch(args.output/'batch')
        try:batch.report()
        finally:batch.close()
        return {'stage':'existing_source_audit_reused','documents':manifest['admitted_documents'],'new_api_calls':0}
    files,total_planned=snapshot_files(args.census_root,read_json(args.download_plan),args.source,args.allow_partial_source)
    cfg_allowed=cfg['source_policy_buckets'].get(args.source)
    if not cfg_allowed:raise ValueError('No declared source policy bucket scope')
    k=cfg['calibration_per_stratum'] if args.mode=='calibration' else cfg['rolling_per_stratum']
    if not 1<=k<=64:raise ValueError('Invalid per-stratum sample size')
    heaps={};population=Counter();excluded=Counter();total_population_tokens=0
    for receipt in files:
        p=args.census_root/receipt['output_relative_path']
        cols=['document_id','source_id','source_revision','source_locator','normalized_sha256','text',
              'own_text_tokens','domain','language','family_id','split','policy_bucket']
        for batch in pq.ParquetFile(p).iter_batches(batch_size=128,columns=cols):
            for row in batch.to_pylist():
                if row['split']!='train':excluded['not_provisional_train']+=1;continue
                if row['policy_bucket'] not in cfg_allowed:excluded['policy_bucket_not_in_audit_scope']+=1;continue
                if sha(row['text'])!=row['normalized_sha256']:raise ValueError('Candidate body hash mismatch')
                h=stratum(row,cfg);population[h]+=1;total_population_tokens+=row['own_text_tokens']
                if h.startswith('over_budget_body_size/'):continue
                rank=sha(canonical([cfg['seed'],args.source,args.mode,args.round_id,row['normalized_sha256']]))
                offer(heaps.setdefault(h,[]),rank,row,k)
        if file_sha(p)!=receipt['output_sha256']:raise ValueError('Candidate file changed during sampling')
        print('SAMPLED_SHARD',args.source,receipt['source_file'],dict(population),flush=True)
    if not population:raise ValueError('No in-scope source candidate population')
    selected=[(h,item[2]) for h in sorted(heaps) for item in sorted(heaps[h],key=lambda x:(-x[0],x[1]))]
    if not selected or len(selected)>cfg['max_sample_documents']:raise ValueError('Empty or excessive sample; change configuration explicitly')
    schema=read_json(args.schema);prompt=args.prompt.read_text().strip();registry=registry_for(profile)
    worker=worker_module();docs=[];index=[];admission=Counter()
    for h,row in selected:
        record={'document_id':row['normalized_sha256'],'document_text':row['text'],'source_id':args.source,
            'source_locator':row['source_locator'],'source_revision':row['source_revision'],
            'family_id':row['family_id'],'language':str(row['language']).lower(),
            'content_kind':'code_file_or_fragment' if row['domain']=='code' else 'text',
            'token_count':row['own_text_tokens'],'tokenizer_sha256':'9d0b99ab9cf20d337217c96a5b39e9ff87590eb462abcbcc8f1119c3d43911a7',
            'license_status':row['policy_bucket'],'sampling_stratum':h,
            'current_bucket':'code_python' if str(row['language']).lower()=='python' else 'code_other' if row['domain']=='code' else 'math_en'}
        normalized=normalize_document(record)
        body=worker.request_body(normalized,prompt,schema,registry['judges']['deepseek'])
        byte_bound=len(canonical(body).encode('utf-8'))+4096
        status='admitted' if byte_bound<=profile['maximum_input_tokens'] else 'full_body_context_hold'
        admission[status]+=1
        index.append({'document_id':record['document_id'],'candidate_document_id':row['document_id'],
            'stratum':h,'model_admission':status,'conservative_input_byte_bound':byte_bound,
            'population':population[h],'selected_in_stratum':len(heaps[h]),
            'inclusion_probability':min(k,population[h])/population[h]})
        if status=='admitted':docs.append(record)
    if not docs:raise ValueError('All selected full documents blocked; no truncation or replacement sampling')
    identity={'mode':args.mode,'round_id':args.round_id,'source_id':args.source,
        'plan_sha256':file_sha(args.download_plan),'sampling_sha256':file_sha(args.sampling),
        'policy_sha256':file_sha(args.policy),'prompt_sha256':file_sha(args.prompt),'schema_sha256':file_sha(args.schema),
        'model_metadata_sha256':profile['model_metadata_sha256'],'adapter_sha256':file_sha(Path(__file__))}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='.source-audit-',dir=args.output.parent))
    try:
        for path,name in [(args.sampling,'sampling.json'),(args.policy,'policy.json'),(args.prompt,'prompt.txt'),(args.schema,'schema.json')]:
            shutil.copyfile(path,tmp/name)
        write_json(tmp/'judges.json',registry)
        with (tmp/'input.jsonl').open('w') as f:
            for row in docs:f.write(canonical(row)+'\n')
        with (tmp/'selection.jsonl').open('w') as f:
            for row in index:f.write(canonical(row)+'\n')
        batch_manifest=prepare(tmp/'input.jsonl',tmp/'batch',tmp/'policy.json',tmp/'schema.json',tmp/'prompt.txt',tmp/'judges.json')
        frozen={n:file_sha(tmp/n) for n in ['sampling.json','policy.json','prompt.txt','schema.json','judges.json','input.jsonl','selection.jsonl']}
        design={'stage':'source_audit_input_prepared_not_source_approval','request_identity':identity,
            'scope':'provisional_train_candidates_in_declared_policy_buckets_only','source_complete_in_plan':len(files)==total_planned,
            'completed_source_shards':len(files),'planned_source_shards':total_planned,'input_source_files':files,
            'population_by_stratum':dict(population),'population_documents':sum(population.values()),
            'population_candidate_body_tokens':total_population_tokens,'excluded_from_scope':dict(excluded),
            'selected_documents':len(index),'admitted_documents':len(docs),'admission':dict(admission),
            'sampling':'hash top K within fixed length/language strata after whole declared snapshot scan',
            'sampling_unit':'document; family correlation not removed, no iid factual-error confidence interval',
            'frozen_files':frozen,'batch_manifest_sha256':file_sha(tmp/'batch/MANIFEST.json'),
            'new_api_calls':0,'source_approved':False,'training_approved':False,'new_ready_train_tokens':0,
            'limits':['Model signals do not certify facts, source rights or independent defect rate.',
                'Oversize/context-held and invalid items stay in coverage reporting; no silent replacement.',
                'Model payload excludes upstream score, existing bucket and prior judge labels.',
                'Calibration and rolling checks do not automatically release source or training data.']}
        write_json(tmp/'SAMPLE_PLAN.json',design);tmp.rename(args.output)
        return {k:design[k] for k in ('stage','population_documents','population_by_stratum','selected_documents','admitted_documents','new_api_calls','new_ready_train_tokens')}
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True);raise


def audit_report(root):
    design=read_json(root/'SAMPLE_PLAN.json')
    for name,expected in design['frozen_files'].items():
        if file_sha(root/name)!=expected:raise ValueError('Audit input changed')
    batch=Batch(root/'batch')
    try:
        if file_sha(root/'batch/MANIFEST.json')!=design['batch_manifest_sha256']:raise ValueError('Wrong review batch')
        report=batch.report();rows={r['document_id']:r for r in batch.decisions()}
    finally:batch.close()
    selected=[json.loads(line) for line in (root/'selection.jsonl').read_text().splitlines()]
    strata={h:Counter() for h in design['population_by_stratum']}
    for r in selected:
        state=rows[r['document_id']]['status'] if r['model_admission']=='admitted' else 'audit_context_hold'
        strata[r['stratum']][state]+=1;strata[r['stratum']]['selected']+=1
    weighted=Counter();total=design['population_documents'];covered=0
    for h,n in design['population_by_stratum'].items():
        k=strata[h]['selected']
        if not k:continue
        covered+=n
        for state,c in strata[h].items():
            if state!='selected':weighted[state]+=n/total*c/k
    return {'stage':'source_audit_diagnostics_not_source_approval','source_id':design['request_identity']['source_id'],
        'mode':design['request_identity']['mode'],'round_id':design['request_identity']['round_id'],
        'source_complete_in_plan':design['source_complete_in_plan'],
        'population_documents':total,'population_by_stratum':design['population_by_stratum'],
        'selected_documents':design['selected_documents'],'admitted_documents':design['admitted_documents'],
        'stratum_review_signals':{h:dict(c) for h,c in strata.items()},
        'document_weighted_model_signals':dict(weighted),'sampled_stratum_population_weight':covered/total,
        'unsampled_population_weight':1-covered/total,'review':report,
        'independent_defect_rate':None,'source_approved':False,'training_approved':False,'new_ready_train_tokens':0,
        'limits':design['limits']}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    for n in ['census-root','download-plan','profile','output']:prep.add_argument('--'+n,type=Path,required=True)
    for n,file in [('sampling','p0_source_audit_sampling_v4.json'),('policy','p0_source_audit_policy_v4.json')]:
        prep.add_argument('--'+n,type=Path,default=ROOT/'configs/curation'/file)
    prep.add_argument('--prompt',type=Path,default=ROOT/'configs/curation/p0_source_audit_prompt_v4.txt')
    prep.add_argument('--schema',type=Path,default=ROOT/'schemas/p0_teacher_quality_v3.json')
    prep.add_argument('--source',required=True);prep.add_argument('--mode',choices=['calibration','rolling'],required=True)
    prep.add_argument('--round-id',required=True);prep.add_argument('--allow-partial-source',action='store_true')
    for name in ['status','report','score']:
        a=sub.add_parser(name);a.add_argument('--run',type=Path,required=True)
        if name=='score':
            a.add_argument('--profile',type=Path,required=True);a.add_argument('--allow-api',action='store_true')
            a.add_argument('--budget-cny',required=True);a.add_argument('--limit',type=int,default=16)
    args=p.parse_args()
    if args.action=='prepare':result=prepare_audit(args)
    elif args.action=='score':
        if not args.allow_api:raise ValueError('Explicit --allow-api required; prepare and status never call a model')
        profile,_,_=load_profile(args.profile)
        if not 1<=args.limit<=128:raise ValueError('Bounded limit required')
        report=audit_report(args.run)
        if report['review']['review_purpose']!='source_quality_audit':raise ValueError('Not a source-audit batch')
        cmd=[sys.executable,str(ROOT/'scripts/p0_curation_deepseek_env_v3.py'),'--profile',str(args.profile),'--',
            '--batch',str(args.run/'batch'),'--judge','deepseek','--allow-api','--ledger',profile['campaign_ledger'],
            '--budget-cny',args.budget_cny,'--workers',str(profile['default_workers']),'--limit',str(args.limit)]
        raise SystemExit(subprocess.call(cmd))
    else:
        result=audit_report(args.run)
        if args.action=='report':
            out=args.run/'reports'/('state-'+result['review']['review_state_sha256']+'.json')
            out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists():
                if read_json(out)!=result:raise ValueError('Immutable report differs')
            else:write_json(out,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
