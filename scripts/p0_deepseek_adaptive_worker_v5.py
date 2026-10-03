#!/usr/bin/env python3
"""DeepSeek-only bounded adaptive concurrency; preserves frozen v3 protocols.

Each request is reserved once in the existing campaign ledger. Invalid returned
labels are quarantined. Rejected/unknown attempts are NEVER automatically resent.
This worker does not replace independently verified quality reference labels.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from decimal import Decimal
import fcntl, json, math
from pathlib import Path
import statistics, sys, time
from urllib.error import HTTPError
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.core import canonical, decoder_schema, sha, validate_annotation
from small_agent.curation.ledger import Ledger
from small_agent.curation.store import Batch, file_sha, protocol
from p0_curation_deepseek_env_v3 import load_profile, profile_worker_args
import p0_curation_worker_v3 as old
from p0_curation_bounded_resume_v3 import recoverable
from small_agent.curation.literal_context_v6 import adapt


def next_width(current, maximum, outcomes, latency, baseline):
    """Rate rejection/slowdown decreases the next window, not retries."""
    if any(r.get('http_status') in (429,503) for r in outcomes):return max(1,current//2),'rate_backoff'
    if baseline is not None and latency>max(baseline*3,60):return max(1,current//2),'latency_backoff'
    return min(maximum,current*2),'ramp' if current<maximum else 'steady'


def attempt(task,judge,prompt,schema,key,timeout):
    job,doc,body,base=task;start=time.monotonic()
    result={**base,'status':'error','generated_at_unix':time.time(),'actual_request_sha256':sha(canonical(body))}
    try:
        response=old.send(judge['endpoint']['base_url'],'responses',body,key,timeout)
        result.update(usage=response.get('usage'),response_id=response.get('id'),served_model=response.get('model'))
        if response.get('model') not in judge['served_models']:raise ValueError('served model mismatch')
        label=old.extract(response,'responses');result['raw_annotation']=label
        validate_annotation(label,doc,schema);validate_annotation(label,doc,decoder_schema(doc,schema,judge))
        result.update(status='ok',annotation=label)
    except HTTPError as e:
        result.update(http_status=e.code,error=f'HTTP {e.code}')
        # Do not store the provider's echoed body, headers or credentials.
    except Exception as e:result['error']=type(e).__name__
    result['latency_seconds']=round(time.monotonic()-start,6)
    return job,result


def safe_known_result(result,base,models):
    if result.get('status')=='ok':return True
    return recoverable(result,base,models)


def run(batch,profile,key,*,initial=8,maximum=2500,limit=2000,timeout=240,cooldown=5):
    if not 1<=initial<=maximum<=2500 or not 1<=limit<=10000:raise ValueError('bounded concurrency/limit required')
    judge=batch.registry['judges']['deepseek'];ep=judge['endpoint']
    if batch.registry['deployment_status']!='verified' or ep['kind']!='responses':raise ValueError('verified official Responses provider required')
    # Reuse the private wrapper's account, scope, budget, generation guards.
    profile_worker_args(profile,['--batch',str(batch.root),'--judge','deepseek','--allow-api',
        '--ledger',profile['campaign_ledger'],'--budget-cny',str(profile['campaign_budget_cny'])])
    if ep['pricing_cny_per_million']!=profile['pricing_cny_per_million']:raise ValueError('pricing differs from shared campaign')
    if sha(key)!=profile['expected_key_sha256']:raise ValueError('account binding differs')
    work=batch.root/'worker'/'deepseek';work.mkdir(parents=True,exist_ok=True)
    # Same per-batch lock as v3; plus one account-wide high-concurrency owner.
    locks=[]
    try:
        for path in [work/'worker.lock',Path(profile['campaign_ledger']).with_suffix('.adaptive.lock')]:
            f=path.open('a');locks.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=Ledger(Path(profile['campaign_ledger']),budget_cny=profile['campaign_budget_cny'],account_sha256=sha(key),pricing=profile['pricing_cny_per_million'])
        try:
            if ledger.report()['requests_by_state'].get('inflight',0):raise ValueError('unknown campaign inflight: investigate before new requests')
            # Failed batches cannot bypass the stop by merely excluding invalid rows from pending().
            for row in ledger.db.execute('SELECT result_json FROM requests WHERE result_json IS NOT NULL'):
                r=json.loads(row[0])
                if r.get('status')=='error' and r.get('http_status') not in (429,503):
                    expected={k:r[k] for k in ('source_id','text_sha256','prompt_sha256','schema_sha256','model','temperature','inference_sha256','judge_identity_sha256','wire_schema_sha256') if k in r}
                    if not recoverable(r,expected,judge['served_models']):
                        raise ValueError('campaign unknown/fatal attempt: explicit inspection required; never auto-resend')
            jobs=batch.pending('deepseek')[:limit];pos=0;width=1;pilot=True;baseline=None;history=[];counts=Counter();reason=None;rate_streak=0
            config=dict(worker_sha256=file_sha(Path(__file__)),batch_manifest_sha256=batch.manifest_hash,model=profile['model'],
                initial_concurrency=initial,maximum_concurrency=maximum,limit=limit,timeout=timeout,
                campaign_ledger=profile['campaign_ledger'],account_sha256=profile['expected_key_sha256'],
                retry_failed_requests=False)
            old.write_receipt(work/'ADAPTIVE_CONFIG.jsonl',config)
            while pos<len(jobs):
                window=jobs[pos:pos+width];tasks=[];cached=[];start=time.monotonic()
                for j in window:
                    doc=batch.docs[j['document_id']];base=protocol(doc,batch.prompt,batch.schema,judge)
                    job=old.job_identity(batch,'deepseek',base);previous=ledger.result(job)
                    if previous is not None:
                        old.write_receipt(work/(job+'.jsonl'),previous);batch.import_results('deepseek',work/(job+'.jsonl'))
                        cached.append(previous);counts['recovered_without_api']+=1
                        if not safe_known_result(previous,base,judge['served_models']):reason='cached_unknown_or_rejected_attempt_requires_inspection'
                        continue
                    body=adapt(old.request_body(doc,batch.prompt,batch.schema,judge),doc,judge)
                    input_bound=len(canonical(body).encode('utf-8'))+4096
                    if input_bound>ep['maximum_input_tokens']:
                        counts['context_held_without_api']+=1;continue
                    # Actual request byte bound saves reservations compared with always reserving max context.
                    bound=old.usage_cost(dict(input_tokens=input_bound,output_tokens=judge['generation']['max_tokens']),ep['pricing_cny_per_million'])
                    try:reserved=ledger.reserve(job,bound)
                    except ValueError:reason='budget_backpressure';break
                    if not reserved:reason='unknown_inflight_not_resent';break
                    tasks.append((job,doc,body,base))
                outcomes=[]
                with ThreadPoolExecutor(max_workers=max(1,len(tasks))) as executor:
                    futures={executor.submit(attempt,t,judge,batch.prompt,batch.schema,key,timeout):t for t in tasks}
                    for fut in as_completed(futures):
                        task=futures[fut];job,result=fut.result()
                        ledger.finish(job,result,observed_cost_cny=old.usage_cost(result.get('usage'),ep['pricing_cny_per_million']))
                        old.write_receipt(work/(job+'.jsonl'),result);batch.import_results('deepseek',work/(job+'.jsonl'))
                        outcomes.append(result);counts['new_api_requests']+=1
                        if result['status']=='ok':counts['valid_outputs']+=1
                        elif recoverable(result,task[3],judge['served_models']):counts['invalid_outputs_quarantined']+=1
                        elif result.get('http_status') in (429,503):counts['rejected_attempts_held']+=1
                        else:reason='unknown_transport_or_fatal_response';counts['unknown_or_fatal']+=1
                elapsed=time.monotonic()-start;pos+=len(window)
                rate_streak=rate_streak+1 if any(r.get('http_status') in (429,503) for r in outcomes) else 0
                if rate_streak>=3:reason='persistent_provider_rejection'
                latencies=[r['latency_seconds'] for r in outcomes];median=statistics.median(latencies) if latencies else 0
                if ledger.report()['cap_exceeded']:reason='cost_bound_exceeded'
                if pilot:
                    ok=any(r.get('status')=='ok' for r in outcomes+cached)
                    if not ok and tasks:reason=reason or 'pilot_not_valid'
                    if ok:pilot=False;baseline=median if median else None;width=initial
                    action='pilot'
                else:width,action=next_width(width,maximum,outcomes,median,baseline)
                row=dict(window=len(history),submitted=len(tasks),elapsed_seconds=round(elapsed,3),median_latency_seconds=median,
                    next_concurrency=width,action=action,counts=dict(counts),billing=ledger.report())
                history.append(row)
                status=dict(stage='running' if reason is None else 'stopped_fail_closed',stop_reason=reason,
                    considered=pos,selected=len(jobs),measured_peak_submitted=max(x['submitted'] for x in history),
                    configuration=config,history=history,counts=dict(counts),ledger=ledger.report(),batch=batch.report(),training_started=False)
                old.atomic_status(work/'ADAPTIVE_STATUS.json',status);print(canonical({k:status[k] for k in ['stage','stop_reason','considered','selected','measured_peak_submitted','counts','ledger']}),flush=True)
                if reason:break
                if action=='rate_backoff':time.sleep(cooldown)
            stage='all_selected_attempted_with_holds' if reason is None else 'stopped_fail_closed'
            status=dict(stage=stage,stop_reason=reason,selected=len(jobs),considered=pos,configuration=config,history=history,
                measured_peak_submitted=max((r['submitted'] for r in history),default=0),counts=dict(counts),ledger=ledger.report(),
                batch=batch.report(),training_approved=False,training_started=False)
            old.atomic_status(work/'ADAPTIVE_STATUS.json',status)
            return status
        finally:ledger.close()
    finally:
        for f in locks:f.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--batch',type=Path,required=True);p.add_argument('--allow-api',action='store_true')
    p.add_argument('--initial-concurrency',type=int,default=8);p.add_argument('--maximum-concurrency',type=int,default=2500)
    p.add_argument('--limit',type=int,default=2000);p.add_argument('--timeout',type=int,default=240)
    a=p.parse_args()
    if not a.allow_api:raise ValueError('explicit API authorization required')
    profile,key,_=load_profile(a.profile);batch=Batch(a.batch)
    try:r=run(batch,profile,key,initial=a.initial_concurrency,maximum=a.maximum_concurrency,limit=a.limit,timeout=a.timeout)
    finally:batch.close()
    return 0 if r['stop_reason'] is None else 3
if __name__=='__main__':raise SystemExit(main())
