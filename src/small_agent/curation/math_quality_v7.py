"""Isolated v7 evidence-ID math quality gate; original thresholds retained.

Preserves v5 Jaccard/length/shared-sketch/dense-hold rules; replaces repeated
whole-inventory scans with a persistent posting index. Not semantic dedup.
"""
from collections import Counter,defaultdict,OrderedDict
import heapq,json,math,sqlite3,struct,time
from pathlib import Path
from .release_gates_v5 import sha,write,bound_json,lexical,ledger_snapshot,inventory_files,ParquetPointer
from .supply_control import file_sha,digest

def layer(row):
    n=row['own_text_tokens'];return ('short' if n<=512 else 'medium' if n<=4096 else 'long')+'/'+row['language'].lower()

def math_quality(source_audit,reference,reference_batch,policy,batch_audit=None):
    from .store_v7 import Batch
    from .core import quality_pass
    ref=bound_json(reference);holds=[];counts=Counter()
    if ref.get('domain')!='math' or ref.get('language')!='en' or ref.get('training_eligible') is not False or not ref.get('independently_verified_arithmetic_steps'):
        raise ValueError('Independent arithmetic reference required')
    def read(root):
        b=Batch(root)
        try:
            annotations={r[0]:json.loads(r[1]) for r in b.db.execute("SELECT document_id,annotation FROM jobs WHERE judge='deepseek' AND status='valid'")}
            return annotations,b.docs,b.policy,b.state_fingerprint(),b.manifest_hash
        finally:b.close()
    labels=json.loads((Path(reference).parent/'labels.json').read_text());ann,docs,rpolicy,state,manifest=read(reference_batch)
    for l in labels:
        c=l['expected'];counts[c+'_total']+=1;a=ann.get(l['document_id'])
        if a is None:counts[c+'_invalid']+=1
        elif quality_pass(a,rpolicy):counts[c+'_keep']+=1
    for c in ('keep','drop'):
        if counts[c+'_total']<policy['minimum_reference_per_class']:holds.append('reference_count_'+c)
        if (counts[c+'_total']-counts[c+'_invalid'])/max(1,counts[c+'_total'])<policy['minimum_valid_rate']:holds.append('reference_coverage_'+c)
    if counts['drop_keep']>policy['maximum_reference_false_keep']:holds.append('controlled_arithmetic_false_keep')
    if counts['keep_keep']/max(1,counts['keep_total'])<policy['minimum_reference_keep_rate']:holds.append('external_positive_keep_rate')
    source_audit=Path(source_audit);plan=json.loads((source_audit/'SAMPLE_PLAN.json').read_text())
    for n,h in plan['frozen_files'].items():
        if file_sha(source_audit/n)!=h:raise ValueError('Source calibration changed')
    src,sd,sp,ss,sm=read(source_audit/'batch')
    if not plan['source_complete_in_plan']:holds.append('partial_source_calibration')
    if file_sha(source_audit/'batch/prompt.txt')!=file_sha(Path(reference_batch)/'prompt.txt') or file_sha(source_audit/'batch/judges.json')!=file_sha(Path(reference_batch)/'judges.json'):holds.append('reference_protocol_differs')
    if time.time()-ref['created_unix']>86400*policy['maximum_calibration_age_days']:holds.append('reference_expired')
    if set(sd)&{l['document_id'] for l in labels}:holds.append('reference_source_exact_overlap')
    strata=defaultdict(Counter);bad=[]
    # Crucially use ALL originally selected items, including context-held ones.
    for line in (source_audit/'selection.jsonl').read_text().splitlines():
        item=json.loads(line);h=item['stratum'];c=strata[h];c['selected']+=1;a=src.get(item['document_id'])
        if a is not None:c['valid']+=1
        if a is not None and quality_pass(a,sp):c['keep']+=1
        else:bad.append(item['document_id'])
    approved=[]
    if not holds:
        approved=[h for h,c in strata.items() if h.endswith('/en') and c['selected']>=policy['minimum_source_samples_per_stratum'] and c['valid']/c['selected']>=policy['minimum_valid_rate'] and c['keep']/c['selected']>=policy['minimum_joint_keep_rate']]
    if not approved:holds.append('no_calibrated_math_stratum')
    if batch_audit is not None:
        ba=bound_json(batch_audit)
        if ba.get('status')!='passed' or ba.get('source_manifest_sha256')!=sm or ba.get('review_state_sha256')!=ss:holds.append('rolling_audit_stale_or_failed')
        else:approved=[h for h in approved if h in ba['approved_strata']]
        if not approved:holds.append('no_joint_batch_stratum')
    return dict(status='passed' if not holds else 'hold',holds=holds,source_id=plan['request_identity']['source_id'],approved_language='en',approved_strata=approved,
        source_strata={h:dict(c) for h,c in strata.items()},reference_counts=dict(counts),known_audit_bad_body_sha256=sorted(set(bad)),
        source_manifest_sha256=sm,source_review_state_sha256=ss,reference_sha256=file_sha(reference),reference_manifest_sha256=manifest,
        reference_review_state_sha256=state,source_level_only=True,natural_corpus_factual_error_rate_certified=False,
        scope='English mathematical content; independently checked arithmetic controls, not per-document correctness')


def rolling_audit(run,source_audit,minimum_valid=.85,minimum_keep=.75):
 from .store_v7 import Batch
 from .core import quality_pass
 run=Path(run);plan=json.loads((run/'SAMPLE_PLAN.json').read_text())
 if file_sha(Path(plan['parquet_path']))!=plan['parquet_sha256'] or file_sha(run/'input.jsonl')!=plan['input_sha256']:raise ValueError('Batch audit body binding changed')
 b=Batch(run/'batch');s=Batch(Path(source_audit)/'batch');strata=defaultdict(Counter);bad=[]
 try:
  if s.manifest_hash!=plan['source_manifest_sha256'] or s.state_fingerprint()!=plan['review_state_sha256']:raise ValueError('Source calibration changed')
  if b.prompt!=s.prompt or b.registry!=s.registry:raise ValueError('Rolling evidence protocol differs')
  annotations={r[0]:json.loads(r[1]) if r[1] else None for r in b.db.execute("SELECT document_id,annotation FROM jobs WHERE judge='deepseek' AND status='valid'")}
  for d in b.docs.values():
   c=strata[d['sampling_stratum']];c['selected']+=1;a=annotations.get(d['document_id'])
   if a is not None:c['valid']+=1
   if a is not None and quality_pass(a,b.policy):c['keep']+=1
   else:bad.append(d['document_id'])
  approved=[h for h,c in strata.items() if c['selected']>=min(plan['per_stratum'],plan['population'][h]) and c['valid']/c['selected']>=minimum_valid and c['keep']/c['selected']>=minimum_keep]
  held=sorted(set(strata)-set(approved));passed=bool(approved)
  return dict(status='passed' if passed else 'hold',approved_strata=sorted(approved),held_strata=held,failed_stratum_action='hold_entire_stratum',source_manifest_sha256=s.manifest_hash,review_state_sha256=s.state_fingerprint(),
   parquet_path=plan['parquet_path'],parquet_sha256=plan['parquet_sha256'],known_bad_body_sha256=bad,strata={h:dict(c) for h,c in strata.items()},
   batch_manifest_sha256=b.manifest_hash,batch_review_state_sha256=b.state_fingerprint(),evidence_files=[dict(path=str((run/'SAMPLE_PLAN.json').resolve()),sha256=file_sha(run/'SAMPLE_PLAN.json'))],
   scope='whole-document stratified batch spotcheck; not individual-document or factual certification')
 finally:b.close();s.close()
