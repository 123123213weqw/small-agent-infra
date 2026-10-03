#!/usr/bin/env python3
"""Bound arithmetic reference gate, reported raw/veto decisions separately."""
import argparse,json,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.store_v8 import Batch
from small_agent.curation.core import quality_pass
from small_agent.curation.release_gates_v5 import bound_json,write
from small_agent.curation.supply_control import file_sha

def reference_gate(root,policy):
    receipt=bound_json(root/'REFERENCE.json')
    if receipt.get('training_eligible') is not False or not receipt.get('independently_verified_arithmetic_steps'):raise ValueError('Independent checked reference required')
    labels=json.loads((root/'labels.json').read_text());b=Batch(root/'batch');c=Counter()
    try:
        labels_by_id={d['document_id']:d['expected'] for d in labels}
        ann={r[0]:json.loads(r[1]) for r in b.db.execute("SELECT document_id,annotation FROM jobs WHERE status='valid'")}
        for l in labels:
            k=l['expected'];c[k+'_total']+=1;a=ann.get(l['document_id'])
            if a is None:c[k+'_invalid']+=1
            elif quality_pass(a,b.policy):c[k+'_keep']+=1
        for row in b.db.execute('SELECT result_json FROM attempts'):
            r=json.loads(row[0]);wire=r.get('wire_annotation',{});k=labels_by_id.get(wire.get('document_id'),'unknown')
            c[k+'_raw_teacher_'+wire.get('decision','missing')]+=1
            if r.get('conservative_keep_veto'):c[k+'_conservative_keep_veto']+=1
        holds=[]
        for k in ['keep','drop']:
            if c[k+'_total']<policy['minimum_reference_per_class']:holds.append('reference_count_'+k)
            if (c[k+'_total']-c[k+'_invalid'])/max(1,c[k+'_total'])<policy['minimum_valid_rate']:holds.append('reference_coverage_'+k)
        if c['drop_keep']>policy['maximum_reference_false_keep']:holds.append('controlled_arithmetic_false_keep')
        if c['keep_keep']/max(1,c['keep_total'])<policy['minimum_reference_keep_rate']:holds.append('positive_keep_rate')
        result=dict(version=8,status='hold' if holds else 'passed',holds=holds,counts=dict(c),reference_manifest_sha256=b.manifest_hash,review_state_sha256=b.state_fingerprint(),reference_sha256=file_sha(root/'REFERENCE.json'),evaluation_reference_only=True,quality_thresholds_unchanged=True)
        write(root/'REFERENCE_GATE.json',result);print(json.dumps(result,ensure_ascii=False));return result
    finally:b.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--policy',type=Path,required=True);a=p.parse_args();r=reference_gate(a.reference,json.loads(a.policy.read_text())['quality']);raise SystemExit(0 if r['status']=='passed' else 4)
