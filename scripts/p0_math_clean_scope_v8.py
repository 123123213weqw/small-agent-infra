#!/usr/bin/env python3
"""Publish measured FILTERED-subset scope, never blanket approval of original pool."""
import json,sys,sqlite3,heapq
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.math_quality_v8 import math_quality
from small_agent.curation.release_gates_v5 import write,bound_json
from small_agent.curation.supply_control import file_sha

def main():
    r=ROOT.parent;source=r/'source_audits/math_clean_calibration_20261002_v8_r5';reference=r/'references/math_quality_v8_r5';subset=r/'stock_117b_v6/math_clean_v7'
    plan=json.loads((source/'SAMPLE_PLAN.json').read_text());f=bound_json(subset/'FILTER_COMPLETE.json')
    if plan['structural_filter_receipt_sha256']!=file_sha(subset/'FILTER_COMPLETE.json') or plan['original_cohort_overlap']!=0:raise ValueError('Filtered cohort identity differs')
    # Reconstruct the paid cohort from frozen pointer population; then bind DB bytes.
    db=sqlite3.connect('file:'+str((subset/'eligible.sqlite').resolve())+'?mode=ro',uri=True)
    excluded={json.loads(l)['document_id'] for l in (r/'source_audits/math_calibration_20261002_v6_r3/selection.jsonl').read_text().splitlines()}
    excluded.update(json.loads(l)['document_id'] for l in (reference/'input.jsonl').read_text().splitlines())
    from small_agent.curation.release_gates_v5 import sha
    heaps=defaultdict(list);population=Counter()
    for h,sid,pos,n,label in db.execute("SELECT h,sid,pos,n,stratum FROM eligible WHERE split='train'"):
        population[label]+=1
        if h in excluded:continue
        v=(-int(sha(str(plan['request_identity']['sampling_seed'])+'|'+h),16),h,sid,pos,n)
        if len(heaps[label])<64:heapq.heappush(heaps[label],v)
        elif v>heaps[label][0]:heapq.heapreplace(heaps[label],v)
    db.close()
    expected={x[1] for heap in heaps.values() for x in heap};actual={json.loads(l)['document_id'] for l in (source/'selection.jsonl').read_text().splitlines()}
    if expected!=actual or dict(population)!=plan['population']:raise ValueError('Paid sample does not match complete filtered population')
    write(subset/'FILTER_DATABASE_BOUND.json',dict(version=8,database_sha256=file_sha(subset/'eligible.sqlite'),filter_receipt_sha256=file_sha(subset/'FILTER_COMPLETE.json'),paid_cohort_recomputed=True,evidence_files=[dict(path=str((subset/'eligible.sqlite').resolve()),sha256=file_sha(subset/'eligible.sqlite')),dict(path=str((subset/'FILTER_COMPLETE.json').resolve()),sha256=file_sha(subset/'FILTER_COMPLETE.json'))]))
    policy=json.loads((ROOT/'configs/curation/p0_release_gates_v5.json').read_text())
    s=math_quality(source,reference/'REFERENCE.json',reference/'batch',policy['quality']);s.update(candidate_scope='ONLY frozen conservative structural subset; not all 10.064B original math tokens',filter_receipt_sha256=file_sha(subset/'FILTER_COMPLETE.json'),filtered_train_body_tokens=f['stats'].get('passed_train_body_tokens',0),new_ready_train_body_tokens=0,training_started=False)
    write(subset/'QUALITY_SCOPE.json',s);print(json.dumps({k:v for k,v in s.items() if k!='known_audit_bad_body_sha256'},ensure_ascii=False))
if __name__=='__main__':main()
