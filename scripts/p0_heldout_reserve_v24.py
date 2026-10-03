"""Bounded heldout reservoir and post-gate checks. Never change split or quality."""
from collections import Counter
import math

class HeldoutShortage(ValueError):
    pass

def select_reserve(store,query,params,strata,*,target_tokens=16384,maximum_documents_per_stratum=128):
    if type(target_tokens) is not int or target_tokens<2 or type(maximum_documents_per_stratum) is not int or not 1<=maximum_documents_per_stratum<=256:
        raise ValueError('Invalid bounded heldout reserve')
    result=[]
    for split in ('validation','test'):
        chosen=[]
        for stratum in sorted(set(strata)):
            tokens=0
            rows=store.execute(query+' AND r.split=? AND r.stratum=? ORDER BY r.rank LIMIT ?',
                               [*params,split,stratum,maximum_documents_per_stratum])
            for row in rows:
                if row['split']!=split or row['n'] is None or row['n']<1:raise ValueError('Invalid heldout metadata')
                chosen.append(row);tokens+=row['n']+1
                if tokens>=target_tokens:break
        if not chosen:return None
        result.extend(chosen)
    return result

def split_report(rows,sequence_length):
    if type(sequence_length) is not int or sequence_length<2:raise ValueError('Invalid sequence length')
    counts=Counter();tokens=Counter();seen=set()
    for row in rows:
        split=row['split'];n=row['own_text_tokens'];body=row['normalized_sha256']
        if split not in ('train','validation','test') or type(n) is not int or n<1 or body in seen:
            raise ValueError('Invalid or duplicate split metadata')
        seen.add(body);counts[split]+=1;tokens[split]+=n
    streams={s:tokens[s]+counts[s] for s in ('train','validation','test')}
    missing=[s for s in ('validation','test') if counts[s]==0 or streams[s]<2]
    return dict(status='needs_heldout' if missing else 'passed',missing_splits=missing,
                documents_by_split=dict(counts),body_tokens_by_split=dict(tokens),stream_tokens_by_split=streams,
                sequence_length=sequence_length,heldout_padded_samples={s:max(0,math.ceil((streams[s]-1)/(sequence_length-1))) for s in ('validation','test')},
                heldout_target_is_not_a_quality_threshold=True,train_split_reassigned=False)

def check_parquet(path,sequence_length):
    import pyarrow.parquet as pq
    rows=(r for b in pq.ParquetFile(path).iter_batches(batch_size=512,columns=['split','own_text_tokens','normalized_sha256']) for r in b.to_pylist())
    return split_report(rows,sequence_length)
