#!/usr/bin/env python3
"""v15: unchanged v6 decisions, Arrow scalar pointers and bounded lexical cache."""
from collections import Counter,defaultdict,OrderedDict
import heapq,json,math,sqlite3,struct
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.release_gates_v5 import lexical,sha,ledger_snapshot,ParquetPointer
from small_agent.curation.supply_control import file_sha
from p0_parallel_cpu_v13 import executor,ordered_map
_context=None

class ArrowPointer:
    """Keep one Arrow row group; do not convert every reference text to Python."""
    def __init__(self,path):
        import pyarrow.parquet as pq
        self.pf=pq.ParquetFile(path);self.starts=[];n=0
        for i in range(self.pf.num_row_groups):
            self.starts.append(n);n+=self.pf.metadata.row_group(i).num_rows
        self.cached=None;self.rows=None
    def text(self,n):
        from bisect import bisect_right
        g=bisect_right(self.starts,n)-1
        if g<0:raise ValueError('Invalid reference row')
        if self.cached!=g:
            self.rows=self.pf.read_row_group(g,columns=['text'],use_threads=False).column(0)
            self.cached=g
        return self.rows[n-self.starts[g]].as_py()


class ReadOnlyIndex:
    def __init__(self,root):
        self.root=Path(root)
        self.db=sqlite3.connect('file:'+str(self.root/'postings.sqlite')+'?mode=ro',uri=True)
        self.db.execute('PRAGMA cache_size=-32768')
        self.db.execute('PRAGMA query_only=ON')
        self.readers=OrderedDict();self.lexical_cache=OrderedDict();self.cached_shingles=0;self.frequency_cache=OrderedDict()
    def close(self):self.db.close()


def exact_anchor_query(ctx,sketch,n,policy,original_query,original_params):
    """Lossless pigeonhole retrieval: if >=m shared hashes, k-m+1 anchors must intersect.

    Rarity only chooses anchors, never a keep/drop decision. Bounded frequency
    hints and a bounded candidate probe fall back to the FULL original query.
    """
    fallback=lambda:ctx.db.execute(original_query,original_params).fetchall()
    if not policy.get('_anchor_shape_verified') or policy['sketch_size']!=64 or policy['minimum_shared_fraction']!=.5:
        return fallback()
    lower=math.ceil(n*policy['threshold']);upper=math.floor(n/policy['threshold'])
    m=max(1,math.ceil(min(len(sketch),min(64,lower))*policy['minimum_shared_fraction']))
    count=len(sketch)-m+1
    ranked=[]
    for h in sketch:
        key=struct.pack('<Q',h)
        if key not in ctx.frequency_cache:
            # 129 means truncated/possibly dense: never pretend it is an upper bound.
            f=len(ctx.db.execute('SELECT d FROM postings WHERE h=? LIMIT 129',(key,)).fetchall())
            ctx.frequency_cache[key]=f
            if len(ctx.frequency_cache)>65536:ctx.frequency_cache.popitem(last=False)
        ranked.append((ctx.frequency_cache[key],key))
    chosen=sorted(ranked)[:count]
    if not chosen or chosen[-1][0]>=129:return fallback()
    keys=[h for _,h in chosen]
    sql='SELECT DISTINCT p.d FROM postings p JOIN docs d ON d.id=p.d WHERE p.h IN ('+','.join('?' for _ in keys)+') AND d.n>=? AND d.n<=? LIMIT 8193'
    ids=[x[0] for x in ctx.db.execute(sql,keys+[lower,upper])]
    if len(ids)>8192:return fallback()
    if not ids:return []
    # Materialized ID set is a SUPerset of every old shared-vote match. Count
    # all original sketch votes and retain the old HAVING, LIMIT and ID order.
    sql='WITH candidates(id) AS MATERIALIZED (SELECT value FROM json_each(?))\nSELECT d.id,d.body,d.split,d.source,d.rowno,d.n,d.k,COUNT(*) c\nFROM candidates c CROSS JOIN docs d CROSS JOIN postings p\nWHERE d.id=c.id AND p.d=d.id AND p.h IN ('+','.join('?' for _ in sketch)+') GROUP BY d.id HAVING c>=MAX(1,CAST((MIN(d.k,?)*?+0.999999) AS INTEGER)) ORDER BY d.id LIMIT ?'
    args=[json.dumps(ids)]+[struct.pack('<Q',h) for h in sketch]+[len(sketch),policy['minimum_shared_fraction'],policy['maximum_pair_checks_per_document']+1]
    return ctx.db.execute(sql,args).fetchall()


def global_chunk(task):
    global _context
    root,rows,policy=task
    if _context is None or str(_context.root)!=root:
        if _context is not None:_context.close()
        _context=ReadOnlyIndex(root)
    self=_context
    excluded={};checks=0;sets=[];sketches=[]
    for i,r in enumerate(rows):
        values=lexical(r['text']);sketch=heapq.nsmallest(policy['sketch_size'],values);sets.append(values);sketches.append(sketch)
        exact=self.db.execute('SELECT split FROM docs WHERE body=?',(bytes.fromhex(r['normalized_sha256']),)).fetchone()
        if exact:excluded[i]=dict(reason='global_exact_duplicate',reference_split=exact[0]);continue
        if not sketch:continue
        needed=max(1,math.ceil(len(sketch)*policy['minimum_shared_fraction']))
        # Existing short sketches need a smaller threshold; HAVING uses min(k).
        query='''SELECT d.id,d.body,d.split,d.source,d.rowno,d.n,d.k,COUNT(*) c FROM postings p JOIN docs d ON d.id=p.d
                 WHERE p.h IN ('''+','.join('?' for _ in sketch)+''') AND d.n>=? AND d.n<=?
                 GROUP BY d.id HAVING c>=MAX(1,CAST((MIN(d.k,?)*?+0.999999) AS INTEGER)) LIMIT ?'''
        params=[struct.pack('<Q',h) for h in sketch]+[math.ceil(len(values)*policy['threshold']),math.floor(len(values)/policy['threshold']),len(sketch),policy['minimum_shared_fraction'],policy['maximum_pair_checks_per_document']+1]
        matches=exact_anchor_query(self,sketch,len(values),policy,query,params)
        if len(matches)>policy['maximum_pair_checks_per_document']:excluded[i]=dict(reason='dense_near_retrieval_hold');continue
        for _,body,split,sid,pos,n,k,c in matches:
            path=self.db.execute('SELECT path FROM sources WHERE id=?',(sid,)).fetchone()[0]
            if path not in self.readers:
                self.readers[path]=ArrowPointer(path)
                if len(self.readers)>2:self.readers.popitem(last=False)
            self.readers.move_to_end(path);text=self.readers[path].text(pos)
            if sha(text)!=body.hex():raise ValueError('Reference body pointer changed')
            # Hash is still checked on EVERY access above; no sampled integrity check.
            key=body.hex()
            if key in self.lexical_cache:
                other=self.lexical_cache[key];self.lexical_cache.move_to_end(key)
            else:
                other=lexical(text)
                if len(other)<=2000000:
                    self.lexical_cache[key]=other;self.cached_shingles+=len(other)
                    while len(self.lexical_cache)>8192 or self.cached_shingles>2000000:
                        _,evicted=self.lexical_cache.popitem(last=False);self.cached_shingles-=len(evicted)
            sim=len(values&other)/max(1,len(values|other));checks+=1
            if sim>=policy['threshold']:excluded[i]=dict(reason='global_near_duplicate',reference=body.hex(),reference_split=split,jaccard=sim);break
    import numpy as np
    return [(np.asarray(list(v),dtype='<u8').tobytes() if i not in excluded else b'',
             sketches[i],excluded.get(i)) for i,v in enumerate(sets)],checks

class ParallelNearIndex(ReadOnlyIndex):
    def filter(self,rows,index_manifest,policy,database,workers=8,progress=None):
        ref=json.loads(Path(index_manifest).read_text());snapshot=ledger_snapshot(database)
        stored=self.db.execute("SELECT v FROM meta WHERE k='ledger'").fetchone()
        if ref['stage']!='complete_registered_disk_index' or ref['ledger']!=snapshot or not stored or json.loads(stored[0])!=snapshot:raise ValueError('Stale disk index')
        for e in ref['inputs']:
            if file_sha(Path(e['path']))!=e['sha256']:raise ValueError('Reference changed')
        # Every reference has k=min(64,n), so shared-vote candidates cannot
        # avoid all k-m+1 anchor hashes. Invalid shape disables acceleration.
        valid_shape=self.db.execute('SELECT 1 FROM docs WHERE k!=MIN(64,n) OR n<0 LIMIT 1').fetchone() is None
        policy=dict(policy,_anchor_shape_verified=valid_shape)
        import numpy as np
        excluded={};checks=0;sets=[];sketches=[]
        tasks=((str(self.root),rows[i:i+32],policy) for i in range(0,len(rows),32))
        with executor(workers) as pool:
            for chunk,n in ordered_map(pool,global_chunk,tasks,workers*2):
                checks+=n
                for values,sketch,reason in chunk:
                    i=len(sets);sets.append(set(np.frombuffer(values,dtype='<u8').tolist()));sketches.append(sketch)
                    if reason is not None:excluded[i]=reason
                if progress:progress(dict(phase="global_near",completed_documents=len(sets),total_documents=len(rows),exact_jaccard_pairs=checks))
        kept=[];self.retained_sketch_rows=[];prior=defaultdict(list)
        for i in sorted(range(len(rows)),key=lambda i:(rows[i]['split']=='train',rows[i]['normalized_sha256'])):
            if i in excluded:continue
            counts=Counter(j for h in sketches[i] for j in prior[h]);candidates=[]
            for j,c in counts.items():
                if min(len(sets[i]),len(sets[j]))<policy['threshold']*max(len(sets[i]),len(sets[j])):continue
                if c>=max(1,math.ceil(min(len(sketches[i]),len(sketches[j]))*policy['minimum_shared_fraction'])):candidates.append(j)
            if len(candidates)>policy['maximum_pair_checks_per_document']:excluded[i]=dict(reason='dense_local_retrieval_hold');continue
            for j in candidates:
                sim=len(sets[i]&sets[j])/max(1,len(sets[i]|sets[j]));checks+=1
                if sim>=policy['threshold']:excluded[i]=dict(reason='local_near_duplicate',reference=rows[j]['normalized_sha256'],jaccard=sim);break
            if i not in excluded:
                kept.append(rows[i])
                self.retained_sketch_rows.append(dict(body_sha256=rows[i]['normalized_sha256'],split=rows[i]['split'],row=len(kept)-1,shingle_count=len(sets[i]),sketch=sketches[i]))
                for h in sketches[i]:prior[h].append(i)
        if ledger_snapshot(database)!=snapshot:raise ValueError('Ledger changed during filtering')
        for e in ref['inputs']:
            if file_sha(Path(e['path']))!=e['sha256']:raise ValueError('Reference changed during filtering')
        return kept,dict(status='passed',input_documents=len(rows),retained_documents=len(kept),excluded_documents=len(excluded),exact_jaccard_pairs=checks,
            inventory_snapshot=snapshot,index_manifest_sha256=file_sha(index_manifest),retrieval='disk indexed bottom64 shared>=half; same v5 exact Jaccard and dense hold thresholds',
            approximate_retrieval=True,semantic_duplicates_certified=False,exclusions=[dict(body_sha256=rows[i]['normalized_sha256'],**x) for i,x in excluded.items()])
