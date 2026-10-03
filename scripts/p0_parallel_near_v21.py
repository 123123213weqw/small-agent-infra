#!/usr/bin/env python3
"""Same v6 global/local near decisions. Parallel read-only global queries, ordered local merge."""
from collections import Counter,defaultdict,OrderedDict
import heapq,json,math,sqlite3,struct
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.release_gates_v5 import lexical,sha,ledger_snapshot,ParquetPointer
from small_agent.curation.supply_control import file_sha
from p0_parallel_cpu_v13 import executor,ordered_map
_context=None

class ReadOnlyIndex:
    def __init__(self,root):
        self.root=Path(root)
        self.db=sqlite3.connect('file:'+str(self.root/'postings.sqlite')+'?mode=ro',uri=True)
        self.db.execute('PRAGMA cache_size=-32768')
        self.db.execute('PRAGMA query_only=ON')
        self.readers=OrderedDict()
    def close(self):self.db.close()


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
        matches=self.db.execute(query,params).fetchall()
        if len(matches)>policy['maximum_pair_checks_per_document']:excluded[i]=dict(reason='dense_near_retrieval_hold');continue
        for _,body,split,sid,pos,n,k,c in matches:
            path=self.db.execute('SELECT path FROM sources WHERE id=?',(sid,)).fetchone()[0]
            if path not in self.readers:
                self.readers[path]=ParquetPointer(path)
                if len(self.readers)>2:self.readers.popitem(last=False)
            self.readers.move_to_end(path);text=self.readers[path].text(pos)
            if sha(text)!=body.hex():raise ValueError('Reference body pointer changed')
            other=lexical(text);sim=len(values&other)/max(1,len(values|other));checks+=1
            if sim>=policy['threshold']:excluded[i]=dict(reason='global_near_duplicate',reference=body.hex(),reference_split=split,jaccard=sim);break
    import numpy as np
    return [(np.asarray(list(v),dtype='<u8').tobytes() if i not in excluded else b'',
             sketches[i],excluded.get(i)) for i,v in enumerate(sets)],checks

class ParallelNearIndex(ReadOnlyIndex):
    def filter(self,rows,index_manifest,policy,database,workers=8):
        ref=json.loads(Path(index_manifest).read_text());snapshot=ledger_snapshot(database)
        stored=self.db.execute("SELECT v FROM meta WHERE k='ledger'").fetchone()
        if ref['stage']!='complete_registered_disk_index' or ref['ledger']!=snapshot or not stored or json.loads(stored[0])!=snapshot:raise ValueError('Stale disk index')
        for e in ref['inputs']:
            if file_sha(Path(e['path']))!=e['sha256']:raise ValueError('Reference changed')
        import numpy as np
        excluded={};checks=0;sets=[];sketches=[]
        tasks=((str(self.root),rows[i:i+32],policy) for i in range(0,len(rows),32))
        with executor(workers) as pool:
            for chunk,n in ordered_map(pool,global_chunk,tasks,workers*2):
                checks+=n
                for values,sketch,reason in chunk:
                    i=len(sets);sets.append(set(np.frombuffer(values,dtype='<u8').tolist()));sketches.append(sketch)
                    if reason is not None:excluded[i]=reason
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
        return kept,dict(status='passed',input_documents=len(rows),retained_documents=len(kept),excluded_documents=len(excluded),exact_jaccard_pairs=checks,
            inventory_snapshot=snapshot,index_manifest_sha256=file_sha(index_manifest),retrieval='disk indexed bottom64 shared>=half; same v5 exact Jaccard and dense hold thresholds',
            approximate_retrieval=True,semantic_duplicates_certified=False,exclusions=[dict(body_sha256=rows[i]['normalized_sha256'],**x) for i,x in excluded.items()])
