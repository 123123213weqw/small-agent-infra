"""Math stock gates and incremental bottom-64 retrieval for large inventories.

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
    from .store import Batch
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

class DiskNearIndex:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.root/'postings.sqlite',timeout=60)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA cache_size=-262144')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS sources(id INTEGER PRIMARY KEY,path TEXT UNIQUE,sha TEXT,complete INTEGER DEFAULT 0);
          CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY,body BLOB UNIQUE,split TEXT,source INTEGER,rowno INTEGER,n INTEGER,k INTEGER);
          CREATE TABLE IF NOT EXISTS postings(h BLOB,d INTEGER,PRIMARY KEY(h,d)) WITHOUT ROWID;
          CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT);
        ''');self.db.commit();self.readers=OrderedDict()
        self.db.create_function('unhex',1,lambda v:bytes.fromhex(v))
    def close(self):self.db.close()
    def update(self,database,legacy_receipt,sketch_root):
        import pyarrow.parquet as pq
        files=inventory_files(database,legacy_receipt);snapshot=ledger_snapshot(database)
        for e in files:
            source=Path(e['path'])
            if file_sha(source)!=e['sha256']:raise ValueError('Inventory source changed')
            old=self.db.execute('SELECT id,sha,complete FROM sources WHERE path=?',(str(source),)).fetchone()
            if old and old[1]!=e['sha256']:raise ValueError('Append-only reference source changed')
            if old and old[2]:continue
            if not old:
                with self.db:self.db.execute('INSERT INTO sources(path,sha) VALUES (?,?)',(str(source),e['sha256']))
            sid=self.db.execute('SELECT id FROM sources WHERE path=?',(str(source),)).fetchone()[0]
            key=digest(e);cached=Path(sketch_root)/(key+'.sketch.parquet');marker=Path(sketch_root)/(key+'.done.json')
            reuse=False
            if marker.exists():
                m=json.loads(marker.read_text())
                reuse=m['input']==e and file_sha(cached)==m['sha256']
            rowno=0
            pf=pq.ParquetFile(cached if reuse else source)
            columns=['body_sha256','split','row','shingle_count','sketch'] if reuse else ['text','normalized_sha256','split']
            for b in pf.iter_batches(batch_size=256,columns=columns):
                with self.db:
                    for r in b.to_pylist():
                        if reuse:h=r['body_sha256'];n=r['shingle_count'];sketch=r['sketch'];pos=r['row']
                        else:
                            h=sha(r['text'])
                            if h!=r['normalized_sha256']:raise ValueError('Reference body differs')
                            values=lexical(r['text']);n=len(values);sketch=heapq.nsmallest(64,values);pos=rowno
                        body=bytes.fromhex(h);old=self.db.execute('SELECT id,split,n,k FROM docs WHERE body=?',(body,)).fetchone()
                        if old:
                            if old[1:]!=(r['split'],n,len(sketch)):raise ValueError('Index body/split/sketch count conflict')
                        else:
                            cur=self.db.execute('INSERT INTO docs(body,split,source,rowno,n,k) VALUES (?,?,?,?,?,?)',(body,r['split'],sid,pos,n,len(sketch)))
                            self.db.executemany('INSERT INTO postings VALUES (?,?)',[(struct.pack('<Q',v),cur.lastrowid) for v in sketch])
                        rowno+=1
                if rowno%16384<256:print('INDEX_PROGRESS',source.name,rowno,flush=True)
            if file_sha(source)!=e['sha256']:raise ValueError('Source changed during index update')
            with self.db:self.db.execute('UPDATE sources SET complete=1 WHERE id=?',(sid,))
        self.db.execute('ATTACH DATABASE ? AS supply',(str(Path(database).resolve()),))
        try:
            missing=self.db.execute('SELECT COUNT(*) FROM supply.documents d LEFT JOIN docs n ON n.body=unhex(d.body_sha) WHERE n.id IS NULL OR n.split!=d.split').fetchone()[0]
            extra=self.db.execute('SELECT COUNT(*) FROM docs n LEFT JOIN supply.documents d ON d.body_sha=lower(hex(n.body)) WHERE d.body_sha IS NULL').fetchone()[0]
        finally:self.db.execute('DETACH DATABASE supply')
        count=self.db.execute('SELECT COUNT(*) FROM docs').fetchone()[0]
        if missing or extra or count!=snapshot['documents'] or ledger_snapshot(database)!=snapshot:raise ValueError('Index ledger coverage differs/stale')
        state=dict(version=6,stage='complete_registered_disk_index',database=str((self.root/'postings.sqlite').resolve()),ledger=snapshot,inputs=files,
            documents=count,missing_documents=0,extra_documents=0,algorithm='same casefold lexical5 bottom64 overlap>=half then exact Jaccard>=0.90; append-only indexed retrieval',
            checker_sha256=file_sha(Path(__file__)),mutable_append_only_index=True)
        generation=self.root/digest(state);generation.mkdir(exist_ok=True);write(generation/'manifest.json',state)
        with self.db:self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('ledger',json.dumps(snapshot,sort_keys=True)))
        pointer=self.root/'.CURRENT.tmp';pointer.write_text(str((generation/'manifest.json').resolve())+'\n');pointer.replace(self.root/'CURRENT')
        return generation/'manifest.json'
    def filter(self,rows,index_manifest,policy,database):
        ref=json.loads(Path(index_manifest).read_text());snapshot=ledger_snapshot(database)
        stored=self.db.execute("SELECT v FROM meta WHERE k='ledger'").fetchone()
        if ref['stage']!='complete_registered_disk_index' or ref['ledger']!=snapshot or not stored or json.loads(stored[0])!=snapshot:raise ValueError('Stale disk index')
        for e in ref['inputs']:
            if file_sha(Path(e['path']))!=e['sha256']:raise ValueError('Reference changed')
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
        kept=[];prior=defaultdict(list)
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
                for h in sketches[i]:prior[h].append(i)
        if ledger_snapshot(database)!=snapshot:raise ValueError('Ledger changed during filtering')
        return kept,dict(status='passed',input_documents=len(rows),retained_documents=len(kept),excluded_documents=len(excluded),exact_jaccard_pairs=checks,
            inventory_snapshot=snapshot,index_manifest_sha256=file_sha(index_manifest),retrieval='disk indexed bottom64 shared>=half; same v5 exact Jaccard and dense hold thresholds',
            approximate_retrieval=True,semantic_duplicates_certified=False,exclusions=[dict(body_sha256=rows[i]['normalized_sha256'],**x) for i,x in excluded.items()])
