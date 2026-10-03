"""Measured internal-training gates. A transport-success flag is never approval."""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib,heapq,json,math,os,re,sqlite3
from pathlib import Path
from .supply_control import file_sha,digest


def sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 data=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
 if path.exists():
  if path.read_text()!=data:raise ValueError('Immutable output differs: '+str(path))
  return
 tmp=path.with_name('.'+path.name+'.tmp');tmp.write_text(data)
 with tmp.open('rb') as f:os.fsync(f.fileno())
 os.replace(tmp,path)

def bound_json(path):
 obj=json.loads(Path(path).read_text())
 for e in obj.get('evidence_files',[]):
  if file_sha(Path(e['path']))!=e['sha256']:raise ValueError('Changed bound evidence')
 return obj

def lexical(text):
 import xxhash
 # Conservatively casefold for cross-domain dedup only. NEVER transform training bodies.
 words=re.findall(r'[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\s]',text.casefold())
 if len(words)<5:return {xxhash.xxh64_intdigest('\x1f'.join(words))} if words else set()
 return {xxhash.xxh64_intdigest('\x1f'.join(words[i:i+5])) for i in range(len(words)-4)}

def ledger_snapshot(database):
 db=sqlite3.connect('file:'+str(Path(database).resolve())+'?mode=ro',uri=True)
 try:
  db.execute('BEGIN');h=hashlib.sha256();count=0
  for r in db.execute('SELECT body_sha,split,domain,tokens FROM documents ORDER BY body_sha'):
   h.update((json.dumps(r,separators=(',',':'))+'\n').encode());count+=1
  return dict(documents=count,document_ledger_sha256=h.hexdigest())
 finally:db.close()

def inventory_files(database,legacy_receipt):
 """Automatic all-accounted-inventory discovery, not a manually selected comparison set."""
 legacy=bound_json(legacy_receipt)
 files={e['path']:e['sha256'] for e in legacy['evidence_files'] if e['path'].endswith('.parquet')}
 db=sqlite3.connect('file:'+str(Path(database).resolve())+'?mode=ro',uri=True)
 try:
  for batch, in db.execute('SELECT batch_id FROM releases'):
   # Every release must have a frozen textual Parquet in its freeze-stage evidence.
   r=db.execute('SELECT files_json FROM stage_receipts WHERE batch_id=? AND stage=4',(batch,)).fetchone()
   if r is None:raise ValueError('Released source has no frozen-text evidence')
   pf={p:h for p,h in json.loads(r[0]).items() if p.endswith('.parquet')}
   if not pf:raise ValueError('Released inventory text missing; no partial global scope allowed')
   files.update(pf)
 finally:db.close()
 return [dict(path=p,sha256=h) for p,h in sorted(files.items())]

def build_inventory_index(database,legacy_receipt,output):
 """Incremental file-level immutable sketches and exact SHA coverage, CPU only."""
 import pyarrow as pa
 import pyarrow.parquet as pq
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 inputs=inventory_files(database,legacy_receipt);snapshot=ledger_snapshot(database)
 identity=dict(version=5,algorithm='casefold-lexical-5-bottom64-v1',inputs=inputs,ledger=snapshot)
 generation=output/digest(identity);generation.mkdir(exist_ok=True)
 write(generation/'INPUTS.json',identity)
 coverage=generation/'coverage.sqlite';db=sqlite3.connect(coverage)
 db.execute('CREATE TABLE IF NOT EXISTS docs(sha TEXT PRIMARY KEY,split TEXT)');db.commit()
 receipts=[]
 for e in inputs:
  source=Path(e['path']);key=digest(e);dest=output/(key+'.sketch.parquet');marker=output/(key+'.done.json')
  if marker.exists():
   rec=json.loads(marker.read_text())
   if rec['input']!=e or file_sha(dest)!=rec['sha256']:raise ValueError('Index changed')
  else:
   if file_sha(source)!=e['sha256']:raise ValueError('Inventory source changed')
   tmp=dest.with_suffix('.tmp');tmp.unlink(missing_ok=True)
   schema=pa.schema([('body_sha256',pa.string()),('split',pa.string()),('row',pa.int64()),('shingle_count',pa.int32()),('sketch',pa.list_(pa.uint64()))])
   writer=pq.ParquetWriter(tmp,schema,compression='zstd');n=0;pending=[]
   try:
    for b in pq.ParquetFile(source).iter_batches(batch_size=64,columns=['text','normalized_sha256','split']):
     for r in b.to_pylist():
      if sha(r['text'])!=r['normalized_sha256']:raise ValueError('Inventory body hash differs')
      values=lexical(r['text']);pending.append(dict(body_sha256=r['normalized_sha256'],split=r['split'],row=n,shingle_count=len(values),sketch=heapq.nsmallest(64,values)));n+=1
     writer.write_table(pa.Table.from_pylist(pending,schema=schema));pending.clear()
   finally:writer.close()
   if file_sha(source)!=e['sha256']:raise ValueError('Inventory changed during indexing')
   os.replace(tmp,dest);rec=dict(input=e,path=str(dest),sha256=file_sha(dest),documents=n);write(marker,rec)
   print('INDEXED_INVENTORY_FILE',source.name,n,flush=True)
  for b in pq.ParquetFile(dest).iter_batches(batch_size=8192,columns=['body_sha256','split']):
   for r in b.to_pylist():
    previous=db.execute('SELECT split FROM docs WHERE sha=?',(r['body_sha256'],)).fetchone()
    if previous and previous[0]!=r['split']:raise ValueError('Reference split conflict')
    db.execute('INSERT OR IGNORE INTO docs VALUES (?,?)',(r['body_sha256'],r['split']))
   db.commit()
  receipts.append(rec)
 db.execute('ATTACH DATABASE ? AS supply',(str(Path(database).resolve()),))
 missing=db.execute('SELECT COUNT(*) FROM supply.documents d LEFT JOIN docs c ON d.body_sha=c.sha WHERE c.sha IS NULL OR d.split!=c.split').fetchone()[0]
 extra=db.execute('SELECT COUNT(*) FROM docs c LEFT JOIN supply.documents d ON c.sha=d.body_sha WHERE d.body_sha IS NULL').fetchone()[0]
 unique=db.execute('SELECT COUNT(*) FROM docs').fetchone()[0];db.close()
 if missing or extra or unique!=snapshot['documents'] or ledger_snapshot(database)!=snapshot:raise ValueError('Global inventory coverage/snapshot changed')
 rec=dict(stage='complete_registered_inventory_index',identity=identity,unique_documents=unique,missing_documents=0,extra_documents=0,files=receipts,
   scope='all legacy and READY train/validation/test documents; unreviewed queues are not READY inventory',algorithm='approximate lexical retrieval, not semantic dedup')
 write(generation/'manifest.json',rec);return generation/'manifest.json'

class ParquetPointer:
 def __init__(self,path):
  import pyarrow.parquet as pq
  self.pf=pq.ParquetFile(path);self.starts=[];n=0
  for i in range(self.pf.num_row_groups):self.starts.append(n);n+=self.pf.metadata.row_group(i).num_rows
  self.cached=None;self.rows=None
 def text(self,n):
  from bisect import bisect_right
  g=bisect_right(self.starts,n)-1
  if self.cached!=g:self.rows=self.pf.read_row_group(g,columns=['text']).column(0).to_pylist();self.cached=g
  return self.rows[n-self.starts[g]]

def global_near(rows,index_manifest,policy,database):
 import pyarrow.parquet as pq
 ref=json.loads(Path(index_manifest).read_text())
 if ref.get('stage')!='complete_registered_inventory_index' or ref['identity']['ledger']!=ledger_snapshot(database):raise ValueError('Stale/incomplete inventory index')
 sets=[lexical(r['text']) for r in rows];sketches=[heapq.nsmallest(policy['sketch_size'],v) for v in sets]
 inverted=defaultdict(list);excluded={};checks=Counter();retrieval=Counter()
 for i,s in enumerate(sketches):
  for h in s:inverted[h].append(i)
 for f in ref['files']:
  if file_sha(Path(f['path']))!=f['sha256'] or file_sha(Path(f['input']['path']))!=f['input']['sha256']:raise ValueError('Reference changed')
  reader=ParquetPointer(f['input']['path'])
  for b in pq.ParquetFile(f['path']).iter_batches(batch_size=512):
   for r in b.to_pylist():
    counts=Counter(i for h in r['sketch'] for i in inverted.get(h,[]) if i not in excluded)
    for i,c in counts.items():
     if rows[i]['normalized_sha256']==r['body_sha256']:excluded[i]=dict(reason='global_exact_duplicate',reference=r['body_sha256']);continue
     a=len(sets[i]);bcount=r['shingle_count']
     if min(a,bcount)<policy['threshold']*max(a,bcount):continue
     if c<max(1,math.ceil(min(len(sketches[i]),len(r['sketch']))*policy['minimum_shared_fraction'])):continue
     retrieval[i]+=1
     if retrieval[i]>policy['maximum_pair_checks_per_document']:excluded[i]=dict(reason='dense_near_retrieval_hold');continue
     oldtext=reader.text(r['row'])
     if sha(oldtext)!=r['body_sha256']:raise ValueError('Reference pointer/body mismatch')
     values=lexical(oldtext);sim=len(sets[i]&values)/max(1,len(sets[i]|values));checks['exact_pairs']+=1
     if sim>=policy['threshold']:excluded[i]=dict(reason='global_near_duplicate',reference=r['body_sha256'],reference_split=r['split'],jaccard=sim)
 # Within-batch, prefer held-out before train. Do not move an existing held-out document to train.
 kept=[];prior=defaultdict(list)
 for i in sorted(range(len(rows)),key=lambda i:(rows[i]['split']=='train',rows[i]['normalized_sha256'])):
  if i in excluded:continue
  matches=Counter(j for h in sketches[i] for j in prior[h]);eligible=[]
  for j,c in matches.items():
   if min(len(sets[i]),len(sets[j]))<policy['threshold']*max(len(sets[i]),len(sets[j])):continue
   if c>=max(1,math.ceil(min(len(sketches[i]),len(sketches[j]))*policy['minimum_shared_fraction'])):eligible.append(j)
  if len(eligible)>policy['maximum_pair_checks_per_document']:excluded[i]=dict(reason='dense_local_retrieval_hold');continue
  for j in eligible:
   sim=len(sets[i]&sets[j])/max(1,len(sets[i]|sets[j]));checks['exact_pairs']+=1
   if sim>=policy['threshold']:excluded[i]=dict(reason='local_near_duplicate',reference=rows[j]['normalized_sha256'],jaccard=sim);break
  if i not in excluded:
   kept.append(rows[i])
   for h in sketches[i]:prior[h].append(i)
 if ledger_snapshot(database)!=ref['identity']['ledger']:raise ValueError('Inventory changed during dedup; rescreen before release')
 report=dict(status='passed',input_documents=len(rows),retained_documents=len(kept),excluded_documents=len(excluded),exact_jaccard_pairs=checks['exact_pairs'],
   inventory_snapshot=ref['identity']['ledger'],index_manifest_sha256=file_sha(Path(index_manifest)),retrieval='bottom64 overlap>=half plus exact Jaccard>=0.90; casefold only for dedup',
   approximate_retrieval=True,semantic_duplicates_certified=False,exclusions=[dict(body_sha256=rows[i]['normalized_sha256'],**r) for i,r in excluded.items()])
 return kept,report


def normalized_words(text):return tuple(re.findall(r'\w+',text.casefold(),flags=re.UNICODE))
def cjk_text(text):return ''.join(re.findall(r'[\u3400-\u4dbf\u4e00-\u9fff]',text))

class EvalScreen:
 def __init__(self,registry,policy):
  import pyarrow.parquet as pq
  from .eval_index_v5 import AnchorAutomaton
  m=json.loads(Path(registry).read_text());files=m['files'];tasks={f['task'] for f in files}
  if m.get('training_eligible') is not False or not set(policy['required_tasks'])<=tasks:raise ValueError('Incomplete benchmark registry')
  self.phrases=[];self.grams=defaultdict(set);self.chinese=defaultdict(set);self.registry_sha=file_sha(Path(registry));self.tasks=sorted(tasks);self.policy=policy
  self.reference_rows=Counter();self.short_parts=Counter()
  for f in files:
   if file_sha(Path(f['path']))!=f['sha256']:raise ValueError('Evaluation reference changed')
   task=f['task']
   for n,r in enumerate(pq.read_table(f['path']).to_pylist()):
    self.reference_rows[task]+=1
    if task.startswith('arc'):parts=[r['question']]
    elif task.startswith('hellaswag'):parts=[r['ctx']+' '+x for x in r['endings']]
    elif task.startswith('ceval'):parts=[r['question']]+[r['question']+' '+str(r[k]) for k in 'ABCD']
    elif task.startswith('humaneval'):parts=[r['prompt'],r['canonical_solution'],r['test']]
    elif task.startswith('mbpp'):parts=[r.get('prompt',r.get('text','')),r['code'],*r['test_list']]
    else:parts=[r['question'],r['answer']]
    for j,t in enumerate(parts):
     w=normalized_words(t);tag=(task,str(n)+':'+str(j))
     if len(w)>=policy['minimum_full_phrase_words']:self.phrases.append((*tag,w))
     else:self.short_parts[task]+=1
     size=policy['word_ngram']
     for k in range(max(0,len(w)-size+1)):self.grams[sha('\x1f'.join(w[k:k+size]))].add(task)
     c=cjk_text(t);size=policy['cjk_ngram']
     for k in range(max(0,len(c)-size+1)):self.chinese[sha(c[k:k+size])].add(task)
  self.automaton=AnchorAutomaton(self.phrases)
 def matches(self,text):
  words=normalized_words(text);hits={t for t,_,_ in self.automaton.find(words)};size=self.policy['word_ngram']
  for k in range(max(0,len(words)-size+1)):hits.update(self.grams.get(sha('\x1f'.join(words[k:k+size])),[]))
  c=cjk_text(text);size=self.policy['cjk_ngram']
  for k in range(max(0,len(c)-size+1)):hits.update(self.chinese.get(sha(c[k:k+size]),[]))
  return sorted(hits)
 def filter(self,rows):
  retained=[];hits=[]
  for r in rows:
   found=self.matches(r['text'])
   if found:hits.append(dict(body_sha256=r['normalized_sha256'],reason='registered_eval_contamination_hold',tasks=found))
   else:retained.append(r)
  return retained,dict(status='passed',registry_sha256=self.registry_sha,registered_tasks=self.tasks,reference_rows=dict(self.reference_rows),input_documents=len(rows),retained_documents=len(retained),exclusions=hits,
    scope='full normalized phrases plus word13/cjk30 ngrams; all batch splits scanned; short/answer-only/paraphrase leakage not certified',short_reference_parts_not_full_phrase_screened=dict(self.short_parts))


def quality_scope(source_audit,reference,reference_batch,policy,batch_audit=None):
 """Fail-closed quality certificate with explicit reference and source strata."""
 from .store import Batch
 from .core import quality_pass
 from datetime import datetime,timezone
 import time
 ref=bound_json(reference);labels=json.loads((Path(reference).parent/'labels.json').read_text())
 if ref.get('training_eligible') is not False or ref.get('independently_verified_positive_code_tests') is not True:raise ValueError('External measured reference required')
 def review(path):
  b=Batch(Path(path))
  try:
   rows={r[0]:json.loads(r[1]) if r[1] else None for r in b.db.execute("SELECT document_id,annotation FROM jobs WHERE judge='deepseek' AND status='valid'")}
   return rows,b.docs,b.policy,b.state_fingerprint(),b.manifest_hash
  finally:b.close()
 annotations,docs,review_policy,state,manifest=review(reference_batch);counts=Counter();holds=[]
 for l in labels:
  a=annotations.get(l['document_id']);kind=l['expected'];counts[kind+'_total']+=1
  if a is None:counts[kind+'_invalid']+=1;continue
  if quality_pass(a,review_policy):counts[kind+'_model_keep']+=1
  else:counts[kind+'_model_reject_or_uncertain']+=1
 for k in ('keep','drop'):
  if counts[k+'_total']<policy['minimum_reference_per_class']:holds.append('insufficient_external_reference_'+k)
  if (counts[k+'_total']-counts[k+'_invalid'])/max(1,counts[k+'_total'])<policy['minimum_valid_rate']:holds.append('reference_output_coverage_'+k)
 if counts['drop_model_keep']>policy['maximum_reference_false_keep']:holds.append('controlled_defects_false_keep')
 if counts['keep_model_keep']/max(1,counts['keep_total'])<policy['minimum_reference_keep_rate']:holds.append('external_positive_keep_rate')
 plan=bound_json(Path(source_audit)/'SAMPLE_PLAN.json');source,source_docs,source_policy,source_state,source_manifest=review(Path(source_audit)/'batch')
 if not plan.get('source_complete_in_plan'):holds.append('partial_source_calibration')
 if plan['request_identity']['prompt_sha256']!=file_sha(Path(reference_batch)/'prompt.txt'):holds.append('reference_source_prompt_differs')
 if set(source_docs) & set(d['document_id'] for d in labels):holds.append('reference_source_exact_overlap')
 if time.time()-ref['created_unix']>86400*policy['maximum_calibration_age_days']:holds.append('reference_calibration_expired')
 if file_sha(Path(reference_batch)/'judges.json')!=file_sha(Path(source_audit)/'batch/judges.json'):holds.append('reference_source_judge_differs')
 # Scope cannot silently extend Python references to all languages.
 strata=defaultdict(Counter);bad_ids=set()
 for d in source_docs.values():
  h=d['sampling_stratum']
  if not h.endswith('/python'):continue
  a=source.get(d['document_id']);strata[h]['selected']+=1
  if a is not None:
   strata[h]['valid']+=1
   if quality_pass(a,source_policy):strata[h]['keep']+=1
   else:bad_ids.add(d['document_id'])
  else:bad_ids.add(d['document_id'])
 approved=[]
 if not holds:
  for h,c in strata.items():
   if c['selected']>=policy['minimum_source_samples_per_stratum'] and c['valid']/c['selected']>=policy['minimum_valid_rate'] and c['keep']/c['selected']>=policy['minimum_joint_keep_rate']:approved.append(h)
 if not approved:holds.append('no_calibrated_python_stratum')
 if policy['require_batch_audit']:
  if batch_audit is None:holds.append('batch_rolling_audit_missing')
  else:
   ba=bound_json(batch_audit)
   if ba.get('status')!='passed' or ba.get('source_manifest_sha256')!=source_manifest or ba.get('review_state_sha256')!=source_state:holds.append('batch_rolling_audit_missing_stale_or_failed')
   else:
    # Failure never becomes approval. Exclude the entire failed stratum instead
    # of discarding an error label and calling the original scope passed.
    approved=[h for h in approved if h in ba.get('approved_strata',[])]
    if not approved:holds.append('no_joint_source_and_batch_approved_stratum')
 return dict(status='passed' if not holds else 'hold',holds=holds,source_id=plan['request_identity']['source_id'],approved_language='python',approved_strata=approved,
   reference_counts=dict(counts),source_strata={k:dict(v) for k,v in strata.items()},known_audit_bad_body_sha256=sorted(bad_ids),
   source_manifest_sha256=source_manifest,source_review_state_sha256=source_state,reference_manifest_sha256=manifest,reference_review_state_sha256=state,
   reference_path=str(Path(reference).resolve()),reference_sha256=file_sha(Path(reference)),reference_scope=ref['reference_type'],
   natural_corpus_factual_error_rate_certified=False,source_level_only=True,created_unix=ref['created_unix'],expires_after_days=policy['maximum_calibration_age_days'])


def rolling_audit(run,source_audit,minimum_valid=.85,minimum_keep=.75):
 from .store import Batch
 from .core import quality_pass
 run=Path(run);plan=json.loads((run/'SAMPLE_PLAN.json').read_text())
 if file_sha(Path(plan['parquet_path']))!=plan['parquet_sha256'] or file_sha(run/'input.jsonl')!=plan['input_sha256']:raise ValueError('Batch audit body binding changed')
 b=Batch(run/'batch');s=Batch(Path(source_audit)/'batch');strata=defaultdict(Counter);bad=[]
 try:
  if s.manifest_hash!=plan['source_manifest_sha256'] or s.state_fingerprint()!=plan['review_state_sha256']:raise ValueError('Source calibration changed')
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
