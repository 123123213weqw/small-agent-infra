#!/usr/bin/env python3
"""Expose ONLY ledger-committed and physically verified READY packs to training."""
import argparse,json,sqlite3,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from small_agent.curation.supply_control import file_sha
from small_agent.data.sharded_format import load_sharded_manifest

def catalog(database):
 db=sqlite3.connect('file:'+str(Path(database).resolve())+'?mode=ro',uri=True);entries=[]
 try:
  db.execute('BEGIN')
  for batch,receipt_sha,tokens in db.execute('SELECT batch_id,receipt_sha,train_tokens FROM releases ORDER BY batch_id'):
   stages=list(db.execute('SELECT stage,files_json FROM stage_receipts WHERE batch_id=? ORDER BY stage',(batch,)))
   if [r[0] for r in stages]!=list(range(7)):raise ValueError('Incomplete release stages')
   names={}
   for stage,raw in stages:
    for name,h in json.loads(raw).items():
     if file_sha(Path(name))!=h:raise ValueError('READY stage evidence changed')
     names[Path(name).name]=Path(name)
   pack=names['manifest.json'];root=pack.parent.parent;ready=root/'READY_INTERNAL.json'
   if not ready.exists():continue # interrupted after ledger commit, before READY publication
   if file_sha(ready)!=receipt_sha:raise ValueError('Published READY receipt differs from committed ledger')
   r=json.loads(ready.read_text());m,_=load_sharded_manifest(pack.parent,verify_hashes=True)
   if r['batch_id']!=batch or r['status']!='READY_INTERNAL' or r['unique_train_body_tokens']!=tokens or m['license_gate'].get('training_eligible') is not True:raise ValueError('READY contract differs')
   if file_sha(pack)!=r['pack_manifest']['sha256'] or m['tokenizer_sha256']!=r['tokenizer_sha256']:raise ValueError('Pack receipt binding differs')
   entries.append(dict(batch_id=batch,manifest=str(pack),manifest_sha256=file_sha(pack),ready_receipt=str(ready),ready_sha256=receipt_sha,tokenizer_sha256=r['tokenizer_sha256'],train_body_tokens=tokens,scope=r.get('scope'),distribution_eligible=False))
  return dict(version=5,stage='verified_ready_training_catalog',packs=entries,unique_train_body_tokens=sum(x['train_body_tokens'] for x in entries),legacy_imports_not_recertified=True,unreviewed_candidates_excluded=True,training_started=False)
 finally:db.close()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--database',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();r=catalog(a.database)
 if a.output:
  a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(r,ensure_ascii=False,sort_keys=True,indent=2)+'\n');tmp.replace(a.output)
 print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
