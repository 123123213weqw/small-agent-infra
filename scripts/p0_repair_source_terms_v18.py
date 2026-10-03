#!/usr/bin/env python3
"""Explicit public pinned-card transport repair only. Never resolve paid/quality HOLD."""
import argparse,fcntl,json,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p0_inventory_supply_v11 import terms
from small_agent.curation.supply_control import file_sha,digest
from small_agent.curation.release_gates_v5 import write,bound_json

def repair(cfg,staging):
 root=Path(cfg['root']);reg=json.loads((root/'REGISTRY.json').read_text());out=root/'parallel_v13/source_terms_repair_v18';out.mkdir(parents=True,exist_ok=True);results=[]
 for source in reg['sources']:
  folder=staging/source['id'];receipt=folder/'FETCH.json';card=folder/'dataset_card.md'
  if not receipt.exists():continue
  fetched=json.loads(receipt.read_text());expected='https://hf-mirror.com/datasets/'+source['dataset']+'/resolve/'+source['revision']+'/README.md'
  if fetched['url']!=expected or fetched['dataset']!=source['dataset'] or fetched['revision']!=source['revision'] or fetched['sha256']!=file_sha(card):raise ValueError('Pinned source card transport differs')
  text=card.read_text();match=re.search(r'^license:\s*([^\n]+)',text.lower(),re.M)
  license=match[1].strip().strip('"\'') if match else None
  if license not in ('odc-by','apache-2.0','mit','cc-by-4.0','cc-by-sa-4.0','cc0-1.0'):continue
  dest=root/'terms'/source['id'];dest.mkdir(parents=True,exist_ok=True)
  if (dest/'dataset_card.md').exists():
   if file_sha(dest/'dataset_card.md')!=file_sha(card):raise ValueError('Existing source card differs')
  else:write_bytes=card.read_bytes();tmp=dest/'.card-repair-v18.tmp';tmp.write_bytes(write_bytes);tmp.replace(dest/'dataset_card.md')
  scope=terms(cfg,source);bound_json(dest/'SOURCE_TERMS.json')
  resolved=[]
  for hold in (root/'holds').glob('*'):
   h=json.loads(hold.read_text())
   if h.get('cohort','').split('/')[0]!=source['id'] or h.get('reason')!='Source terms unavailable; no API for unlicensed data':continue
   if h.get('stage')!='EXPLICIT_COHORT_HOLD_NO_AUTO_API_RETRY':raise ValueError('Unknown HOLD schema')
   target=out/('resolved_'+hold.name+'.json')
   action=out/('receipt_'+hold.name+'.json')
   write(action,dict(version=18,action='resolved_network_only_terms_hold_after_explicit_pinned_card_verification',cohort=h['cohort'],old_hold_sha256=file_sha(hold),source_terms_sha256=file_sha(dest/'SOURCE_TERMS.json'),fetch_receipt_sha256=file_sha(receipt),paid_attempts_retried=False,quality_thresholds_changed=False,training_started=False))
   if target.exists():raise ValueError('Prior repair archive exists; inspect instead of overwriting')
   hold.rename(target);resolved.append(h['cohort'])
  results.append(dict(source=source['id'],license=license,resolved_network_holds=resolved))
 result=dict(version=18,results=results,quality_thresholds_changed=False,api_holds_removed=False,training_started=False)
 write(out/('REPAIR_'+digest(result)+'.json'),result);print(json.dumps(result,ensure_ascii=False),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--staging',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text())
 # Explicit repair at a natural queue boundary, never touching an active review.
 with (Path(cfg['root'])/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);repair(cfg,a.staging)
if __name__=='__main__':main()
