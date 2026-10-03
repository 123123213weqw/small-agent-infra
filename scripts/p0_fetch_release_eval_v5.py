#!/usr/bin/env python3
"""Fetch only checksum-pinned evaluation references. NEVER training input."""
import argparse, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.supply_control import file_sha
from p0_code_production_pilot_v4 import publish
FILES=[
 ('humaneval_test','openai/openai_humaneval','7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544','openai_humaneval/test-00000-of-00001.parquet','2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6'),
 ('mbpp_test','google-research-datasets/mbpp','4bb6404fdc6cacfda99d4ac4205087b89d32030c','sanitized/test-00000-of-00001.parquet','e9e9efa2c0d59ef5e55537a9d126b8f875d5ac010a8d75628d76824884e15850'),
 ('mbpp_validation','google-research-datasets/mbpp','4bb6404fdc6cacfda99d4ac4205087b89d32030c','sanitized/validation-00000-of-00001.parquet','27e065fcab3c863959933328a7fdbf404e1bcb5464b1be6fe0dcd9530e420204'),
 ('gsm8k_test','openai/gsm8k','740312add88f781978c0658806c59bc2815b9866','main/test-00000-of-00001.parquet','ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59')]
def main():
 import requests
 from p0_screen_fineweb_wave2_eval import REFERENCE_FILES
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--old-reference',type=Path,required=True);p.add_argument('--ceval-reference',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);items=[]
 for task,repo,rev,rel,sha in FILES:
  dest=a.output/task/Path(rel).name;dest.parent.mkdir(exist_ok=True)
  if not dest.exists():
   r=requests.get(f'https://hf-mirror.com/datasets/{repo}/resolve/{rev}/{rel}',timeout=120);r.raise_for_status()
   tmp=dest.with_suffix('.tmp');tmp.write_bytes(r.content)
   if file_sha(tmp)!=sha:tmp.unlink();raise ValueError('Evaluation SHA mismatch')
   os.replace(tmp,dest)
  if file_sha(dest)!=sha:raise ValueError('Evaluation file changed')
  items.append(dict(task=task,path=str(dest.resolve()),sha256=sha,repository=repo,revision=rev))
 for task,(rel,sha) in REFERENCE_FILES.items():
  path=a.old_reference/rel
  if file_sha(path)!=sha:raise ValueError('Old evaluation hash differs')
  items.append(dict(task=task,path=str(path.resolve()),sha256=sha))
 m=json.loads((a.ceval_reference/'manifest.json').read_text())
 if m.get('training_eligible') is not False or m['source_revision']!='617524a00b307ff6f9933702f724131fe12ca7ce':raise ValueError('C-Eval binding missing')
 for i in m['files']:
  path=a.ceval_reference/i['file']
  if file_sha(path)!=i['sha256']:raise ValueError('C-Eval changed')
  items.append(dict(task='ceval_validation',path=str(path.resolve()),sha256=i['sha256'],revision=m['source_revision']))
 publish(a.output/'registry.json',dict(version=5,training_eligible=False,files=items,scope='registered evaluation references only'))
 print(json.dumps(dict(stage='evaluation_registry_frozen',files=len(items),tasks=sorted({x['task'] for x in items}))))
if __name__=='__main__':main()
