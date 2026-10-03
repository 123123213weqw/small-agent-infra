#!/usr/bin/env python3
"""Bind explicit internal-use policy and primary source terms, not individual rights certification."""
import argparse,json,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from small_agent.curation.release_gates_v5 import write
from small_agent.curation.supply_control import file_sha
URLS={
 'dataset_card.txt':'https://huggingface.co/datasets/HuggingFaceTB/finemath/resolve/e92b25a616738fe95dc186b64dfb19f9c8525594/README.md',
 'odc_by_terms.html':'https://opendatacommons.org/licenses/by/1-0/',
 'commoncrawl_terms.html':'https://commoncrawl.org/terms-of-use',
}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--authorize-source-level-internal-use',action='store_true');a=p.parse_args()
 if not a.authorize_source_level_internal_use:raise ValueError('explicit source-level internal-use authorization required')
 if (a.output/'SOURCE_TERMS.json').exists():print('EXISTING_TERMS_SCOPE_REUSED');return
 a.output.mkdir(parents=True,exist_ok=True)
 for name,url in URLS.items():
  dest=a.output/name
  if not dest.exists():
   req=urllib.request.Request(url,headers={'User-Agent':'small-agent-data-provenance/6'})
   with urllib.request.urlopen(req,timeout=60) as response:data=response.read(2000000)
   if not data or len(data)>=2000000:raise ValueError('terms snapshot invalid/oversize')
   dest.write_bytes(data)
 if 'license: odc-by' not in (a.output/'dataset_card.txt').read_text():raise ValueError('pinned dataset license differs')
 write(a.output/'ATTRIBUTION.txt',dict(dataset='HuggingFaceTB/finemath',revision='e92b25a616738fe95dc186b64dfb19f9c8525594',license='ODC-By v1.0 (database)',dataset_url='https://huggingface.co/datasets/HuggingFaceTB/finemath',paper='https://arxiv.org/abs/2502.02737',preserve_original_url=True))
 write(a.output/'SOURCE_TERMS.json',dict(version=6,status='source_level_internal_scope_authorized',source_id='finemath_4plus_supply_v4',revision='e92b25a616738fe95dc186b64dfb19f9c8525594',
  authorization='2026-10-02 human reply: 采用来源级内部预训练口径',created_unix=int(time.time()),dataset_license='odc-by-1.0',
  internal_training_eligible=True,distribution_eligible=False,individual_contents_rights_certified=False,
  preserve_source_locator=True,scope='internal pretraining only; original item rights not certified; public data and weights require separate review',
  evidence_files=[dict(path=str((a.output/n).resolve()),sha256=file_sha(a.output/n),url=url) for n,url in URLS.items()]+[dict(path=str((a.output/'ATTRIBUTION.txt').resolve()),sha256=file_sha(a.output/'ATTRIBUTION.txt'))]))
 print('BOUND_INTERNAL_SOURCE_TERMS_NOT_INDIVIDUAL_RIGHTS_CERTIFICATION')
if __name__=='__main__':main()
