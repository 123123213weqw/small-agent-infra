#!/usr/bin/env python3
"""Audit/import old internal corpus; no new quality approval or GPU/API calls."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from small_agent.curation.supply_control import SupplyControl, file_sha, recipe_identity

DOMAIN_MAP = {'code_other':'code', 'code_python':'code', 'code_shell':'code',
              'general_en':'general_en', 'wave2_general_en':'general_en',
              'general_zh':'general_zh', 'finewiki_zh':'general_zh', 'math_en':'math'}
FROZEN = ('pretrain_ready_v1/inputs/prior_approved_real_v2/manifest.json',
          'finewiki_zh_short_internal_v2/manifest.json',
          'pretrain_reuse_wave2_v1/frozen_source/manifest.json')

def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.inventory-')
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        os.link(tmp,path)
    finally: os.unlink(tmp)

def verify_pack(root: Path, output: Path):
    if output.exists(): raise FileExistsError(output)
    manifest = root/'manifest.json'; x=json.loads(manifest.read_text())
    if x['stage'] != 'approved_sharded_token_stream' or x['eos_token_id'] != 2 or not x['license_gate']['training_eligible']:
        raise ValueError('not an existing internally eligible pack')
    before=file_sha(manifest);files=[];body=Counter();docs=Counter()
    for source in x.get('source_packs',{}).values():
        if file_sha(Path(source['path'])/'manifest.json') != source['manifest_sha256']:
            raise ValueError('source pack manifest binding mismatch')
    for s in x['shards']:
        if s['domain'] not in DOMAIN_MAP or s['split'] not in ('train','validation','test'):
            raise ValueError('unknown domain/split')
        for suffix in ('bin','idx'):
            p=root/s[suffix]
            if p.parent != root or p.stat().st_size != s[suffix+'_bytes'] or file_sha(p) != s[suffix+'_sha256']:
                raise ValueError(f'physical shard mismatch: {p}')
            files.append({'path':str(p),'sha256':s[suffix+'_sha256'],'bytes':p.stat().st_size})
        if s['bin_bytes'] != s['tokens']*2 or not 0 <= s['documents'] <= s['tokens']:
            raise ValueError('pack counter invalid')
        key=s['domain']+'/'+s['split'];body[key]+=s['tokens']-s['documents'];docs[key]+=s['documents']
    if file_sha(manifest) != before: raise ValueError('manifest changed during audit')
    publish(output,{'stage':'physical_pack_hashes_verified','manifest_path':str(manifest),
        'manifest_sha256':before,'tokenizer_sha256':x['tokenizer_sha256'],
        'body_tokens_by_raw_domain_split':dict(body),'documents_by_raw_domain_split':dict(docs),
        'verified_files':files,'quality_recertified':False,'gpu_used':False,
        'limits':'Hashes/sizes and manifest counts checked; not full re-tokenization or loader/cache evaluation.'})

def audit(root: Path, pack_report: Path, output: Path, recipe: dict):
    import pyarrow.parquet as pq
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.legacy-audit-',dir=output.parent))
    index=temp/'documents.jsonl';body=Counter();docs=Counter();evidence=[];sources=[]
    packed=json.loads(pack_report.read_text());expected_tokenizer=recipe['tokenizer_sha256']
    if packed['tokenizer_sha256'] != expected_tokenizer: raise ValueError('tokenizer mismatch')
    reuse=json.loads((root/FROZEN[-1]).read_text())
    bindings={str(Path(i['path'])/'manifest.json'):i['manifest_sha256'] for i in reuse['inputs']['prior_sources']}
    try:
        with index.open('w') as out:
            for rel in FROZEN:
                p=root/rel;x=json.loads(p.read_text());mh=file_sha(p)
                if x['stage'] != 'accepted_real' or not x['license_gate']['training_eligible'] or x['family_split_conflicts'] != 0:
                    raise ValueError('old source not internally eligible')
                if str(p) in bindings and bindings[str(p)] != mh: raise ValueError('prior binding mismatch')
                tok=x.get('tokenizer',{}).get('tokenizer_json_sha256')
                if tok and tok != expected_tokenizer: raise ValueError('old tokenizer mismatch')
                evidence.append({'path':str(p),'sha256':mh});local_body=Counter();local_docs=Counter()
                for s in x['shards']:
                    shard=p.parent/s['domain']/s['file'];sh=file_sha(shard)
                    if sh != s['sha256'] or shard.stat().st_size != s['bytes']:
                        raise ValueError('frozen parquet hash mismatch')
                    evidence.append({'path':str(shard),'sha256':sh})
                    pf=pq.ParquetFile(shard);count=0
                    token_field='own_text_tokens' if 'own_text_tokens' in pf.schema_arrow.names else 'text_tokens'
                    for batch in pf.iter_batches(batch_size=1024,columns=['text','normalized_sha256','family_id','domain','split',token_field]):
                        for row in batch.to_pylist():
                            count+=1;raw=row['domain'];split=row['split'];n=row[token_field]
                            sha=hashlib.sha256(row['text'].encode('utf-8')).hexdigest()
                            if sha != row['normalized_sha256']: raise ValueError('body SHA differs from frozen record')
                            if raw != s['domain'] or raw not in DOMAIN_MAP or split not in ('train','validation','test') or type(n) is not int or n <= 0:
                                raise ValueError('invalid frozen document')
                            key=raw+'/'+split;body[key]+=n;docs[key]+=1;local_body[split]+=n;local_docs[split]+=1
                            out.write(json.dumps({'body_sha256':sha,'family_id':row['family_id'],
                                'raw_domain':raw,'domain':DOMAIN_MAP[raw],'split':split,'body_tokens':n,
                                'source_manifest_sha256':mh},ensure_ascii=False)+'\n')
                    if count != s['rows']: raise ValueError('frozen rows differ')
                for split, totals in x['splits'].items():
                    if local_body[split] != totals['text_tokens'] or local_docs[split] != totals['documents']:
                        raise ValueError('train-only row count differs from source manifest')
                sources.append({'path':str(p),'sha256':mh,'splits':x['splits'],
                    'license_gate':x['license_gate'],'quality_gate':x['quality_gate']})
                print('AUDITED',rel,dict(local_body),flush=True)
            out.flush();os.fsync(out.fileno())
        if dict(body) != packed['body_tokens_by_raw_domain_split'] or dict(docs) != packed['documents_by_raw_domain_split']:
            raise ValueError('frozen rows differ from physically verified combined pack')
        final_index=output/'documents.jsonl'
        evidence.extend([{'path':str(pack_report.resolve()),'sha256':file_sha(pack_report)},
                         {'path':str(final_index),'sha256':file_sha(index)}])
        receipt={'stage':'legacy_inventory_audit_internal_only','recipe_sha256':recipe_identity(recipe),
            'tokenizer_sha256':expected_tokenizer,'pack_manifest_sha256':packed['manifest_sha256'],
            'pack_verification_path':str(pack_report.resolve()),'document_index_path':str(final_index),
            'body_tokens_by_raw_domain_split':dict(body),'documents_by_raw_domain_split':dict(docs),
            'evidence_files':evidence,'input_identity':{'source_manifests':[{k:s[k] for k in ('path','sha256')} for s in sources],
                'pack_manifest_sha256':packed['manifest_sha256']},'source_policies':sources,
            'quality_recertified':False,'near_dedup_recomputed':False,'tokens_reencoded':False,
            'notes':['Counts use frozen per-document token fields, checked against physical pack totals.',
                'No per-document factual certification or permission to publish corpus/weights.']}
        (temp/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
        temp.rename(output)
    except BaseException:
        import shutil
        shutil.rmtree(temp,ignore_errors=True);raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['verify-pack','audit','import'])
    p.add_argument('--root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--pack-report',type=Path)
    p.add_argument('--recipe',type=Path);p.add_argument('--database',type=Path);p.add_argument('--receipt',type=Path)
    a=p.parse_args()
    if a.action=='verify-pack': verify_pack(a.root.resolve(),a.output.resolve());return
    recipe=json.loads(a.recipe.read_text())
    if a.action=='audit': audit(a.root.resolve(),a.pack_report.resolve(),a.output.resolve(),recipe);return
    control=SupplyControl(a.database,recipe)
    try: print(json.dumps(control.import_legacy(a.receipt),ensure_ascii=False,indent=2))
    finally:control.close()

if __name__=='__main__':main()
