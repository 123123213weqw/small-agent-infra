#!/usr/bin/env python3
"""Hash-pin all existing inventory; reuse measured census bodies instead of retokenizing FineMath."""
import argparse,json,sqlite3,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.supply_control import file_sha,digest
from small_agent.curation.release_gates_v5 import write,bound_json
from p0_inventory_sources_v1 import RAW_ROOTS

def register(cfg):
    root=Path(cfg['raw_root']);out=Path(cfg['root']);out.mkdir(parents=True,exist_ok=True)
    registry=out/'REGISTRY.json'
    if registry.exists():
        s=json.loads(registry.read_text())
        if s['config_sha256']!=digest(cfg):raise ValueError('Frozen registry config changed')
        return s
    inventory=json.loads(Path(cfg['source_inventory']).read_text());meta={str(root/f['path']):f for g in inventory['raw_groups'] for f in g['files']}
    # Reuse canonical census Parquets, binding original source file identities too.
    census=Path(cfg['census_root']);db=sqlite3.connect('file:'+str(census/'census.sqlite')+'?mode=ro',uri=True)
    try:receipts=[json.loads(x[0]) for x in db.execute('SELECT receipt FROM shards ORDER BY key')]
    finally:db.close()
    plan=json.loads(Path(cfg['download_plan']).read_text());dest={(x['source_id'],x['path']):x for x in plan['items']};used=set();groups={}
    def add(key,domain,language,revision,terms,path,expected=None,normalized=False,origin=None):
        p=Path(path);h=file_sha(p)
        if expected and expected!=h:raise ValueError('Pinned source SHA mismatch: '+str(p))
        item=dict(path=str(p.resolve()),sha256=h,bytes=p.stat().st_size,normalized=normalized)
        if origin:item['original_source']=origin
        source=groups.setdefault(key,dict(id=key,domain=domain,language=language,revision=revision,terms_scope=terms,normalized=normalized,files=[]))
        source['files'].append(item);
        m=meta.get(str(p),{})
        if origin:m=origin
        if m.get('dataset'):source['dataset']=m['dataset']
        if m.get('declared_policy'):source['declared_policy']=m['declared_policy']
        if key=='fineweb_edu_wave2_v1':source['dataset']='HuggingFaceFW/fineweb-edu'
        if key=='finewiki_zh_2025_candidate_v1':source['dataset']='HuggingFaceFW/finewiki'
        print('REGISTERED',key,p.name,flush=True)
    for x in receipts:
        raw=dest[(x['source_id'],x['source_file'])];used.add(str(Path(raw['destination']).resolve()));used.add(raw['sha256'])
        if raw['sha256']!=x['source_sha256']:raise ValueError('Census original source differs')
        if file_sha(Path(raw['destination']))!=raw['sha256']:raise ValueError('Original raw file changed')
        add(x['source_id'],'math' if 'finemath' in x['source_id'] else 'code','en' if 'finemath' in x['source_id'] else 'unknown',raw['revision'],'finemath_authorized_internal' if 'finemath' in x['source_id'] else 'record_license_only',census/x['output_relative_path'],x['output_sha256'],True,raw)
    for name in RAW_ROOTS:
        raw=root/name/'raw'
        if not raw.exists():continue
        for p in sorted(raw.rglob('*')):
            if not p.is_file() or p.is_symlink() or p.suffix not in ('.parquet','.jsonl'):continue
            f=meta.get(str(p),{});expected=f.get('expected_sha256');key=p.parent.name if p.parent!=raw else name
            if str(p.resolve()) in used or expected in used:continue
            domain='code' if any(x in key for x in ('code','coding')) else 'math' if any(x in key for x in ('math','textbook')) else 'general_zh' if any(x in key for x in ('_zh','finewiki')) else 'general_en'
            if 'cosmopedia' in key:domain='general_en'
            lang='zh' if domain=='general_zh' else 'python' if key in ('swallow_code_v2_python','ultradata_code_l2_py') else 'unknown' if domain=='code' else 'en'
            terms='pending_source_terms'
            if key.startswith('github_code'):terms='record_license_only'
            if key.startswith('fineweb_edu'):terms='inherited_FineWeb_internal_source_policy'
            if key.startswith('finewiki'):terms='inherited_FineWiki_internal_source_policy'
            revision=f.get('revision')
            if key=='fineweb_edu_wave2_v1':
                m=json.loads((root/key/'manifest.json').read_text());revision=m['revision'];expected=next(x['sha256'] for x in m['items'] if Path(x['path']).name==p.name)
            if key=='finewiki_zh_2025_candidate_v1':
                m=json.loads((root/key/'raw_manifest.json').read_text());revision=m.get('revision',revision)
                entries=m.get('files',m.get('items',[]))
                if isinstance(entries,list):
                    matches=[x for x in entries if Path(x.get('path',x.get('remote_file',x.get('file','')))).name==p.name]
                    if matches:expected=matches[0].get('sha256',expected)
            add(key,domain,lang,revision or 'file-pinned-revision-unknown',terms,p,expected)
    mdn=root/'mdn_full_wave4_v1/normalized/mdn_en_full_candidates_v1.jsonl'
    if mdn.exists():
        m=json.loads((root/'mdn_full_wave4_v1/audit/mdn_en_full_candidates_v1.manifest.json').read_text())
        add('mdn_content_en','technical','en',m['source_revision'],'pending_source_terms',mdn,m['output_sha256'])
    registry_value=dict(version=11,stage='full_existing_inventory_frozen_not_quality_approval',config_sha256=digest(cfg),sources=sorted(groups.values(),key=lambda x:(0 if x['id']=='mdn_content_en' else 2 if x['domain']=='math' else 1,sum(f['bytes'] for f in x['files']),x['id'])),files=sum(len(x['files']) for x in groups.values()),source_groups=len(groups),tokenizer_sha256=cfg['tokenizer_sha256'],raw_scope_roots=list(RAW_ROOTS),old_inventory_reused_not_reimported=True,training_started=False)
    write(registry,registry_value);return registry_value

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();s=register(json.loads(a.config.read_text()));print(json.dumps(dict(sources=s['source_groups'],files=s['files'])))
