#!/usr/bin/env python3
"""Reverify already imported legacy evidence and bind explicit internal goal credit; no re-import."""
import argparse,json,sqlite3,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.supply_control import file_sha,recipe_identity
from small_agent.curation.release_gates_v5 import bound_json,write
from p0_import_legacy_inventory_v4 import DOMAIN_MAP


def verify(database,receipt,recipe_path,output,inherit_physical_verification=False):
    recipe=json.loads(recipe_path.read_text());legacy=bound_json(receipt);h=file_sha(receipt)
    if legacy['stage']!='legacy_inventory_audit_internal_only' or legacy['recipe_sha256']!=recipe_identity(recipe) or legacy['tokenizer_sha256']!=recipe['tokenizer_sha256']:
        raise ValueError('Existing legacy internal receipt identity differs')
    pack=bound_json(Path(legacy['pack_verification_path']))
    if pack['tokenizer_sha256']!=recipe['tokenizer_sha256'] or pack['stage']!='physical_pack_hashes_verified' or pack['manifest_sha256']!=legacy['pack_manifest_sha256'] or pack['body_tokens_by_raw_domain_split']!=legacy['body_tokens_by_raw_domain_split']:
        raise ValueError('Legacy physical evidence/counters differ')
    # The original physical verification was executed on L40, not V100.
    # Explicit inherited mode binds that immutable report and does NOT claim a fresh file rehash.
    if not inherit_physical_verification:
        for item in pack['verified_files']:
            p=Path(item['path'])
            if p.stat().st_size!=item['bytes'] or file_sha(p)!=item['sha256']:raise ValueError('Legacy physical shard changed')
        if file_sha(Path(pack['manifest_path']))!=pack['manifest_sha256']:raise ValueError('Legacy pack manifest changed')
    db=sqlite3.connect('file:'+str(database.resolve())+'?mode=ro',uri=True)
    try:
        db.execute('BEGIN');rows=db.execute('SELECT batch_id FROM legacy_imports WHERE receipt_sha=?',(h,)).fetchall()
        if len(rows)!=1:raise ValueError('Must reference exactly one already imported immutable legacy batch')
        batch=rows[0][0]
        if db.execute('SELECT 1 FROM releases WHERE batch_id=?',(batch,)).fetchone():raise ValueError('Legacy must not be relabeled as new READY')
        actual=dict(db.execute("SELECT domain,SUM(tokens) FROM documents WHERE batch_id=? AND split='train' GROUP BY domain",(batch,)))
        expected_by_domain=Counter()
        for key,n in legacy['body_tokens_by_raw_domain_split'].items():
            domain,split=key.rsplit('/',1)
            if split=='train':expected_by_domain[DOMAIN_MAP[domain]]+=n
        expected=sum(expected_by_domain.values())
        if actual!=dict(expected_by_domain):raise ValueError('Legacy train-only per-domain ledger differs')
    finally:db.close()
    credit=dict(version=10,stage='verified_existing_internal_legacy_goal_credit_not_new_READY',
                explicit_user_request='我原来的10亿也要算进去',legacy_import_batch_id=batch,
                legacy_receipt_sha256=h,recipe_sha256=recipe_identity(recipe),tokenizer_sha256=recipe['tokenizer_sha256'],
                credited_train_body_tokens=expected,credited_by_domain=actual,
                already_imported_no_new_ledger_rows=True,
                physical_files_rehashed_this_run=not inherit_physical_verification,
                physical_verification_scope='inherited_hash_bound_L40_report' if inherit_physical_verification else 'rehashed_actual_local_pack_files',
                quality_recertified=False,distribution_eligible=False,
                existing_internal_training_scope_preserved=True,
                evidence_files=[dict(path=str(receipt.resolve()),sha256=h),dict(path=str(Path(legacy['pack_verification_path']).resolve()),sha256=file_sha(Path(legacy['pack_verification_path'])))])
    write(output,credit);return credit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('database','receipt','recipe','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--inherit-physical-verification',action='store_true',help='Use already hash-bound L40 physical report; not a fresh V100 file rehash')
    a=p.parse_args();print(json.dumps(verify(a.database,a.receipt,a.recipe,a.output,a.inherit_physical_verification),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
