#!/usr/bin/env python3
"""Inherited internal + new READY 20B domain balance, candidate routes and replenishment work items."""
import argparse,json,sqlite3,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.full_stock_v9 import priorities
from small_agent.curation.goal_credit_v10 import goal_balance
from small_agent.curation.supply_control import recipe_identity
from small_agent.curation.release_gates_v5 import bound_json
from small_agent.curation.release_gates_v5 import write
from small_agent.curation.supply_control import file_sha
from p0_curation_worker_v3 import atomic_status


def report(config):
    recipe=json.loads(Path(config['recipe']).read_text())
    proof=bound_json(Path(config['legacy_goal_credit_receipt']))
    if (config['goal_credit_policy']!='inherited_internal_plus_new_READY' or
        proof['stage']!='verified_existing_internal_legacy_goal_credit_not_new_READY' or
        proof['recipe_sha256']!=recipe_identity(recipe) or proof['tokenizer_sha256']!=recipe['tokenizer_sha256'] or
        proof['quality_recertified'] is not False or proof['existing_internal_training_scope_preserved'] is not True):
        raise ValueError('Explicit legacy goal credit evidence/policy differs')
    db=sqlite3.connect('file:'+str(Path(config['database']).resolve())+'?mode=ro',uri=True)
    try:
        db.execute('BEGIN')
        if db.execute('SELECT batch_id FROM legacy_imports WHERE receipt_sha=?',(proof['legacy_receipt_sha256'],)).fetchall()!=[(proof['legacy_import_batch_id'],)]:
            raise ValueError('Goal credit receipt not imported uniquely')
        if db.execute('SELECT 1 FROM releases JOIN legacy_imports USING(batch_id)').fetchone():
            raise ValueError('Legacy cannot be relabeled as a new READY release')
        ready=dict(db.execute("SELECT d.domain,SUM(d.tokens) FROM documents d JOIN batches b USING(batch_id) JOIN releases r USING(batch_id) WHERE d.split='train' AND b.status='ready' GROUP BY d.domain"))
        legacy=dict(db.execute("SELECT d.domain,SUM(d.tokens) FROM documents d JOIN legacy_imports l USING(batch_id) WHERE d.split='train' GROUP BY d.domain"))
        if legacy!=proof['credited_by_domain'] or sum(legacy.values())!=proof['credited_train_body_tokens']:
            raise ValueError('Goal credit count changed; verify a new immutable proof first')
        catalog_count=db.execute('SELECT COUNT(*) FROM releases').fetchone()[0]
    finally:db.close()
    orig=sqlite3.connect('file:'+str((Path(config['original_stock'])/'stock.sqlite').resolve())+'?mode=ro',uri=True)
    try:candidates=dict(orig.execute("SELECT domain,SUM(n) FROM records WHERE split='train' GROUP BY domain"))
    finally:orig.close()
    state=goal_balance(recipe,ready,legacy,candidates);state.update(version=10,stage='FULL_STOCK_BALANCE_AND_REPLENISHMENT',ledger_release_count=catalog_count,legacy_goal_credit_receipt_sha256=file_sha(Path(config['legacy_goal_credit_receipt'])),
          shared_campaign_budget_cny=config['shared_campaign_budget_cny'],maximum_concurrency=config['maximum_concurrency'],
          tokenizer_sha256=recipe['tokenizer_sha256'],auto_training=False,legacy_preserved_not_silently_counted_as_ready=True,
          ledger_counts_pending_physical_catalog_verification=True)
    scan=Path(config['stock'])/'SCAN_COMPLETE.json';state['full_stock_scan_complete']=scan.exists()
    if scan.exists():state['routing']=json.loads(scan.read_text())['routes']
    state['production_priority']=priorities(recipe,state['accounted_by_domain'],set(recipe['targets_by_domain']))
    state['replenishment_jobs']=[dict(domain=d,remaining_goal_train_body_tokens=state['remaining_goal_by_domain'][d],
        status='NEEDS_NEW_FROZEN_SOURCE_INGEST_EXISTING_INTERNAL_LEGACY_ALREADY_CREDITED' if d in ('general_en','general_zh','technical') else 'USE_EXISTING_STOCK_FIRST_THEN_REPLENISH',
        preference=config['replenishment_preferences'][d],raw_tokens_are_not_a_READY_commitment=True)
        for d in state['production_priority']]
    state['training_mixture_contract']=dict(target_weights=state['target_weights'],sampling_unit='loss_tokens',
         missing_domain_action=state['missing_domain_action'],oversupply='reserve_bank_not_higher_training_weight',
         replay_does_not_advance_unique_20b_goal=True,actual_sampler_configuration_not_deployed=True,
         inherited_plus_READY_training_bundle_not_yet_assembled=True)
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text());r=report(cfg)
    atomic_status(Path(cfg['stock'])/'BALANCE_STATUS.json',r);print(json.dumps({k:v for k,v in r.items() if k not in ('routing','training_mixture_contract')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
