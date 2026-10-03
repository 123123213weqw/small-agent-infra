#!/usr/bin/env python3
"""READY-only 20B domain balance, candidate routes and replenishment work items."""
import argparse,json,sqlite3,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.full_stock_v9 import quota_status,priorities
from small_agent.curation.release_gates_v5 import write
from small_agent.curation.supply_control import file_sha
from p0_curation_worker_v3 import atomic_status


def report(config):
    recipe=json.loads(Path(config['recipe']).read_text());db=sqlite3.connect('file:'+str(Path(config['database']).resolve())+'?mode=ro',uri=True)
    try:
        db.execute('BEGIN')
        ready=dict(db.execute("SELECT d.domain,SUM(d.tokens) FROM documents d JOIN batches b USING(batch_id) JOIN releases r USING(batch_id) WHERE d.split='train' AND b.status='ready' GROUP BY d.domain"))
        legacy=dict(db.execute("SELECT d.domain,SUM(d.tokens) FROM documents d JOIN legacy_imports l USING(batch_id) WHERE d.split='train' GROUP BY d.domain"))
        catalog_count=db.execute('SELECT COUNT(*) FROM releases').fetchone()[0]
    finally:db.close()
    orig=sqlite3.connect('file:'+str((Path(config['original_stock'])/'stock.sqlite').resolve())+'?mode=ro',uri=True)
    try:candidates=dict(orig.execute("SELECT domain,SUM(n) FROM records WHERE split='train' GROUP BY domain"))
    finally:orig.close()
    state=quota_status(recipe,ready,legacy,candidates);state.update(version=9,stage='FULL_STOCK_BALANCE_AND_REPLENISHMENT',ledger_release_count=catalog_count,
          shared_campaign_budget_cny=config['shared_campaign_budget_cny'],maximum_concurrency=config['maximum_concurrency'],
          tokenizer_sha256=recipe['tokenizer_sha256'],auto_training=False,legacy_preserved_not_silently_counted_as_ready=True,
          ledger_counts_pending_physical_catalog_verification=True)
    scan=Path(config['stock'])/'SCAN_COMPLETE.json';state['full_stock_scan_complete']=scan.exists()
    if scan.exists():state['routing']=json.loads(scan.read_text())['routes']
    state['production_priority']=priorities(recipe,ready,set(recipe['targets_by_domain']))
    state['replenishment_jobs']=[dict(domain=d,remaining_READY_train_body_tokens=state['remaining_ready_by_domain'][d],
        status='NEEDS_FROZEN_SOURCE_INGEST_OR_LEGACY_RECERTIFICATION' if d in ('general_en','general_zh','technical') else 'USE_EXISTING_STOCK_FIRST_THEN_REPLENISH',
        preference=config['replenishment_preferences'][d],raw_tokens_are_not_a_READY_commitment=True)
        for d in state['production_priority']]
    state['training_mixture_contract']=dict(target_weights=state['target_weights'],sampling_unit='loss_tokens',
         missing_domain_action=state['missing_domain_action'],oversupply='reserve_bank_not_higher_training_weight',
         replay_does_not_advance_unique_20b_goal=True,actual_sampler_configuration_not_deployed=True)
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text());r=report(cfg)
    atomic_status(Path(cfg['stock'])/'BALANCE_STATUS.json',r);print(json.dumps({k:v for k,v in r.items() if k not in ('routing','training_mixture_contract')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
