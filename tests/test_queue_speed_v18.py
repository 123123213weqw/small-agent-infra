"""No API; deterministic scheduling and pre-billing equivalence fixtures."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import p0_inventory_supply_v18 as q
from test_parallel_cpu_v13 import ParallelTests


class EndToEndTests(unittest.TestCase):
    setUp=ParallelTests.setUp

    def test_historical_publisher_and_performance_preserved(self):
        out=self.r/'release';out.mkdir();args=['--output',str(out)]
        self.assertEqual(q.publisher_for(args),'p0_release_inventory_batch_v18.py')
        for version in (11,13,15,17,18):
            old=self.r/f'performance{version}.json';old.write_text('{}')
            (out/'INPUT_BINDING.json').write_text(json.dumps(dict(
                engine_sha256={f'/old/p0_release_inventory_batch_v{version}.py':'h'},
                performance_config_path=str(old),performance_config_sha256=q.file_sha(old))))
            self.assertEqual(q.publisher_for(args),f'p0_release_inventory_batch_v{version}.py')
            self.assertEqual(q.performance_for(args,'new'),str(old))
        old.write_text('changed')
        with self.assertRaisesRegex(ValueError,'Historical'):q.performance_for(args,'new')

    def test_score_terminal_reuse_and_bounded_ramp(self):
        folder=self.r/'review';status=folder/'batch/worker/deepseek/ADAPTIVE_STATUS.json'
        status.parent.mkdir(parents=True)
        cfg=dict(allow_api=True,profile='private-not-read',_performance=dict(initial_teacher_concurrency=256))
        terminal=dict(stage='all_selected_attempted_with_holds',considered=500,selected=500,stop_reason=None)
        status.write_text(json.dumps(terminal))
        with patch.object(q,'component') as call:
            q.score(cfg,folder);call.assert_not_called()
        status.unlink()
        def component(script,args):
            self.assertEqual(script,'p0_deepseek_worker_v8.py')
            self.assertEqual(args[args.index('--initial-concurrency')+1],256)
            self.assertEqual(args[args.index('--maximum-concurrency')+1],2500)
            status.write_text(json.dumps(terminal))
        with patch.object(q,'component',side_effect=component):q.score(cfg,folder)
        terminal['considered']=499;status.write_text(json.dumps(terminal))
        with patch.object(q,'component'):
            with self.assertRaises(q.CampaignHeld):q.score(cfg,folder)

    def test_round_robin_preserves_all_cohorts(self):
        c=[(dict(id=x),'code','python') for x in ('a','b','c')]
        rotated=q.rotate(c,'b/code/python')
        self.assertEqual([x[0]['id'] for x in rotated],['c','a','b'])
        self.assertEqual(q.rotate(c,'unknown'),c)
        self.assertEqual(q.rotate([],None),[])

    def test_pending_does_not_upgrade_fallback_or_abandon_paid_plan(self):
        store=q.open_store(self.r/'stock');self.addCleanup(store.close)
        with store:
            store.execute('INSERT INTO files(id,path,sha,source,state) VALUES(1,?,?,?,?)',('p','h','s','complete'))
            store.execute('INSERT INTO records(h,fid,domain,lang,stratum,gate,split,job) VALUES(?,1,?,?,?,?,?,NULL)',('h','code','python','long','CANDIDATE_FALLBACK','train'))
        s=dict(id='s')
        self.assertFalse(q.pending(store,s,'code','python','source_audit',['long']))
        self.assertTrue(q.pending(store,s,'code','python','individual',['long']))
        self.assertFalse(q.pending(store,s,'code','python','source_audit',[]))
        with store:
            store.execute('INSERT INTO jobs(name,source,mode,state) VALUES(?,?,?,?)',('paid','s','source_audit','planned'))
            store.execute('UPDATE records SET job=?',('paid',))
        self.assertTrue(q.pending(store,s,'code','python','source_audit',[]))

    def test_fair_fast_first_and_fallback_cursor(self):
        root=self.r/'queue';store=q.open_store(root)
        with store:
            for fid,s in enumerate(('a','b'),1):
                store.execute('INSERT INTO files(id,path,sha,source,state) VALUES(?,?,?,?,?)',(fid,s,'h',s,'complete'))
                for suffix,gate in (('fast','CANDIDATE'),('slow','CANDIDATE_FALLBACK')):
                    store.execute('INSERT INTO records(h,fid,domain,lang,stratum,gate,split) VALUES(?,?,?,?,?,?,?)',(s+suffix,fid,'code','python','short',gate,'train'))
        store.close()
        (root/'REGISTRY.json').write_text(json.dumps(dict(sources=[dict(id='a'),dict(id='b')])))
        policy=self.r/'policy.json';policy.write_text(json.dumps(dict(quality={})))
        cfg=dict(root=str(root),release_policy=str(policy),shared_campaign_budget_cny=1000,maximum_concurrency=2500,auto_training=False,distribution_eligible=False,maximum_batches_per_tick=2,_performance=dict(fallback_every_ticks=1))
        recipe=self.r/'recipe.json';recipe.write_text('{}');cfg.update(database=str(self.r/'not-used.sqlite'),recipe=str(recipe))
        calls=[]
        def process(cfg,store,source,d,l,*rest):
            mode=rest[3];calls.append((source['id'],mode))
            gate='CANDIDATE' if mode=='source_audit' else 'CANDIDATE_FALLBACK'
            with store:store.execute('UPDATE records SET gate=? WHERE fid=(SELECT id FROM files WHERE source=?) AND gate=?',('READY_INTERNAL',source['id'],gate))
            return True
        import small_agent.curation.full_inventory_v11 as control
        with patch.object(control,'AllStockControl') as ctl,patch.object(q,'source_cohorts',return_value=[('code','python')]),patch.object(q,'cached_approved',return_value=['short']),patch.object(q,'terms'),patch.object(q,'reference'),patch.object(q,'source_audit',return_value=(self.r,dict(approved_strata=['short']))),patch.object(q,'process',side_effect=process):
            ctl.return_value.status.return_value={'unique_ready_train_body_tokens':0}
            q.run(cfg);q.run(cfg)
        self.assertEqual(calls,[('a','source_audit'),('a','individual'),('b','source_audit'),('b','individual')])

    def test_unknown_paid_state_halts_all_lanes(self):
        root=self.r/'held';root.mkdir();(root/'API_HOLD.json').write_text('{}')
        cfg=dict(root=str(root),shared_campaign_budget_cny=1000,maximum_concurrency=2500,auto_training=False,distribution_eligible=False)
        recipe=self.r/'recipe-held.json';recipe.write_text('{}');cfg.update(database=str(self.r/'not-used.sqlite'),recipe=str(recipe))
        import small_agent.curation.full_inventory_v11 as control
        with patch.object(control,'AllStockControl') as ctl,patch.object(q,'open_store') as store,patch.object(q,'score') as score:
            ctl.return_value.status.return_value={'unique_ready_train_body_tokens':0}
            q.run(cfg);store.assert_not_called();score.assert_not_called()

    def test_parallel_pre_billing_equals_frozen_original(self):
        # Reuse the exact serial/global/local/dense-hold/ref-integrity fixture,
        # but exercise the v17 pre-billing wrapper at 16 workers as well.
        from p0_parallel_near_v13 import ParallelNearIndex
        original=ParallelNearIndex.filter
        def sixteen(index,rows,manifest,policy,database,workers):
            return original(index,rows,manifest,policy,database,16)
        with patch.object(ParallelNearIndex,'filter',sixteen):
            ParallelTests.test_parallel_near_global_local_decisions_equal_frozen_v6(self)

    def test_pre_billing_wrapper_16_and_stale_snapshot_rejected(self):
        inventory=self.r/'inventory';inventory.mkdir()
        manifest=inventory/'manifest.json'
        manifest.write_text(json.dumps(dict(ledger={'documents':1},database=str(inventory/'postings.sqlite'))))
        (inventory/'CURRENT').write_text(str(manifest))
        cfg=dict(inventory_root=str(inventory),database=str(self.r/'ledger.sqlite'),release_policy=str(self.r/'near.json'),_performance=dict(pre_billing_cpu_workers=16))
        Path(cfg['release_policy']).write_text(json.dumps(dict(near={})))
        with patch.object(q,'ledger_snapshot',return_value={'documents':1}),patch.object(q,'ParallelNearIndex') as index:
            index.return_value.filter.return_value=(self.rows,{})
            self.assertEqual(q.cheap_near(cfg,self.rows),self.rows)
            self.assertEqual(index.return_value.filter.call_args.args[-1],16)
            index.return_value.close.assert_called_once()
        with patch.object(q,'ledger_snapshot',return_value={'documents':2}),patch.object(q,'ParallelNearIndex') as index:
            self.assertIsNone(q.cheap_near(cfg,self.rows));index.assert_not_called()

    def test_fast_batch_multiplier_does_not_enlarge_individual_or_upgrade_fallback(self):
        store=q.open_store(self.r/'allocate');self.addCleanup(store.close)
        with store:
            store.execute('INSERT INTO files(id,path,sha,source,state) VALUES(1,?,?,?,?)',('p','h','s','complete'))
            for i in range(44):
                split='train' if i<40 else 'validation' if i<42 else 'test'
                store.execute('INSERT INTO records(h,fid,domain,lang,stratum,gate,split,n,rank) VALUES(?,1,?,?,?,?,?,?,?)',(str(i),'math','en','medium','CANDIDATE',split,10,str(i).zfill(4)))
        prompt=self.r/'prompt';prompt.write_text('fixed')
        cfg=dict(prompt=str(prompt),smoke_train_body_tokens=10,batch_train_body_tokens=100,maximum_batch_documents=10,_performance=dict(fast_batch_size_multiplier=4))
        name,rows=q.allocate(cfg,store,'s','math','en',dict(approved_strata=['medium']),'source_audit',False)
        self.assertEqual(len(rows),44)
        with store:store.execute('UPDATE records SET job=NULL,gate=?',('CANDIDATE_FALLBACK',))
        self.assertIsNone(q.allocate(cfg,store,'s','math','en',dict(approved_strata=['medium']),'source_audit',False))
        name,rows=q.allocate(cfg,store,'s','math','en',dict(approved_strata=['medium']),'individual',False)
        self.assertEqual(len(rows),14)

    def test_goal_requires_whole_hour_not_peak_or_legacy(self):
        from p0_ready_throughput_v18 import record
        initial=1_000_000_000
        record(self.r,initial,now=0)
        self.assertFalse(record(self.r,initial+200_000_000,now=60)['goal_verified'])
        result=record(self.r,initial+200_000_000,now=3600)
        self.assertTrue(result['goal_verified']);self.assertEqual(result['measured_hour_window']['tokens_per_hour'],200_000_000)
        with self.assertRaises(ValueError):record(self.r,0,now=3700)

if __name__=='__main__':unittest.main()
