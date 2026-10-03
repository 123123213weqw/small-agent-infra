"""Real CPU packing, bounded reserve, immutable derivation and historical routes."""
import json,os,sqlite3,unittest
from pathlib import Path
from unittest.mock import patch
from p0_heldout_reserve_v24 import select_reserve,split_report,check_parquet
from p0_recover_heldout_v24 import validate_derivation,finalize_hold
from small_agent.curation.supply_control import file_sha
from test_parallel_cpu_v13 import ParallelTests

class HeldoutTests(unittest.TestCase):
    setUp=ParallelTests.setUp

    def test_stratum_balanced_bounded_reserve(self):
        import p0_inventory_supply_v24 as q
        db=q.open_store(self.r/'stock');self.addCleanup(db.close)
        with db:
            db.execute("INSERT INTO files(id,path,sha,source,state) VALUES(1,'p','h','source','complete')")
            for split in ('train','validation','test'):
                for stratum in ('medium','short'):
                    for i in range(20):
                        db.execute('INSERT INTO records(h,fid,domain,lang,stratum,gate,split,n,rank) VALUES(?,1,?,?,?,?,?,?,?)',
                                   (split+stratum+str(i),'math','en',stratum,'CANDIDATE',split,10,str(i).zfill(4)))
        query=q.source_query(db,"r.gate='CANDIDATE' AND r.job IS NULL")
        rows=select_reserve(db,query,['source','math','en'],['medium','short'],target_tokens=33,maximum_documents_per_stratum=4)
        self.assertEqual(len(rows),12)
        self.assertEqual({(r['split'],r['stratum']) for r in rows},{(s,t) for s in ('validation','test') for t in ('medium','short')})
        self.assertEqual([dict(r) for r in rows],[dict(r) for r in select_reserve(db,query,['source','math','en'],['short','medium'],target_tokens=33,maximum_documents_per_stratum=4)])
        with db:db.execute("DELETE FROM records WHERE split='validation'")
        self.assertIsNone(select_reserve(db,query,['source','math','en'],['short']))

    def test_after_gates_empty_validation_is_a_recoverable_hold(self):
        import pyarrow as pa,pyarrow.parquet as pq
        retained=[r for r in self.rows if r['split']!='validation']
        p=self.r/'filtered.parquet';pq.write_table(pa.Table.from_pylist(retained),p)
        report=check_parquet(p,4096)
        self.assertEqual(report['status'],'needs_heldout');self.assertEqual(report['missing_splits'],['validation'])
        self.assertFalse(report['train_split_reassigned'])

    def test_short_heldout_is_physically_padded_not_fabricated(self):
        from small_agent.data.sharded_pack import TokenizedDocument,pack_sharded
        from small_agent.data.sharded_dataset import ShardedTokenDataset
        docs=[TokenizedDocument('t','train','math',[3]*20),TokenizedDocument('v','validation','math',[4]),TokenizedDocument('x','test','math',[5])]
        metadata=[dict(normalized_sha256=d.document_id,split=d.split,own_text_tokens=len(d.token_ids)) for d in docs]
        self.assertEqual(split_report(metadata,16)['status'],'passed')
        out=self.r/'packed'
        pack_sharded(docs,out,sequence_length=16,target_shard_tokens=100,vocab_size=32,eos_token_id=2,tokenizer_sha256='fixture',mixture={'math':1},license_gate={'training_eligible':False},source_manifest_sha256='fixture')
        dataset=ShardedTokenDataset(out,'validation',16,verify_hashes=True)
        self.assertEqual(len(dataset),1);sample=dataset[0]
        self.assertEqual(int(sample['valid_tokens']),2)
        self.assertTrue((sample['labels'][2:]==-100).all())

    def test_recovery_keeps_original_bodies_splits_and_rejects_train_additions(self):
        def row(i,s):return dict(normalized_sha256=str(i),source_id='s',domain='math',language='en',family_id=str(i),split=s)
        old=[row(1,'train'),row(2,'validation')];extra=[row(3,'validation'),row(4,'test')]
        self.assertEqual(validate_derivation(old,extra)[:len(old)],old)
        with self.assertRaisesRegex(ValueError,'training'):validate_derivation(old,[row(5,'train')])
        with self.assertRaisesRegex(ValueError,'Duplicate'):validate_derivation(old,[row(2,'validation')])
        extra[0]['family_id']='1'
        with self.assertRaisesRegex(ValueError,'family'):validate_derivation(old,extra)

    def test_hold_archive_is_idempotent_and_does_not_clear_new_hold(self):
        new=self.r/'new';new.mkdir();hold=self.r/'hold';hold.write_text('original')
        intent=new/'HELDOUT_RECOVERY_V24.json';intent.write_text('{}')
        receipt=dict(old_hold_path=str(hold),old_hold_sha256=file_sha(hold))
        finalize_hold(self.r,new,receipt);self.assertFalse(hold.exists())
        hold.write_text('new unrelated quality failure')
        finalize_hold(self.r,new,receipt);self.assertEqual(hold.read_text(),'new unrelated quality failure')

    def test_historical_queues_and_api_unknown_holds_are_preserved(self):
        import test_queue_speed_v21 as original
        import p0_inventory_supply_v24 as q
        for method in ('test_score_terminal_reuse_and_bounded_ramp','test_unknown_paid_state_halts_all_lanes','test_fast_batch_multiplier_does_not_enlarge_individual_or_upgrade_fallback'):
            fixture=original.EndToEndTests(method);fixture.setUp()
            try:
                with patch.object(original,'q',q):getattr(fixture,method)()
            finally:fixture.doCleanups()
        out=self.r/'release';out.mkdir();args=['--output',str(out)]
        self.assertEqual(q.publisher_for(args),'p0_release_inventory_batch_v24.py')
        for version in (11,13,15,17,18,21,24):
            (out/'INPUT_BINDING.json').write_text(json.dumps(dict(engine_sha256={f'/old/p0_release_inventory_batch_v{version}.py':'h'})))
            self.assertEqual(q.publisher_for(args),f'p0_release_inventory_batch_v{version}.py')

    def test_full_five_domain_pack_and_post_commit_restore(self):
        import test_full_inventory_v11 as oldtests
        import p0_release_inventory_batch_v24 as publisher
        perf=self.r/'perf.json';perf.write_text(json.dumps(dict(release_cpu_workers=2)))
        original_write=oldtests.write
        def fixture_write(path,value):
            if Path(path).name=='config.json':value=dict(value,sketch_root=str(Path(path).parent/'sketch-reuse'))
            return original_write(path,value)
        with patch.dict(os.environ,P0_PARALLEL_CONFIG=str(perf)),patch.object(oldtests,'publisher',publisher),patch.object(oldtests,'write',side_effect=fixture_write):
            oldtests.InventoryTests().test_generic_real_pack_recovery_all_domains()

    def test_publisher_reports_shortage_without_quarantining_or_packing(self):
        import sys,pyarrow.parquet as pq
        from test_math_release_v6 import MathPipelineTests
        from small_agent.curation.release_gates_v5 import write
        from small_agent.curation.full_inventory_v11 import AllStockControl
        import p0_release_inventory_batch_v24 as publisher
        r=self.r/'fixture';r.mkdir()
        argv,out,db,recipe,quality,_=MathPipelineTests().fixture(r)
        argv[argv.index('--reference-batch'):argv.index('--reference-batch')+2]=[]
        rows=pq.read_table(r/'candidates.parquet').to_pylist()
        write(r/'registry/REGISTRY.json',{})
        write(r/'config.json',dict(root=str(r/'registry'),sketch_root=str(r/'cache')))
        (r/'candidate_binding.json').unlink()
        write(r/'candidate_binding.json',dict(candidate_sha256=file_sha(r/'candidates.parquet'),source_id='fixture',domain='math',language='en',registry_sha256=file_sha(r/'registry/REGISTRY.json')))
        write(r/'source/PLAN.json',dict(meta=dict(source_id='fixture')))
        argv+=['--config',str(r/'config.json'),'--domain','math','--mode','individual']
        quality.update(kept_ids=[x['normalized_sha256'] for x in rows if x['split']=='train'],excluded_ids=[x['normalized_sha256'] for x in rows if x['split']!='train'],batch_state_sha256='mocked_engineering_only')
        perf=r/'perf.json';perf.write_text(json.dumps(dict(release_cpu_workers=2)))
        with patch.dict(os.environ,P0_PARALLEL_CONFIG=str(perf)),patch.object(sys,'argv',argv),patch.object(publisher,'publication_quality',return_value=quality):publisher.main()
        self.assertEqual(json.loads((out/'HOLD.json').read_text())['stage'],'NEEDS_HELDOUT')
        self.assertFalse((out/'tokenized').exists());self.assertFalse((out/'READY_INTERNAL.json').exists())
        ctl=AllStockControl(db,recipe)
        try:
            status=ctl.status();self.assertEqual(status['unique_ready_train_body_tokens'],0)
            self.assertEqual(status['batch_statuses'],{'waiting_gate':1})
        finally:ctl.close()

    def test_exact_teacher_protocol_reuses_paid_cache_across_batch_manifests(self):
        from types import SimpleNamespace
        from p0_curation_worker_v3 import job_identity
        a=SimpleNamespace(policy={'review_purpose':'source_quality_audit'},manifest_hash='old')
        b=SimpleNamespace(policy=a.policy,manifest_hash='new')
        protocol={'text_sha256':'body','prompt_sha256':'prompt','model':'teacher'}
        self.assertEqual(job_identity(a,'deepseek',protocol),job_identity(b,'deepseek',protocol))
        self.assertNotEqual(job_identity(a,'deepseek',protocol),job_identity(b,'deepseek',dict(protocol,prompt_sha256='changed')))

    def test_recovery_stock_commit_crash_resumes_without_touching_old_batch(self):
        import pyarrow as pa,pyarrow.parquet as pq
        import p0_recover_heldout_v24 as recovery
        import p0_inventory_supply_v24 as queue
        from small_agent.curation.full_inventory_v11 import adapter
        from small_agent.curation.release_gates_v5 import write
        root=self.r/'campaign';root.mkdir();state=self.r/'state';state.mkdir()
        source=dict(id='source',revision='fixed',domain='math',language='en',normalized=True,terms_scope='internal_fixture')
        raw=[]
        for i,row in enumerate(self.rows):raw.append(dict(row,domain='math',language='en',family_id=str(i)))
        file=self.r/'raw.parquet';pq.write_table(pa.Table.from_pylist(raw),file)
        f=dict(path=str(file),sha256=file_sha(file));source['files']=[f]
        write(root/'REGISTRY.json',dict(sources=[source]))
        store=queue.open_store(root)
        job='batch_old';original_positions=[0,1,66,68]
        with store:
            store.execute('INSERT INTO files(id,path,sha,source,state) VALUES(1,?,?,?,?)',(str(file),file_sha(file),'source','complete'))
            store.execute('INSERT INTO jobs(name,source,mode) VALUES(?,?,?)',(job,'source','source_audit'))
            for i,row in enumerate(raw):
                store.execute('INSERT INTO records(h,fid,pos,byte,domain,lang,family,split,n,rank,stratum,gate,job,initial_gate) VALUES(?,1,?,0,?,?,?,?,?,?,?,?,?,?)',
                              (row['normalized_sha256'],i,'math','en',row['family_id'],row['split'],row['own_text_tokens'],str(i).zfill(4),'short/math/en','CANDIDATE',job if i in original_positions else None,'CANDIDATE'))
        store.close()
        batch=root/'batches'/job;old=batch/'release';old.mkdir(parents=True)
        original=[adapter(raw[i],source,f,i,0) for i in original_positions]
        for r in original:r['own_text_tokens']=raw[int(r['origin_row'])]['own_text_tokens']
        pq.write_table(pa.Table.from_pylist(original),batch/'candidates.parquet')
        write(old/'INPUT_BINDING.json',{'fixture':True});write(old/'QUALITY.json',{'fixture':True})
        frozen={p.name:p.read_bytes() for p in batch.iterdir() if p.is_file()}
        cb=dict(source_id='source',domain='math',language='en')
        quality=dict(kept_ids=[r['normalized_sha256'] for r in original if r['split']=='train'])
        hold=root/'holds'/queue.digest('source/math/en');write(hold,dict(failure_type='CalledProcessError',reason=job+' heldout shortage'))
        cfg=dict(root=str(root),database=str(state/'ledger.sqlite'))
        mocked=(batch,old,{'mode':'source_audit'},cb,quality,{}, {'status':'needs_heldout'})
        with patch.object(recovery,'validate_failed',return_value=mocked),patch.object(recovery,'finalize_hold',side_effect=RuntimeError('controlled after stock commit')):
            with self.assertRaisesRegex(RuntimeError,'controlled'):recovery.apply(cfg,job)
        with patch.object(recovery,'validate_failed',side_effect=AssertionError('must not reallocate or score after commit')):
            recovery.apply(cfg,job);recovery.apply(cfg,job)
        self.assertEqual(frozen,{p.name:p.read_bytes() for p in batch.iterdir() if p.is_file()})
        store=queue.open_store(root)
        try:
            self.assertEqual(store.execute('SELECT state FROM jobs WHERE name=?',(job,)).fetchone()[0],'SUPERSEDED_HELDOUT_V24')
            new=store.execute('SELECT name FROM jobs WHERE name!=?',(job,)).fetchone()[0]
            derived=pq.read_table(root/'batches'/new/'candidates.parquet').to_pylist()
            self.assertEqual(derived[:len(original)],original)
            self.assertEqual([r['normalized_sha256'] for r in derived if r['split']=='train'],[r['normalized_sha256'] for r in original if r['split']=='train'])
            self.assertTrue((root/'batches'/new/'OLD_COHORT_HOLD_V24.json').exists())
        finally:store.close()

if __name__=='__main__':unittest.main()
