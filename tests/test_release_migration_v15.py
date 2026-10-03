import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import p0_migrate_release_v15 as migration
import p0_inventory_supply_v15 as queue
from small_agent.curation.supply_control import digest,file_sha
from small_agent.curation.full_inventory_v11 import AllStockControl


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.r=Path(self.tmp.name)
        self.batch=self.r/'batches/batch_fixture';self.old=self.batch/'release';self.old.mkdir(parents=True)
        self.recipe=json.loads((ROOT/'configs/curation/base_1b_supply_20b_v4.json').read_text())
        cfg=dict(root=str(self.r),database=str(self.r/'state/ledger.sqlite'))
        self.args={'output':str(self.old),'database':cfg['database'],'domain':'code','mode':'source_audit'}
        mapping={'candidate_sha256':'candidates','candidate_binding_sha256':'candidate-binding','config_sha256':'config',
            'source_terms_sha256':'source-terms','policy_sha256':'policy','tokenizer_sha256':'tokenizer',
            'eval_registry_sha256':'eval-registry','inventory_manifest_sha256':'inventory-index'}
        binding=dict(domain='code',mode='source_audit')
        for field,arg in mapping.items():
            path=self.batch/(arg+'.json');path.write_text(json.dumps(cfg) if arg=='config' else '{}')
            self.args[arg]=str(path);binding[field]=file_sha(path)
        recipepath=self.r/'recipe.json';recipepath.write_text(json.dumps(self.recipe));self.args['recipe']=str(recipepath)
        for k in ('source-audit','reference','batch-audit'):self.args[k]=str(self.r/k)
        engine=self.r/'frozen_engine.py';engine.write_text('# immutable');binding['engine_sha256']={str(engine):file_sha(engine)}
        p=self.old/'INPUT_BINDING.json';p.write_text(json.dumps(binding))
        ctl=AllStockControl(Path(cfg['database']),self.recipe)
        self.bid=ctl.register('code',binding);ctl.close()
        self.quality={'status':'passed'};self.review={'terminal':True}
        self.intent=dict(version=15,job=self.batch.name,arguments=self.args,old_batch_id=self.bid,new_output=str(self.batch/'release_v15'),
            old_binding_sha256=file_sha(p),paid_reviews={k:self.review for k in ('source-audit','reference','batch-audit')},quality_digest=digest(self.quality))
        self.path=self.r/'MIGRATION.json';self.path.write_text(json.dumps(self.intent))
        self.addCleanup(patch.stopall)
        patch.object(migration,'terminal_review',return_value=self.review).start()
        patch.object(migration,'publication_quality',return_value=self.quality).start()

    def status(self):
        with sqlite3.connect(self.args['database']) as db:
            return db.execute('SELECT status FROM batches WHERE batch_id=?',(self.bid,)).fetchone()[0]

    def test_idempotent_quarantine_preserves_all_old_files(self):
        before={p.name:p.read_bytes() for p in self.old.iterdir()}
        migration.apply(self.path);migration.apply(self.path)
        self.assertEqual(self.status(),'quarantined')
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.old.iterdir()})
        receipt=json.loads((self.batch/'CPU_MIGRATION_V15.json').read_text())
        self.assertEqual(receipt['api_requests'],0)
        # The route additionally pins the real candidate file.
        self.assertFalse((self.batch/'release_v15').exists())

    def test_recover_after_ledger_hold_before_receipt(self):
        ctl=AllStockControl(Path(self.args['database']),self.recipe)
        ctl.hold(self.bid,'CPU_MIGRATION_V15:'+file_sha(self.path),quarantine=True);ctl.close()
        migration.apply(self.path)
        self.assertTrue((self.batch/'CPU_MIGRATION_V15.json').exists())

    def test_ready_and_intent_each_block_migration(self):
        for name in ('READY_INTERNAL.json','RELEASE_INTENT.json'):
            p=self.old/name;p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'publication'):migration.apply(self.path)
            self.assertEqual(self.status(),'queued');p.unlink()

    def test_committed_ledger_blocks_even_without_ready_filename(self):
        with sqlite3.connect(self.args['database']) as db:
            db.execute('INSERT INTO releases VALUES (?,?,?)',(self.bid,'fake_receipt',1))
        with self.assertRaisesRegex(ValueError,'Committed'):migration.apply(self.path)
        self.assertEqual(self.status(),'queued')

    def test_changed_candidate_or_gate_blocks(self):
        Path(self.args['candidates']).write_text('tampered')
        with self.assertRaisesRegex(ValueError,'frozen input'):migration.apply(self.path)
        self.assertEqual(self.status(),'queued')
        Path(self.args['candidates']).write_text('{}')
        with patch.object(migration,'publication_quality',return_value={'status':'HOLD'}):
            with self.assertRaisesRegex(ValueError,'Quality gate'):migration.apply(self.path)
        self.assertEqual(self.status(),'queued')

    def test_changed_campaign_or_review_blocks(self):
        with patch.object(migration,'terminal_review',return_value={'terminal':True,'inflight':1}):
            with self.assertRaisesRegex(ValueError,'campaign changed'):migration.apply(self.path)
        self.assertEqual(self.status(),'queued')

    def test_route_rejects_path_escape(self):
        migration.apply(self.path)
        p=self.batch/'CPU_MIGRATION_V15.json';m=json.loads(p.read_text());m['new_output']='/tmp/unrelated';p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'path differs'):queue.release_output(self.batch)

    def test_arrow_scalar_matches_original_rowgroup_pointer(self):
        import pyarrow as pa,pyarrow.parquet as pq
        from small_agent.curation.release_gates_v5 import ParquetPointer
        from p0_parallel_near_v15 import ArrowPointer
        rows=['含中文 Unicode α\n'*i for i in range(33)]
        p=self.r/'texts.parquet';pq.write_table(pa.table({'text':rows}),p,row_group_size=7)
        old=ParquetPointer(p);new=ArrowPointer(p)
        for i in [0,15,32,7,8,31,1,2,15,14,0]:self.assertEqual(old.text(i),new.text(i))

    def test_raw_reader_preserves_nested_null_unicode_fields_and_adapter(self):
        import pyarrow as pa,pyarrow.parquet as pq
        from small_agent.curation.full_inventory_v11 import RawReader,adapter
        from p0_arrow_reader_v15 import ArrowRawReader
        rows=[dict(content='def f%d(x):\n    return x + 1\n# 数学 α'%i,language='python',repo_name='project'+str(i%3),
            path='code'+str(i)+'.py',detected_licenses=['mit','apache-2.0'] if i%2 else None,
            license='mit' if i%2==0 else '',url=None) for i in range(30)]
        p=self.r/'raw.parquet';pq.write_table(pa.Table.from_pylist(rows),p,row_group_size=7)
        file=dict(path=str(p),sha256=file_sha(p));src=dict(id='code',revision='fixed',domain='code',language='unknown',terms_scope='record_license_only')
        old=RawReader();new=ArrowRawReader()
        for i in [0,29,15,7,8,1,2,14,15,29,0]:
            a=old.row(file,i,None);b=new.row(file,i,None);self.assertEqual(a,b)
            self.assertEqual(adapter(a,src,file,i,None),adapter(b,src,file,i,None))

    def test_raw_jsonl_byte_pointer_unchanged(self):
        from small_agent.curation.full_inventory_v11 import RawReader
        from p0_arrow_reader_v15 import ArrowRawReader
        p=self.r/'raw.jsonl';first=json.dumps({'text':'α\n 中文'},ensure_ascii=False).encode()+b'\n'
        p.write_bytes(first+b'{"text":"second"}\n');f={'path':str(p)}
        self.assertEqual(RawReader().row(f,1,len(first)),ArrowRawReader().row(f,1,len(first)))

    def test_historical_publisher_reuses_its_bound_performance_config(self):
        perf=self.r/'old_performance.json';perf.write_text('{"release_cpu_workers":8}')
        p=self.old/'INPUT_BINDING.json';j=json.loads(p.read_text());j['performance_config_path']=str(perf);j['performance_config_sha256']=file_sha(perf);p.write_text(json.dumps(j))
        self.assertEqual(queue.performance_for(['--output',str(self.old)],'/new_config.json'),str(perf))
        perf.write_text('tampered')
        with self.assertRaisesRegex(ValueError,'Historical performance'):queue.performance_for(['--output',str(self.old)],'/new_config.json')

    def test_secondary_indexes_change_no_rows_or_selected_samples(self):
        from p0_prepare_queue_indexes_v15 import build
        d=queue.open_store(self.r)
        d.executemany('INSERT INTO files(id,path,sha,source,state) VALUES (?,?,?,?,?)',[(1,'a','sha','code_a','complete'),(2,'b','sha','code_b','complete')])
        rows=[]
        for i in range(400):
            rows.append((str(i),1+i%2,i,None,'code','cpp','family'+str(i),'train' if i%5 else 'test',100,digest(i),'short/code/cpp','CANDIDATE',None,'CANDIDATE'))
        d.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',rows);d.commit()
        before=[tuple(x) for x in d.execute('SELECT * FROM records ORDER BY h')]
        expected=queue.sample(d,'code_a','code','cpp',64,gate_clause="r.initial_gate='CANDIDATE'")
        expected=([dict(x) for x in expected[0]],expected[1])
        expected_cohorts=[tuple(x) for x in queue.source_cohorts(d,'code_a')]
        receipt=self.r/'INDEXES.json';build(self.r/'stock.sqlite',receipt)
        actual=queue.sample(d,'code_a','code','cpp',64,gate_clause="r.initial_gate='CANDIDATE'")
        self.assertEqual(expected,([dict(x) for x in actual[0]],actual[1]))
        self.assertEqual(expected_cohorts,[tuple(x) for x in queue.source_cohorts(d,'code_a')])
        self.assertEqual(before,[tuple(x) for x in d.execute('SELECT * FROM records ORDER BY h')]);d.close()
        self.assertEqual(json.loads(receipt.read_text())['corpus_rows_modified'],0)

    def test_ddl_failure_rolls_back_all_new_indexes(self):
        from p0_prepare_queue_indexes_v15 import build
        d=queue.open_store(self.r);d.execute('CREATE INDEX records_selection_v15 ON records(h)');d.commit();d.close()
        with self.assertRaisesRegex(ValueError,'definition differs'):build(self.r/'stock.sqlite',self.r/'INDEXES.json')
        with sqlite3.connect(self.r/'stock.sqlite') as db:
            self.assertIsNone(db.execute('SELECT name FROM sqlite_master WHERE name=?',('records_calibration_v15',)).fetchone())

    def test_source_first_allocation_matches_frozen_original(self):
        from p0_prepare_queue_indexes_v15 import build
        d=queue.open_store(self.r)
        d.executemany('INSERT INTO files(id,path,sha,source,state) VALUES (?,?,?,?,?)',[(1,'a','sha','code_a','complete'),(2,'b','sha','code_b','complete')])
        rows=[]
        for i in range(400):
            split='test' if i%7==0 else 'validation' if i%7==1 else 'train'
            rows.append((str(i),1+i%2,i,None,'code','cpp','family'+str(i),split,100,digest(i),'short/code/cpp','CANDIDATE',None,'CANDIDATE'))
        d.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',rows);d.commit()
        build(self.r/'stock.sqlite',self.r/'INDEXES.json')
        prompt=self.r/'prompt.txt';prompt.write_text('fixed')
        cfg=dict(prompt=str(prompt),batch_train_body_tokens=1000,smoke_train_body_tokens=500,maximum_batch_documents=20)
        scope={'approved_strata':['short/code/cpp']}
        for mode in ('source_audit','individual'):
            mode_scope=scope if mode=='source_audit' else {'approved_strata':[]}
            expected=queue.original.allocate(cfg,d,'code_a','code','cpp',mode_scope,mode,False)
            expected=(expected[0],[dict(x) for x in expected[1]])
            d.execute('UPDATE records SET job=NULL');d.execute('DELETE FROM jobs');d.commit()
            actual=queue.allocate(cfg,d,'code_a','code','cpp',mode_scope,mode,False)
            self.assertEqual(expected,(actual[0],[dict(x) for x in actual[1]]))
            d.execute('UPDATE records SET job=NULL');d.execute('DELETE FROM jobs');d.commit()
        d.close()

if __name__=='__main__':unittest.main()
