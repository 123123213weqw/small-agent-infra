"""Engineering fixtures only. Mocked teacher decisions are NOT corpus approval."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from small_agent.curation.full_inventory_v11 import adapter,rule,open_store,AllStockControl
from small_agent.curation.supply_control import file_sha,digest,SupplyControl
from small_agent.curation.release_gates_v5 import write,sha
import small_agent.curation.quality_v11 as q
import p0_full_inventory_scan_v11 as scanner
import p0_release_inventory_batch_v11 as publisher
import p0_full_inventory_register_v11 as registrar
import p0_inventory_supply_v11 as queue
from test_math_release_v6 import MathPipelineTests

class InventoryTests(unittest.TestCase):
 def test_known_invalid_pilot_is_only_cohort_hold_but_unknown_is_global(self):
  s=dict(stop_reason='pilot_not_valid',counts={'invalid_outputs_quarantined':1},ledger={'requests_by_state':{'usage_estimated':1},'cap_exceeded':False})
  with self.assertRaises(ValueError) as e:queue.stopped_worker(s)
  self.assertNotIsInstance(e.exception,queue.CampaignHeld)
  s['counts']['unknown_or_fatal']=1
  with self.assertRaises(queue.CampaignHeld):queue.stopped_worker(s)
 def test_api_nonzero_exit_holds_whole_campaign_without_retry(self):
  import subprocess
  with tempfile.TemporaryDirectory() as d:
   with patch.object(queue,'component',side_effect=subprocess.CalledProcessError(3,['worker'])) as call:
    with self.assertRaises(queue.CampaignHeld):queue.score(dict(allow_api=True,profile='not-used'),Path(d))
    self.assertEqual(call.call_count,1)
 def test_r2_prompt_keeps_context_notice_semantics_explicit(self):
  prompt=(ROOT/'configs/curation/p0_inventory_audit_prompt_v11_r2.txt').read_text()
  self.assertIn('registration_prompt、ad_present、external_figure',prompt)
  self.assertIn('severity 必须是 info',prompt)
  original=json.loads((ROOT/'configs/curation/p0_full_inventory_supply_v11.json').read_text());overlay=json.loads((ROOT/'configs/curation/p0_full_inventory_supply_v11_r2.json').read_text())
  for key in ('root','database','tokenizer_sha256','shared_campaign_budget_cny','maximum_concurrency','auto_training'):self.assertEqual(original[key],overlay[key])
 def test_flat_raw_roots_keep_separate_sources(self):
  import sqlite3
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);census=r/'census';census.mkdir()
   c=sqlite3.connect(census/'census.sqlite');c.execute('create table shards(key text,receipt text)');c.close()
   write(r/'plan.json',dict(items=[]));write(r/'inventory.json',dict(raw_groups=[]))
   for name in ('fineweb_edu_wave2_v1','finewiki_zh_2025_candidate_v1'):
    p=r/name/'raw/a.parquet';p.parent.mkdir(parents=True);p.write_bytes(b'fixture')
    if 'finewiki' in name:write(r/name/'raw_manifest.json',dict(revision='wiki-rev',files=[dict(file=p.name,sha256=file_sha(p))]))
    else:write(r/name/'manifest.json',dict(revision='web-rev',items=[dict(path=p.name,sha256=file_sha(p))]))
   cfg=dict(raw_root=str(r),root=str(r/'stock'),source_inventory=str(r/'inventory.json'),census_root=str(census),download_plan=str(r/'plan.json'),tokenizer_sha256='fixture')
   result=registrar.register(cfg)
   self.assertEqual({s['id'] for s in result['sources']},{'fineweb_edu_wave2_v1','finewiki_zh_2025_candidate_v1'})
   self.assertEqual({s['domain'] for s in result['sources']},{'general_en','general_zh'})
 def test_code_license_and_verbatim_body(self):
  s=dict(id='fixture',domain='code',language='unknown',revision='a',terms_scope='record_license_only')
  f=dict(path='test.parquet',sha256='a'*64)
  text='def f():\n    return " A  B "\n'
  r=adapter(dict(code=text,path='lib.py',language='Python'),s,f,0)
  self.assertEqual(r['text'],text);self.assertEqual(r['language'],'python')
  self.assertEqual(rule(r),('HOLD','record_license_unresolved'))
  r['record_license']='mit';self.assertEqual(rule(r)[0],'CANDIDATE')
 def test_all_parser_control_pairs(self):
  for lang in q.CODE:
   rows,labels,verified=q.control_rows('code',lang)
   self.assertEqual(len(rows),32);self.assertEqual(sum(x['expected']=='keep' for x in labels),16)
   self.assertTrue(all(x['passed'] for x in verified))
 def test_atomic_jsonl_scan_resume_and_frozen_engine(self):
  from tokenizers import Tokenizer,models,pre_tokenizers
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);tok=Tokenizer(models.WordLevel({'<unk>':3,'a':15,'b':16},unk_token='<unk>'));tok.pre_tokenizer=pre_tokenizers.Whitespace();tok.save(str(r/'tok.json'))
   recipe=json.loads((ROOT/'configs/curation/base_1b_supply_20b_v4.json').read_text());recipe['tokenizer_sha256']=file_sha(r/'tok.json');c=SupplyControl(r/'ledger',recipe);c.close()
   f=r/'raw.jsonl';f.write_text(''.join(json.dumps(dict(text='a '*50+'b '*i,url='https://example.org/'+str(i)))+'\n' for i in range(1,601)))
   source=dict(id='fixture',domain='general_en',language='en',revision='a',terms_scope='internal',files=[dict(path=str(f),sha256=file_sha(f),normalized=False)])
   cfg=dict(root=str(r/'stock'),tokenizer=str(r/'tok.json'),database=str(r/'ledger'),reserved_tokens=['<eos>'])
   write(r/'stock/REGISTRY.json',dict(tokenizer_sha256=file_sha(r/'tok.json'),files=1,sources=[source]))
   self.assertEqual(scanner.scan(cfg,257)['rows'],257)
   db=open_store(r/'stock');self.assertEqual(db.execute('select count(*) from records').fetchone()[0],257);self.assertEqual(db.execute('select next_pos from files').fetchone()[0],257);db.close()
   scanner.scan(cfg);db=open_store(r/'stock');self.assertEqual(db.execute('select count(*) from records').fetchone()[0],600);self.assertEqual(db.execute('select state from files').fetchone()[0],'complete');db.close()
   scanner.scan(cfg);db=open_store(r/'stock');self.assertEqual(db.execute('select count(*) from records').fetchone()[0],600);db.close()
   with patch.object(scanner,'file_sha',side_effect=lambda p:'changed' if p.name=='p0_full_inventory_scan_v11.py' else file_sha(p)),self.assertRaisesRegex(ValueError,'Frozen scanner'):scanner.scan(cfg)
 def test_source_failed_stratum_and_individual_full_coverage(self):
  import pyarrow as pa,pyarrow.parquet as pq
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);rows=[dict(normalized_sha256=str(i),source_id='fixture',domain='general_en',language='en',own_text_tokens=100) for i in range(12)]
   pq.write_table(pa.Table.from_pylist(rows),r/'data.parquet')
   source=dict(meta=dict(strata={str(i):'short/general_en/en' for i in range(12)},source_id='fixture',domain='general_en',language='en'),body_ids=[str(i) for i in range(12)])
   write(r/'source/PLAN.json',source)
   write(r/'ref/REFERENCE_V11.json',dict(domain='general_en',language='en'));write(r/'ref/labels.json',[])
   plan=dict(meta=dict(candidate_sha256=file_sha(r/'data.parquet'),mode='individual',source_manifest_sha256=file_sha(r/'source/PLAN.json')),body_ids=['0'])
   write(r/'audit/PLAN.json',plan);cfg=dict(release_policy=str(ROOT/'configs/curation/p0_release_gates_v5.json'))
   with patch.object(q,'reference_gate',return_value={'status':'passed'}),patch.object(q,'annotations',return_value=({},set(),'state','manifest')):
    self.assertEqual(q.source_gate(r/'source',json.loads(Path(cfg['release_policy']).read_text())['quality'])['approved_strata'],[])
    with self.assertRaisesRegex(ValueError,'Every document'):q.publication_quality(cfg,r/'data.parquet',r/'source',r/'ref',r/'audit','individual')
    plan['body_ids']=[str(i) for i in range(12)];(r/'audit/PLAN.json').unlink();write(r/'audit/PLAN.json',plan)
    self.assertEqual(q.publication_quality(cfg,r/'data.parquet',r/'source',r/'ref',r/'audit','individual')['status'],'hold')
    (r/'ref/REFERENCE_V11.json').unlink();write(r/'ref/REFERENCE_V11.json',dict(domain='math',language='en'))
    with self.assertRaisesRegex(ValueError,'Wrong independent'):q.publication_quality(cfg,r/'data.parquet',r/'source',r/'ref',r/'audit','individual')
 def test_generic_real_pack_recovery_all_domains(self):
  import pyarrow as pa,pyarrow.parquet as pq
  for domain in ('general_en','general_zh','technical','code','math'):
   with self.subTest(domain=domain),tempfile.TemporaryDirectory() as d:
    r=Path(d);argv,out,db,recipe,quality,_=MathPipelineTests().fixture(r)
    argv[argv.index('--reference-batch'):argv.index('--reference-batch')+2]=[]
    rows=pq.read_table(r/'candidates.parquet').to_pylist()
    for row in rows:row.update(domain=domain,record_license='mit')
    pq.write_table(pa.Table.from_pylist(rows),r/'candidates.parquet')
    terms=json.loads((r/'terms.json').read_text());terms['dataset_license']='internal-fixture';(r/'terms.json').write_text(json.dumps(terms))
    write(r/'registry/REGISTRY.json',{});write(r/'config.json',dict(root=str(r/'registry')))
    (r/'candidate_binding.json').unlink();write(r/'candidate_binding.json',dict(candidate_sha256=file_sha(r/'candidates.parquet'),source_id='fixture',domain=domain,language='en',registry_sha256=file_sha(r/'registry/REGISTRY.json')))
    write(r/'source/PLAN.json',dict(meta=dict(source_id='fixture')))
    argv+=['--config',str(r/'config.json'),'--domain',domain,'--mode','individual']
    quality.update(kept_ids=[x['normalized_sha256'] for x in rows],excluded_ids=[],batch_state_sha256='mocked_engineering_only')
    realwrite=publisher.write
    def interruption(path,value):
     if Path(path).name=='READY_INTERNAL.json':raise RuntimeError('controlled after ledger commit')
     return realwrite(path,value)
    with patch.object(sys,'argv',argv),patch.object(publisher,'publication_quality',return_value=quality),patch.object(publisher,'write',side_effect=interruption):
     with self.assertRaisesRegex(RuntimeError,'controlled'):publisher.main()
    self.assertFalse((out/'READY_INTERNAL.json').exists())
    with patch.object(sys,'argv',argv):publisher.main();publisher.main()
    ctl=AllStockControl(db,recipe)
    try:self.assertEqual(ctl.status()['unique_ready_train_body_tokens'],512);self.assertEqual(ctl.db.execute('select count(*) from releases').fetchone()[0],1)
    finally:ctl.close()

if __name__=='__main__':unittest.main()
