"""CPU fixtures; these prove transport equivalence, NOT corpus quality."""
import json,sqlite3,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from tokenizers import Tokenizer,models,pre_tokenizers,trainers,decoders
import pyarrow as pa,pyarrow.parquet as pq
import p0_distributed_scan_v12 as coord
import p0_distributed_worker_v12 as worker
import p0_full_inventory_scan_v11 as original
from small_agent.curation.full_inventory_v11 import open_store
from small_agent.curation.supply_control import file_sha,digest

class DistributedTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.r=Path(self.tmp.name)
  text='A clear technical document explains integer arithmetic and example code, including input and output. '
  self.rows=[dict(text=text+str(i),url='https://example.org/'+str(i)) for i in range(8)];self.rows+=[self.rows[0],dict(text='tiny')]
  tok=Tokenizer(models.BPE());tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);tok.decoder=decoders.ByteLevel()
  tok.train_from_iterator([x['text'] for x in self.rows],trainers.BpeTrainer(vocab_size=300,special_tokens=['<special'+str(i)+'>' for i in range(15)],initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
  self.token=self.r/'tokenizer.json';tok.save(str(self.token));self.p=self.r/'raw.parquet';pq.write_table(pa.Table.from_pylist(self.rows),self.p,row_group_size=3)
  self.source=dict(id='fixture',domain='general_en',language='en',revision='fixture',terms_scope='internal_source_level')
  self.file=dict(path=str(self.p),sha256=file_sha(self.p));self.ledger=self.r/'supply.sqlite'
  with sqlite3.connect(self.ledger) as d:
   d.executescript('CREATE TABLE documents(body_sha TEXT,split TEXT);CREATE TABLE legacy_families(family_id TEXT,split TEXT);')
  self.cfg=dict(distributed_root=str(self.r/'distributed'),tokenizer=str(self.token),tokenizer_sha256=file_sha(self.token),registry_sha256='f'*64,reserved_tokens=[],database=str(self.ledger))
 def task(self,start=0,end=10,byte=0):
  return dict(coord.task_for(self.cfg,self.source,self.file,start,end,byte),local_file=str(self.p),local_tokenizer=str(self.token))
 def output(self,task):
  out=self.r/'distributed/results'/worker.identity(task)/'output';worker.run(task,out);return out
 def db(self):
  db=open_store(self.r/'new');db.execute('ATTACH DATABASE ? AS supply',(str(self.ledger),));db.execute('INSERT INTO files(path,sha,source) VALUES (?,?,?)',(self.file['path'],self.file['sha256'],self.source['id']));db.commit();self.addCleanup(db.close);return db
 def test_cross_rowgroup_range_is_exact(self):
  task=self.task(2,7);out=self.output(task);r,rows=coord.verify_result(task,out);self.assertEqual([x['pos'] for x in rows],list(range(2,7)));self.assertEqual(r['rows'],5)
 def test_whole_scan_matches_frozen_v11(self):
  task=self.task();self.output(task);db=self.db();coord.import_result(db,self.cfg,task)
  old=self.r/'old';old.mkdir();registry=dict(tokenizer_sha256=file_sha(self.token),files=1,sources=[dict(self.source,files=[self.file])]);(old/'REGISTRY.json').write_text(json.dumps(registry))
  original.scan(dict(self.cfg,root=str(old)));od=open_store(old);self.addCleanup(od.close)
  columns='h,pos,byte,domain,lang,family,split,n,rank,stratum,gate,initial_gate'
  self.assertEqual([tuple(x) for x in db.execute('SELECT '+columns+' FROM records ORDER BY h')],[tuple(x) for x in od.execute('SELECT '+columns+' FROM records ORDER BY h')])
  self.assertEqual(json.loads(db.execute('SELECT stats FROM files').fetchone()[0]),json.loads(od.execute('SELECT stats FROM files').fetchone()[0]))
 def test_restart_completed_is_idempotent(self):
  task=self.task();out=self.output(task);before=file_sha(out/'COMPLETE.json');worker.run(task,out);self.assertEqual(file_sha(out/'COMPLETE.json'),before)
 def test_restart_after_part_commit(self):
  task=self.task();out=self.output(task);(out/'COMPLETE.json').unlink();r=worker.run(task,out);self.assertEqual(r['rows'],10);self.assertEqual(len(r['parts']),1)
 def test_changed_worker_rejected_even_when_completed(self):
  task=self.task();out=self.output(task);task['worker_sha256']='0'*64
  with self.assertRaises(ValueError):worker.run(task,out)
 def test_tampered_result_rejected_before_import(self):
  task=self.task();out=self.output(task);(out/'part-000000.parquet').write_bytes(b'broken');db=self.db()
  with self.assertRaises(ValueError):coord.import_result(db,self.cfg,task)
  self.assertEqual(db.execute('SELECT next_pos FROM files').fetchone()[0],0)
 def test_out_of_order_cannot_skip_cursor(self):
  task=self.task(3,8);self.output(task);db=self.db()
  with self.assertRaises(ValueError):coord.import_result(db,self.cfg,task)
 def test_reimport_does_not_repeat_rows(self):
  task=self.task();self.output(task);db=self.db();self.assertTrue(coord.import_result(db,self.cfg,task));self.assertFalse(coord.import_result(db,self.cfg,task));self.assertEqual(db.execute('SELECT next_pos FROM files').fetchone()[0],10)
 def test_ranges_resume_without_duplicates_or_gaps(self):
  db=self.db()
  for a,b in [(0,3),(3,6),(6,10)]:
   task=self.task(a,b);self.output(task);coord.import_result(db,self.cfg,task)
  self.assertEqual(db.execute('SELECT next_pos FROM files').fetchone()[0],10);self.assertEqual(json.loads(db.execute('SELECT stats FROM files').fetchone()[0])['rows'],10)
 def test_global_ledger_and_family_split_enforced(self):
  task=self.task();out=self.output(task);_,rows=coord.verify_result(task,out)
  with sqlite3.connect(self.ledger) as d:
   d.execute('INSERT INTO documents VALUES (?,?)',(rows[0]['h'],rows[0]['split']));d.execute('INSERT INTO legacy_families VALUES (?,?)',(rows[1]['family'],'test' if rows[1]['split']!='test' else 'train'))
  db=self.db();coord.import_result(db,self.cfg,task);self.assertIsNone(db.execute('SELECT h FROM records WHERE h=?',(rows[0]['h'],)).fetchone());self.assertEqual(db.execute('SELECT gate FROM records WHERE h=?',(rows[1]['h'],)).fetchone()[0],'HOLD')
 def test_jsonl_byte_boundary_and_bad_json(self):
  self.p=self.r/'raw.jsonl';data=[b'not-json\n']+[(json.dumps(x)+'\n').encode() for x in self.rows];self.p.write_bytes(b''.join(data));self.file=dict(path=str(self.p),sha256=file_sha(self.p));db=self.db()
  task=self.task(0,3);self.output(task);coord.import_result(db,self.cfg,task)
  task=self.task(3,11,sum(map(len,data[:3])));self.output(task);coord.import_result(db,self.cfg,task)
  self.assertEqual(db.execute('SELECT next_byte FROM files').fetchone()[0],sum(map(len,data)))
 def test_completed_coordinator_restart_does_not_recompute(self):
  db=self.db();root=self.r/'new';reg=dict(files=1,sources=[dict(self.source,files=[self.file])]);(root/'REGISTRY.json').write_text(json.dumps(reg))
  binding=digest(dict(registry=file_sha(root/'REGISTRY.json'),adapter=file_sha(ROOT/coord.CODE[1]),scanner=file_sha(ROOT/'scripts/p0_full_inventory_scan_v11.py')))
  db.execute("INSERT INTO meta VALUES ('scan_identity',?)",(binding,));db.execute("UPDATE files SET state='complete',next_pos=10");db.commit()
  cfg=dict(self.cfg,root=str(root),nodes=[]);terminal=dict(stage='CPU_DONE_NOT_APPROVED',registry_sha256=file_sha(root/'REGISTRY.json'),files=1,held_files=0,nodes={})
  p=self.r/'distributed/COMPLETE.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(terminal))
  self.assertEqual(coord.run(cfg),terminal);self.assertEqual(coord.run(cfg),terminal)
  self.assertTrue((root/'SCAN_COMPLETE.json').exists())
 def test_unlicensed_code_stays_hold(self):
  self.source['domain']='code';self.source['language']='python';task=self.task();out=self.output(task);_,rows=coord.verify_result(task,out);self.assertEqual(rows[0]['gate'],'HOLD');self.assertEqual(rows[0]['reason'],'record_license_unresolved')

if __name__=='__main__':unittest.main()
