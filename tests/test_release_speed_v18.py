import heapq,json,os,sqlite3,unittest
from pathlib import Path
from unittest.mock import patch
from test_parallel_cpu_v13 import ParallelTests
from small_agent.curation.stock_v6 import DiskNearIndex
from small_agent.curation.release_gates_v5 import lexical,sha
from small_agent.curation.supply_control import file_sha
from p0_parallel_near_v18 import ParallelNearIndex
from p0_sketch_cache_v18 import store_checked_sketch

class SpeedTests(unittest.TestCase):
 setUp=ParallelTests.setUp
 def test_near_decisions_and_exact_reused_sketches(self):
  import pyarrow as pa,pyarrow.parquet as pq
  ledger=self.r/'ledger.sqlite'
  with sqlite3.connect(ledger) as db:
   db.execute('CREATE TABLE documents(body_sha TEXT,split TEXT,domain TEXT,tokens INTEGER)');db.execute('CREATE TABLE releases(batch_id TEXT)')
   db.executemany('INSERT INTO documents VALUES(?,?,?,?)',[(r['normalized_sha256'],r['split'],r['domain'],r['own_text_tokens']) for r in self.rows])
  legacy=self.r/'legacy.json';legacy.write_text(json.dumps(dict(evidence_files=[dict(path=str(self.source),sha256=file_sha(self.source))])))
  old=DiskNearIndex(self.r/'index');new=ParallelNearIndex(self.r/'index');self.addCleanup(old.close);self.addCleanup(new.close)
  manifest=old.update(ledger,legacy,self.r/'oldcache')
  rows=[dict(r,text=r['text']+'# comment\n',normalized_sha256=sha(r['text']+'# comment\n')) for r in self.rows]
  fresh='Unique symbols and fresh mathematics are present in this fully separate document. '+('consider different integers '*20)
  rows+=[dict(self.rows[0],text=fresh,normalized_sha256=sha(fresh)),dict(self.rows[0],text=fresh+'\n',normalized_sha256=sha(fresh+'\n'))]
  policy=dict(threshold=.9,sketch_size=64,minimum_shared_fraction=.5,maximum_pair_checks_per_document=2000)
  self.assertEqual(old.filter(rows,manifest,policy,ledger),new.filter(rows,manifest,policy,ledger,2))
  kept,_=new.filter(rows,manifest,policy,ledger,2)
  for i,(r,s) in enumerate(zip(kept,new.retained_sketch_rows)):
   values=lexical(r['text']);self.assertEqual(s,dict(body_sha256=r['normalized_sha256'],split=r['split'],row=i,shingle_count=len(values),sketch=heapq.nsmallest(64,values)))
  p=self.r/'dedup.parquet';pq.write_table(pa.Table.from_pylist(kept),p);entry=dict(path=str(p),sha256=file_sha(p))
  m=store_checked_sketch(entry,new.retained_sketch_rows,self.r/'reuse')
  self.assertEqual(pq.read_table(m['path']).to_pylist(),new.retained_sketch_rows)
  self.assertEqual(m,store_checked_sketch(entry,new.retained_sketch_rows,self.r/'reuse'))
  Path(m['path']).write_bytes(b'corrupt')
  with self.assertRaisesRegex(ValueError,'cache changed'):store_checked_sketch(entry,new.retained_sketch_rows,self.r/'reuse')
  self.source.write_bytes(b'changed')
  with self.assertRaisesRegex(ValueError,'Reference changed'):new.filter(rows,manifest,policy,ledger,2)
 def test_real_five_domain_atomic_pack_recovery(self):
  import test_full_inventory_v11 as oldtests
  import p0_release_inventory_batch_v18 as pub
  perf=self.r/'perf.json';perf.write_text(json.dumps(dict(release_cpu_workers=2)))
  original_write=oldtests.write
  def fixture_write(path,value):
   if Path(path).name=='config.json':value=dict(value,sketch_root=str(Path(path).parent/'sketch-reuse'))
   return original_write(path,value)
  with patch.dict(os.environ,P0_PARALLEL_CONFIG=str(perf)),patch.object(oldtests,'publisher',pub),patch.object(oldtests,'write',side_effect=fixture_write):
   oldtests.InventoryTests().test_generic_real_pack_recovery_all_domains()

if __name__=='__main__':unittest.main()
