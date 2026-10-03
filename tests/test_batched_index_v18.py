import json,sqlite3,unittest
from test_parallel_cpu_v13 import ParallelTests
from p0_batched_index_v18 import BatchedDiskNearIndex
from small_agent.curation.stock_v6 import DiskNearIndex
from small_agent.curation.supply_control import file_sha

class IndexTests(ParallelTests):
 def test_old_new_index_identical_and_resume(self):
  ledger=self.r/'ledger.sqlite'
  with sqlite3.connect(ledger) as c:
   c.execute('CREATE TABLE documents(body_sha TEXT PRIMARY KEY,split TEXT,domain TEXT,tokens INTEGER)');c.execute('CREATE TABLE releases(batch_id TEXT)')
   c.executemany('INSERT INTO documents VALUES(?,?,?,?)',[(x['normalized_sha256'],x['split'],x['domain'],x['own_text_tokens']) for x in self.rows])
  legacy=self.r/'legacy.json';legacy.write_text(json.dumps(dict(evidence_files=[dict(path=str(self.source),sha256=file_sha(self.source))])))
  old=DiskNearIndex(self.r/'old');new=BatchedDiskNearIndex(self.r/'new');self.addCleanup(old.close);self.addCleanup(new.close)
  a=old.update(ledger,legacy,self.r/'cache');b=new.update(ledger,legacy,self.r/'cache')
  for table in ('sources','docs','postings'):
   self.assertEqual(old.db.execute('SELECT * FROM '+table+' ORDER BY 1,2').fetchall(),new.db.execute('SELECT * FROM '+table+' ORDER BY 1,2').fetchall())
  self.assertEqual(json.loads(a.read_text())['ledger'],json.loads(b.read_text())['ledger'])
  self.assertEqual(b,new.update(ledger,legacy,self.r/'cache'))
  self.source.write_bytes(b'changed')
  with self.assertRaises(ValueError):new.update(ledger,legacy,self.r/'cache')

if __name__=='__main__':unittest.main()
