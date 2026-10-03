"""Transport repair cannot admit unassigned rows; unchanged frozen worker."""
import json
from pathlib import Path
from unittest.mock import patch
import unittest
import pyarrow as pa,pyarrow.parquet as pq
import test_distributed_scan_v12 as old
import p0_distributed_scan_v18 as optimized
from small_agent.curation.supply_control import file_sha
from small_agent.curation.release_gates_v5 import sha

class ScanTests(old.DistributedTests):
 def setUp(self):
  super().setUp()
  p=patch.object(old,'coord',optimized);p.start();self.addCleanup(p.stop)
 def normalized(self):
  self.source.update(normalized=True)
  for i,r in enumerate(self.rows):
   r.update(normalized_sha256=sha(r['text']),family_id=sha('family'+str(i)),split='unassigned' if i==2 else 'train',own_text_tokens=100,domain='math',language='en')
  pq.write_table(pa.Table.from_pylist(self.rows),self.p,row_group_size=3);self.file['sha256']=file_sha(self.p)
 def test_unassigned_is_recomputed_and_remains_held(self):
  self.normalized();task=self.task();self.output(task);db=self.db();optimized.import_result(db,self.cfg,task)
  row=db.execute('SELECT gate,split FROM records WHERE pos=2').fetchone()
  self.assertEqual(tuple(row),('HOLD','unassigned'));self.assertEqual(db.execute('SELECT next_pos FROM files').fetchone()[0],10)
 def forge(self,gate):
  self.normalized();task=self.task();out=self.output(task)
  m=json.loads((out/'COMPLETE.json').read_text());part=m['parts'][0];p=out/part['name'];rows=pq.read_table(p).to_pylist();rows[2]['gate']=gate
  pq.write_table(pa.Table.from_pylist(rows),p);part['sha256']=file_sha(p);(out/'COMPLETE.json').write_text(json.dumps(m));return task,out
 def test_unassigned_candidate_fails_before_commit(self):
  task,out=self.forge('CANDIDATE');db=self.db()
  with self.assertRaisesRegex(ValueError,'split'):optimized.import_result(db,self.cfg,task)
  self.assertEqual(db.execute('SELECT next_pos FROM files').fetchone()[0],0)
 def test_forged_unassigned_drop_fails_full_recompute(self):
  task,out=self.forge('DROP');db=self.db()
  with self.assertRaisesRegex(ValueError,'recomputation'):optimized.import_result(db,self.cfg,task)
  self.assertEqual(db.execute('SELECT next_pos FROM files').fetchone()[0],0)

if __name__=='__main__':unittest.main()
