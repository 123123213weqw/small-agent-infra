"""Actual CPU ranges under simulated node transport; no API or quality promotion."""
import json,sqlite3,time,unittest
from pathlib import Path
from unittest.mock import patch
import test_distributed_scan_v12 as old
import p0_distributed_affinity_v19 as q
from small_agent.curation.supply_control import file_sha,digest

class AffinityTests(old.DistributedTests):
 def test_three_file_lanes_keep_registry_commit_order_and_restart(self):
  db=self.db();root=self.r/'new';files=[]
  for i in range(3):
   p=self.r/('raw'+str(i)+'.parquet');p.write_bytes(self.p.read_bytes());f=dict(path=str(p),sha256=file_sha(p));files.append(f)
  with db:
   db.execute('DELETE FROM files')
   db.executemany('INSERT INTO files(path,sha,source) VALUES(?,?,?)',[(f['path'],f['sha256'],self.source['id']) for f in files])
  reg=dict(files=3,sources=[dict(self.source,files=files)]);(root/'REGISTRY.json').write_text(json.dumps(reg))
  binding=digest(dict(registry=file_sha(root/'REGISTRY.json'),adapter=file_sha(q.base.ROOT/q.base.CODE[1]),scanner=file_sha(q.base.ROOT/'scripts/p0_full_inventory_scan_v11.py')))
  with db:db.execute("INSERT INTO meta VALUES ('scan_identity',?)",(binding,))
  nodes=[dict(id='n'+str(i),workers=2,root=str(self.r/('node'+str(i))),python='unused') for i in range(3)]
  cfg=dict(self.cfg,root=str(root),nodes=nodes,range_rows=3)
  commits=[];original=q.base.import_result
  def execute(node,cfg,task):
   if node['id']=='n0':time.sleep(.01)
   task=dict(task,local_file=task['file']['path'],local_tokenizer=str(self.token))
   out=Path(cfg['distributed_root'])/'results'/q.base.worker.identity(task)/'output';r=old.worker.run(task,out)
   return dict(node=node['id'],rows=r['rows'],elapsed_seconds=r['elapsed_seconds'])
  def commit(db,cfg,task,*a):
   commits.append(task['file']['path']);return original(db,cfg,task,*a)
  with patch.object(q.base,'prepare',side_effect=lambda n,*a:n),patch.object(q.base,'execute',side_effect=execute),patch.object(q.base,'import_result',side_effect=commit):
   q.run(cfg);count=db.execute('SELECT COUNT(*) FROM records').fetchone()[0];q.run(cfg)
  self.assertEqual(commits,[f['path'] for f in files for _ in range(4)])
  self.assertEqual([tuple(r) for r in db.execute('SELECT state,next_pos FROM files ORDER BY id')],[('complete',10)]*3)
  self.assertEqual(count,db.execute('SELECT COUNT(*) FROM records').fetchone()[0])
  self.assertTrue(all('CANDIDATE' in r[0] or r[0]=='DROP' for r in db.execute('SELECT gate FROM records')))

if __name__=='__main__':unittest.main()
