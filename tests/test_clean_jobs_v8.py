import json,sys,tempfile,unittest,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p0_stock_plan_v6 import open_plan
from p0_math_clean_jobs_v8 import allocate_clean
from small_agent.curation.supply_control import file_sha
class CleanJobTests(unittest.TestCase):
 def test_only_filtered_records_and_idempotence(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);old=r/'old';old.mkdir();(old/'STOCK_REGISTERED.json').write_text('{}');db=open_plan(old)
   with db:
    db.execute("INSERT INTO sources(id,path,sha,source_id,revision,complete) VALUES (1,'fixture','hash','finemath_4plus_supply_v4','rev',1)")
    for h,pos,s in [('a',0,'train'),('b',1,'validation'),('c',2,'test'),('d',3,'train')]:db.execute('INSERT INTO records(h,sid,pos,split,n,rank,gate) VALUES (?,?,?,?,?,?,?)',(h,1,pos,s,100,b'0'+h.encode(),'held'))
   db.close();f=r/'filtered';f.mkdir();e=sqlite3.connect(f/'eligible.sqlite');e.execute('CREATE TABLE eligible(h TEXT,sid INTEGER,pos INTEGER,split TEXT,n INTEGER,stratum TEXT)');e.executemany('INSERT INTO eligible VALUES (?,1,?,?,100,?)',[('a',0,'train','short/en'),('b',1,'validation','short/en'),('c',2,'test','short/en')]);e.commit();e.close()
   (f/'FILTER_DATABASE_BOUND.json').write_text(json.dumps(dict(database_sha256=file_sha(f/'eligible.sqlite'),evidence_files=[dict(path=str(f/'eligible.sqlite'),sha256=file_sha(f/'eligible.sqlite'))])))
   scope=dict(status='passed',approved_strata=['short/en'],known_audit_bad_body_sha256=[])
   self.assertEqual(allocate_clean(r/'new',old,f,scope),1);self.assertEqual(allocate_clean(r/'new',old,f,scope),1)
   db=open_plan(old)
   try:self.assertIsNone(db.execute("SELECT job FROM records WHERE h='d'").fetchone()[0]);self.assertTrue(db.execute("SELECT job FROM records WHERE h='a'").fetchone()[0]>0)
   finally:db.close()
 def test_hold_scope_cannot_allocate(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t)
   with self.assertRaises(ValueError):allocate_clean(r,r,r,dict(status='hold'))
if __name__=='__main__':unittest.main()
