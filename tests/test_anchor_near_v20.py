"""Lossless candidate retrieval: all original votes/dense gates, independent small sketches."""
import random,sqlite3,struct,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from test_release_speed_v18 import SpeedTests
from p0_parallel_near_v20 import ReadOnlyIndex,ParallelNearIndex,exact_anchor_query

class AnchorTests(SpeedTests):
 def test_near_decisions_and_exact_reused_sketches(self):
  import test_release_speed_v18 as old
  with patch.object(old,'ParallelNearIndex',ParallelNearIndex):super().test_near_decisions_and_exact_reused_sketches()
 def test_independent_candidate_superset_and_dense_limits(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'postings.sqlite'
   with sqlite3.connect(p) as db:
    db.executescript('CREATE TABLE docs(id INTEGER PRIMARY KEY,body BLOB,split TEXT,source INTEGER,rowno INTEGER,n INTEGER,k INTEGER);CREATE TABLE postings(h BLOB,d INTEGER,PRIMARY KEY(h,d)) WITHOUT ROWID;')
    rng=random.Random(20)
    for i in range(300):
     n=rng.choice([1,3,5,20,40,63,64,100,200]);sk=rng.sample(range(100),min(n,64));db.execute('INSERT INTO docs VALUES(?,?,?,?,?,?,?)',(i,bytes([i%256]),'train',1,i,n,len(sk)))
     db.executemany('INSERT INTO postings VALUES(?,?)',[(struct.pack('<Q',x),i) for x in sk])
   ctx=ReadOnlyIndex(td)
   try:
    for n in [1,3,5,20,40,63,64,100,200]:
     for limit in [0,10,2000]:
      sk=rng.sample(range(100),min(n,64));lo=__import__('math').ceil(n*.9);hi=__import__('math').floor(n/.9)
      query='SELECT d.id,d.body,d.split,d.source,d.rowno,d.n,d.k,COUNT(*) c FROM postings p JOIN docs d ON d.id=p.d WHERE p.h IN ('+','.join('?' for _ in sk)+') AND d.n>=? AND d.n<=? GROUP BY d.id HAVING c>=MAX(1,CAST((MIN(d.k,?)*?+0.999999) AS INTEGER)) LIMIT ?'
      args=[struct.pack('<Q',h) for h in sk]+[lo,hi,len(sk),.5,limit+1]
      expected=ctx.db.execute(query,args).fetchall();policy=dict(sketch_size=64,threshold=.9,minimum_shared_fraction=.5,maximum_pair_checks_per_document=limit,_anchor_shape_verified=True)
      self.assertEqual(expected,exact_anchor_query(ctx,sk,n,policy,query,args))
   finally:ctx.close()
if __name__=='__main__':unittest.main()
