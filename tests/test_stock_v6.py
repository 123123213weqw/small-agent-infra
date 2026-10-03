"""Offline engineering fixtures only; never certify natural-corpus facts or rights."""
import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p0_math_reference_v6 import arithmetic,verified_steps
from small_agent.curation.stock_v6 import DiskNearIndex,layer
from small_agent.curation.release_gates_v5 import sha,global_near,build_inventory_index,ledger_snapshot
from small_agent.curation.supply_control import SupplyControl,file_sha

class ArithmeticTests(unittest.TestCase):
    def test_exact_fraction(self):self.assertEqual(str(arithmetic('3/4+0.25')),'1')
    def test_code_and_large_numbers_rejected(self):
        for text in ["__import__('os').system('true')",'a+2','10**9999','100000000000000000000']:
            with self.assertRaises((ValueError,SyntaxError)):arithmetic(text)
    def test_wrong_annotation_rejected(self):
        with self.assertRaises(ValueError):verified_steps('claim <<2+2=5>>')
    def test_verified_annotation(self):self.assertEqual(verified_steps('check <<2+2=4>>')[0]['actual'],'4')
    def test_language_layer(self):self.assertEqual(layer({'own_text_tokens':600,'language':'EN'}),'medium/en')

class IndexTests(unittest.TestCase):
    def fixture(self,r):
        import pyarrow as pa,pyarrow.parquet as pq
        rows=[]
        for i,s in enumerate(('train','validation','test')):
            text=' '.join(f'item{i}_{j} meaningful section content' for j in range(80))
            rows.append(dict(text=text,normalized_sha256=sha(text),split=s))
        p=r/'legacy.parquet';pq.write_table(pa.Table.from_pylist(rows),p)
        receipt=r/'legacy.json';receipt.write_text(json.dumps(dict(evidence_files=[dict(path=str(p),sha256=file_sha(p))])))
        recipe=json.loads((ROOT/'configs/curation/base_1b_supply_20b_v4.json').read_text());db=r/'supply.sqlite';ctl=SupplyControl(db,recipe);bid=ctl.register('code',{'fixture':1})
        with ctl.db:ctl.db.executemany('INSERT INTO documents VALUES (?,?,?,?,?)',[(v['normalized_sha256'],bid,v['split'],'code',100) for v in rows])
        ctl.close();return db,receipt,rows
    def test_same_threshold_retrieval_and_idempotence(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);db,receipt,old=self.fixture(r)
            v5=build_inventory_index(db,receipt,r/'v5cache');idx=DiskNearIndex(r/'v6')
            try:
                v6=idx.update(db,receipt,r/'v5cache');v6again=idx.update(db,receipt,r/'v5cache');self.assertEqual(v6,v6again)
                near=old[0]['text'].replace('item0_1 ','changed_1 ')
                unique='Completely unrelated arithmetic lesson with distinct tokens and examples. '*10
                rows=[dict(text=x,normalized_sha256=sha(x),split=s) for x,s in [(old[1]['text'],'train'),(near,'train'),(unique,'train'),(unique,'test')]]
                policy=json.loads((ROOT/'configs/curation/p0_release_gates_v5.json').read_text())['near']
                a,ar=global_near(rows,v5,policy,db);b,br=idx.filter(rows,v6,policy,db)
                self.assertEqual([x['normalized_sha256'] for x in a],[x['normalized_sha256'] for x in b]);self.assertEqual(len(b),1);self.assertEqual(b[0]['split'],'test')
                self.assertEqual(idx.db.execute('SELECT COUNT(*) FROM docs').fetchone()[0],3)
            finally:idx.close()
    def test_missing_coverage_refuses_publication(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);db,receipt,old=self.fixture(r);idx=DiskNearIndex(r/'v6')
            try:
                idx.update(db,receipt,r/'cache')
                with idx.db:idx.db.execute('DELETE FROM docs WHERE id=1')
                with self.assertRaises(ValueError):idx.update(db,receipt,r/'cache')
            finally:idx.close()
    def test_changed_reference_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);db,receipt,old=self.fixture(r);idx=DiskNearIndex(r/'v6')
            try:
                idx.update(db,receipt,r/'cache');(r/'legacy.parquet').write_bytes(b'changed')
                with self.assertRaises(ValueError):idx.update(db,receipt,r/'cache')
            finally:idx.close()
    def test_changed_ledger_refuses_stale_filter(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);db,receipt,old=self.fixture(r);idx=DiskNearIndex(r/'v6')
            try:
                m=idx.update(db,receipt,r/'cache')
                import sqlite3
                d=sqlite3.connect(db)
                with d:d.execute("UPDATE documents SET tokens=tokens+1 WHERE split='train'")
                d.close()
                with self.assertRaises(ValueError):idx.filter([],m,{},db)
            finally:idx.close()

if __name__=='__main__':unittest.main()

class LiteralContextTests(unittest.TestCase):
    def test_catalog_exact_unique_bounded_deterministic(self):
        from small_agent.curation.literal_context_v6 import catalog
        text='Repeated formula 2+2=4\n'*20+'Unicode α, exact whitespace  x =  7\n'+'Mathematics distinct explanation. '*10
        a=catalog(text);self.assertTrue(a);self.assertEqual(a,catalog(text));self.assertLessEqual(len(a),384)
        self.assertTrue(all(8<=len(q)<=100 and text.count(q)==1 for q in a))
    def test_adapter_does_not_change_body_schema_or_plain_protocol(self):
        from small_agent.curation.literal_context_v6 import adapt,extension
        text='Distinct unique arithmetic equation 2 + 2 = 4.';doc={'document_text':text,'review_payload':json.dumps({'document_text':text})}
        original={'input':[{'role':'user','content':doc['review_payload']}],'text':{'format':{'schema':{'fixture':1}}}}
        judge={'endpoint':{'kind':'responses'},'generation':{}}
        self.assertIs(adapt(original,doc,judge),original)
        judge['generation']['literal_context_adapter']=extension();body=adapt(original,doc,judge)
        self.assertEqual(body['text'],original['text']);p=json.loads(body['input'][0]['content']);self.assertEqual(p['document_text'],text);self.assertTrue(p['literal_evidence_catalog'])
        self.assertNotIn('literal_evidence_catalog',json.loads(original['input'][0]['content']))
    def test_changed_adapter_fails_closed(self):
        from small_agent.curation.literal_context_v6 import adapt,extension
        e=extension();e['adapter_sha256']='0'*64
        with self.assertRaises(ValueError):adapt({}, {}, {'endpoint':{'kind':'responses'},'generation':{'literal_context_adapter':e}})
