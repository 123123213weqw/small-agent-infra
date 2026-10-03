import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('census',SCRIPTS/'p0_count_supply_candidates_v4.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FakeTokenizer:
    def encode_batch(self,texts,add_special_tokens=False):
        assert not add_special_tokens
        self.lookup=getattr(self,'lookup',{})
        out=[]
        for t in texts:
            ids=[ord(c)+15 for c in t];self.lookup[tuple(ids)]=t
            out.append(type('Encoding',(),{'ids':ids})())
        return out
    def decode(self,ids,skip_special_tokens=False):
        assert not skip_special_tokens
        return self.lookup[tuple(ids)]

class CensusTests(unittest.TestCase):
    def db(self):
        db=sqlite3.connect(':memory:');m.initialize(db,{'id':1});return db
    def code(self,text='def f():\n    return 1\n',**kw):
        r={'code':text,'repo_name':'owner/repo','path':'x.py','license':'mit','language':'Python'};r.update(kw)
        return m.prepare('github_code_clean_supply_v4',r,1000000)
    def test_code_whitespace_exact_except_crlf(self):
        text='\r\n    x = "汉字  "  \r\n    y = 1\r\n'
        r,flags=self.code(text)
        self.assertEqual(flags,[]);self.assertEqual(r['text'],text.replace('\r\n','\n'))
    def test_missing_license_is_counted_but_held(self):
        r,flags=self.code(license='');self.assertFalse(flags);self.assertEqual(r['policy_bucket'],'record_license_hold')
    def test_gpl_not_whitelisted(self):
        r,_=self.code(license='gpl-3.0');self.assertEqual(r['policy_bucket'],'record_license_hold')
    def test_math_source_not_assumed_approved(self):
        r,f=m.prepare('finemath_4plus_supply_v4',{'text':'Example\n1+1=2','language':'en','int_score':4,'url':'https://example.org/a?q=2'},100)
        self.assertFalse(f);self.assertEqual(r['policy_bucket'],'source_terms_hold')
    def test_math_low_score_dropped(self):
        r,f=m.prepare('finemath_4plus_supply_v4',{'text':'x','language':'en','int_score':3},100)
        self.assertIsNone(r);self.assertIn('upstream_math_score_or_language',f)
    def test_vendor_exclusion(self):
        _,f=self.code(path='node_modules/pkg/x.py');self.assertIn('vendor_or_build_path',f)
    def test_syntax_failure_flag_not_base_rejection(self):
        r,f=self.code('def f(:\n    return 1\n');self.assertFalse(f);self.assertIn('parse_failed',r['python_parse_status'])
    def test_missing_family_unassigned(self):
        r,f=self.code(repo_name='');self.assertFalse(f)
        db=self.db();self.assertIsNone(m.reserve_body(db,r,'s',42));self.assertEqual(r['split'],'unassigned')
    def test_url_family_query_fragment_same(self):
        def record(url):return m.prepare('finemath_4plus_supply_v4',{'text':'x','int_score':4,'language':'en','url':url},100)[0]
        self.assertEqual(record('https://e.org/a?x=2')['family_id'],record('https://e.org/a#b')['family_id'])
    def test_legacy_family_preserves_heldout(self):
        db=self.db();r,_=self.code();db.execute('INSERT INTO legacy_families VALUES (?,?)',(bytes.fromhex(r['family_id']),'test'))
        self.assertIsNone(m.reserve_body(db,r,'s',42));self.assertEqual(r['split'],'test')
    def test_exact_legacy_alias_excluded(self):
        db=self.db();r,_=self.code('def f():  \r\n    return 1  \r\n')
        old=m.sha_text(m.normalize_text('def f():  \r\n    return 1  \r\n'))
        db.execute('INSERT INTO legacy_bodies VALUES (?)',(bytes.fromhex(old),))
        self.assertEqual(m.reserve_body(db,r,'s',42),'exact_overlap_legacy')
    def test_repeat_body_not_counted_twice(self):
        db=self.db();a,_=self.code();b,_=self.code(repo_name='other/repo')
        self.assertIsNone(m.reserve_body(db,a,'s',42));self.assertEqual(m.reserve_body(db,b,'t',42),'exact_duplicate_census')
    def test_identity_change_rejected(self):
        db=self.db()
        with self.assertRaises(ValueError):m.initialize(db,{'id':2})
    def test_duplicate_input_paths_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            i={'source_id':'finemath_4plus_supply_v4','revision':'a'*40,'sha256':'b'*64,'destination':d+'/x.parquet','path':'x.parquet'}
            with self.assertRaises(ValueError):m.validate_items({'allowed_root':d,'items':[i,i]})
    def test_shard_resume_output_and_counter_idempotence(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'raw.parquet';pout=root/'out';pout.mkdir()
            rows=[{'code':'def f():\n    return 1\n','repo_name':'a/b','path':'x.py','language':'Python','license':'mit'},
                  {'code':'def f():\n    return 1\n','repo_name':'c/d','path':'y.py','language':'Python','license':'mit'}]
            pq.write_table(pa.Table.from_pylist(rows),p)
            item={'source_id':'github_code_clean_supply_v4','revision':'a'*40,'sha256':m.sha256_file(p),
                  'destination':str(p),'path':'data/x.parquet','bytes':p.stat().st_size}
            cfg={'batch_size':1,'split_seed':42,'max_chars':10000,'full_roundtrip':True,'reserved_tokens':['<eos>']}
            db=self.db();tok=FakeTokenizer();progress=lambda *a:None
            m.process_shard(db,item,tok,pout,cfg,'i',progress)
            before=m.collect_totals(db);m.process_shard(db,item,tok,pout,cfg,'i',progress)
            self.assertEqual(m.collect_totals(db),before)
            self.assertEqual(before['counts']['candidate_documents'],1)
            self.assertEqual(before['counts']['candidate_body_tokens'],len(rows[0]['code']))
            self.assertEqual(before['counts']['exact_duplicate_census'],1)
            out=pout/before['files'][0]['output_relative_path']
            self.assertFalse(pq.read_table(out)['training_approved'][0].as_py())
            with out.open('ab') as f:f.write(b'broken')
            with self.assertRaises(ValueError):m.process_shard(db,item,tok,pout,cfg,'i',progress)
    def test_rollback_after_roundtrip_failure(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        class BrokenTokenizer(FakeTokenizer):
            def decode(self,*a,**kw):return 'bad'
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'raw.parquet';pout=root/'out';pout.mkdir()
            pq.write_table(pa.Table.from_pylist([{'code':'def f():\n    return 1\n','repo_name':'a/b','path':'x.py','language':'Python','license':'mit'}]),p)
            item={'source_id':'github_code_clean_supply_v4','revision':'a'*40,'sha256':m.sha256_file(p),
                  'destination':str(p),'path':'x.parquet','bytes':p.stat().st_size}
            cfg={'batch_size':1,'split_seed':42,'max_chars':10000,'full_roundtrip':True,'reserved_tokens':['<eos>']}
            db=self.db()
            with self.assertRaises(ValueError):m.process_shard(db,item,BrokenTokenizer(),pout,cfg,'i',lambda *a:None)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM documents').fetchone()[0],0)
            m.process_shard(db,item,FakeTokenizer(),pout,cfg,'i',lambda *a:None)
            self.assertEqual(m.collect_totals(db)['counts']['candidate_documents'],1)

if __name__=='__main__':unittest.main()
