import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p0_code_production_pilot_v4 import near_prune,pick_quota,publish,body_sha
from p0_verify_code_production_pilot_v4 import verify
from small_agent.curation.supply_control import file_sha
CFG=json.loads((ROOT/'configs/curation/p0_code_production_pilot_v4.json').read_text())


class PilotTests(unittest.TestCase):
    def row(self,text,split='train',rank='a'):
        return {'text':text,'split':split,'rank':rank,'own_text_tokens':100,'normalized_sha256':body_sha(text)}

    def test_exact_near_representative_keeps_heldout(self):
        text='def calculate(x): return x * x + 42\n'
        rows,excluded,counts=near_prune([self.row(text),self.row(text,'validation','b')],CFG)
        self.assertEqual([r['split'] for r in rows],['validation']);self.assertEqual(len(excluded),1)
    def test_case_sensitive_code_not_silently_equal(self):
        rows,_,_=near_prune([self.row('A A A A A'),self.row('a a a a a',rank='b')],CFG)
        self.assertEqual(len(rows),2)
    def test_dense_retrieval_is_explicit_hold(self):
        cfg=copy.deepcopy(CFG);cfg['maximum_near_candidates']=1
        text='def f(x): return x * x + 4'
        _,excluded,counts=near_prune([self.row(text,rank=str(i)) for i in range(3)],cfg)
        self.assertEqual(counts['dense_retrieval_hold'],1);self.assertEqual(len(excluded),2)
    def test_quota_uses_hash_rank_not_input_order(self):
        rows=[self.row('x',rank='b'),self.row('y',rank='a')]
        selected,n=pick_quota(rows,{'train':50});self.assertEqual(selected[0]['rank'],'a');self.assertEqual(n['train'],100)
    def test_immutable_output_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.json';publish(p,{'a':1});publish(p,{'a':1})
            with self.assertRaises(ValueError):publish(p,{'a':2})

    def fixture(self,root):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tokenizers import Tokenizer,models,pre_tokenizers
        from small_agent.data.sharded_pack import TokenizedDocument,pack_sharded
        tok=Tokenizer(models.WordLevel({'<pad>':0,'<unk>':1,'<eos>':2,'a':3,'b':4,'c':5},unk_token='<unk>'))
        tok.pre_tokenizer=pre_tokenizers.Whitespace();path=root/'tokenizer.json';tok.save(str(path))
        rows=[];index=[];body={};docs=[]
        for s,word,n in [('train','a',512),('validation','b',150),('test','c',150)]:
            text=' '.join([word]*n);sha=body_sha(text)
            rows.append({'text':text,'split':s,'family_id':s,'normalized_sha256':sha,'own_text_tokens':n})
            index.append({'body_sha256':sha,'split':s,'family_id':s,'domain':'code','body_tokens':n})
            body[s]=n;docs.append(TokenizedDocument(sha,s,'code',tok.encode(text).ids))
        pq.write_table(pa.Table.from_pylist(rows),root/'deduplicated.parquet')
        (root/'documents.jsonl').write_text('\n'.join(json.dumps(r) for r in index)+'\n')
        (root/'FROZEN.json').write_text(json.dumps({'parquet_sha256':file_sha(root/'deduplicated.parquet'),'index_sha256':file_sha(root/'documents.jsonl'),'tokenizer_sha256':file_sha(path),'training_eligible':False,'documents':3,'body_tokens':body}))
        pack_sharded(docs,root/'tokenized',sequence_length=128,target_shard_tokens=1024,vocab_size=6,eos_token_id=2,tokenizer_sha256=file_sha(path),mixture={'code':1},license_gate={'training_eligible':False},source_manifest_sha256=file_sha(root/'FROZEN.json'))
        db=root/'ledger.sqlite'
        with sqlite3.connect(db) as c:c.executescript('CREATE TABLE documents(body_sha TEXT); CREATE TABLE legacy_families(family_id TEXT,split TEXT);')
        return path,db,index

    def test_real_pack_full_roundtrip_loader_and_commit_restore(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);tok,db,_=self.fixture(root);r=verify(root,tok,db)
            self.assertTrue(r['all_document_decode_reencode_and_eos_verified']);self.assertTrue(r['committed_two_rank_sampler_restore_verified'])
            self.assertFalse(r['training_approved']);self.assertFalse(r['global_near_approved'])
    def test_physical_bin_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);tok,db,_=self.fixture(root);p=next((root/'tokenized').glob('train*.bin'))
            b=bytearray(p.read_bytes());b[0]^=1;p.write_bytes(b)
            with self.assertRaises(ValueError):verify(root,tok,db)
    def test_released_global_exact_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);tok,db,index=self.fixture(root)
            with sqlite3.connect(db) as c:c.execute('INSERT INTO documents VALUES (?)',(index[0]['body_sha256'],))
            with self.assertRaises(ValueError):verify(root,tok,db)
    def test_legacy_family_split_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);tok,db,index=self.fixture(root)
            with sqlite3.connect(db) as c:c.execute('INSERT INTO legacy_families VALUES (?,?)',('train','validation'))
            with self.assertRaises(ValueError):verify(root,tok,db)

if __name__=='__main__':unittest.main()
