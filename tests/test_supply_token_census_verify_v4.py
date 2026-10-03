import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(SCRIPTS))
import p0_count_supply_candidates_v4 as census
import p0_verify_supply_token_census_v4 as check

class VerifyTests(unittest.TestCase):
    def fixture(self,root):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tokenizers import Tokenizer,models,pre_tokenizers,decoders,trainers
        text='def f():\n    return "中文"\n'
        tok=Tokenizer(models.BPE());tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder=decoders.ByteLevel()
        special=['<reserved_'+str(i)+'>' for i in range(15)]
        tok.train_from_iterator([text],trainers.BpeTrainer(vocab_size=512,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),special_tokens=special,show_progress=False))
        tokpath=root/'tokenizer.json';tok.save(str(tokpath));ts=census.sha256_file(tokpath)
        raw=root/'raw.parquet'
        pq.write_table(pa.Table.from_pylist([{'code':text,'repo_name':'a/b','path':'x.py','language':'Python','license':'mit'},
            {'code':text,'repo_name':'c/d','path':'y.py','language':'Python','license':'mit'}]),raw)
        out=root/'out';out.mkdir();db=sqlite3.connect(out/'census.sqlite');census.initialize(db,{'test':1})
        item={'source_id':'github_code_clean_supply_v4','path':'data/x.parquet','destination':str(raw),
            'revision':'a'*40,'sha256':census.sha256_file(raw),'bytes':raw.stat().st_size}
        cfg={'batch_size':1,'split_seed':42,'max_chars':10000,'full_roundtrip':True,'reserved_tokens':['<eos>']}
        census.process_shard(db,item,tok,out,cfg,'i',lambda *a:None)
        report={'stage':'candidate_token_census_complete_not_ready','complete':True,'training_approved':False,
            'new_ready_train_tokens':0,'planned_shards':1,'identity':{'tokenizer_sha256':ts},'identity_sha256':'i',
            'pending':['test_pending'],**census.collect_totals(db)}
        db.close();(out/'report.json').write_text(json.dumps(report));return out,tokpath,ts
    def test_full_body_hashes_counters_and_independent_token_sample(self):
        with tempfile.TemporaryDirectory() as d:
            out,tok,sha=self.fixture(Path(d))
            with patch.object(check,'FROZEN_SHA',sha):r=check.verify(out,tok)
            self.assertEqual(r['verified_shards'],1);self.assertEqual(r['reencoded_documents'],1)
            self.assertEqual(r['counts']['candidate_documents'],1);self.assertFalse(r['training_approved'])
    def test_changed_aggregate_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out,tok,sha=self.fixture(Path(d));p=out/'report.json';r=json.loads(p.read_text())
            r['counts']['candidate_body_tokens']+=1;p.write_text(json.dumps(r))
            with patch.object(check,'FROZEN_SHA',sha),self.assertRaises(ValueError):check.verify(out,tok)
    def test_missing_output_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out,tok,sha=self.fixture(Path(d));r=json.loads((out/'report.json').read_text())
            (out/r['files'][0]['output_relative_path']).unlink()
            with patch.object(check,'FROZEN_SHA',sha),self.assertRaises(FileNotFoundError):check.verify(out,tok)
    def test_changed_unique_index_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out,tok,sha=self.fixture(Path(d));db=sqlite3.connect(out/'census.sqlite')
            db.execute('UPDATE documents SET tokens=tokens+1');db.commit();db.close()
            with patch.object(check,'FROZEN_SHA',sha),self.assertRaises(ValueError):check.verify(out,tok)

if __name__=='__main__':unittest.main()
