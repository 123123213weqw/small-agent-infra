import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from small_agent.curation.core import canonical,normalize_document,sha,decide
from small_agent.curation.store import Batch,prepare,protocol,read_json,validate_config,check_release,file_sha
import p0_source_audit_v4 as adapter
import p0_curation_worker_v3 as worker
POLICY=read_json(ROOT/'configs/curation/p0_source_audit_policy_v4.json')
SCHEMA=read_json(ROOT/'schemas/p0_teacher_quality_v3.json')
PROFILE={'model':'deepseek-flash','model_display_name':'fixture-model','model_metadata_sha256':'a'*64,
    'max_output_tokens':2048,'maximum_input_tokens':131072,'api_base':'https://api.deepseek.com',
    'pricing_cny_per_million':{'uncached_input':2,'output':8,'margin':1.15}}
REGISTRY=adapter.registry_for(PROFILE)

def doc():
    text='This is a unique introduction to graph search. A queue holds the next nodes.'
    return normalize_document({'document_text':text,'source_id':'fixture','source_locator':'https://example.org/a',
        'token_count':20,'language':'en'})
def label(d):
    return {'document_id':d['document_id'],'body_sha256':d['body_sha256'],'coverage':'full_document',
        'decision':'keep','confidence':.95,'quality':4,'completeness':2,'educational_value':1,'format_integrity':4,
        'recommended_bucket':'technical','anchors':[d['document_text'][:50]],'findings':[]}

class AuditTests(unittest.TestCase):
    def test_single_reviewer_only_with_explicit_audit_policy(self):
        validate_config(POLICY,REGISTRY)
        legacy=read_json(ROOT/'configs/curation/p0_policy_v3.json')
        with self.assertRaises(ValueError):validate_config(legacy,REGISTRY)
    def test_audit_cannot_enable_corpus_release(self):
        p=copy.deepcopy(POLICY);p['corpus_release_allowed']=True
        with self.assertRaises(ValueError):validate_config(p,REGISTRY)
    def test_keep_is_a_signal_not_eligible(self):
        d=doc();r=decide(d,{'deepseek':{'status':'valid','annotation':label(d)}},REGISTRY['judges'],POLICY)
        self.assertEqual(r['status'],'audit_keep_signal');self.assertFalse(r['source_approved']);self.assertFalse(r['training_approved'])
    def test_invalid_not_treated_as_drop_or_keep(self):
        d=doc();r=decide(d,{'deepseek':{'status':'invalid'}},REGISTRY['judges'],POLICY)
        self.assertEqual(r['status'],'audit_invalid')
    def test_audit_cache_shared_between_rounds(self):
        a=SimpleNamespace(policy=POLICY,manifest_hash='one');b=SimpleNamespace(policy=POLICY,manifest_hash='two')
        self.assertEqual(worker.job_identity(a,'deepseek',{'body':'same'}),worker.job_identity(b,'deepseek',{'body':'same'}))
    def test_legacy_cache_remains_per_batch(self):
        a=SimpleNamespace(policy={},manifest_hash='one');b=SimpleNamespace(policy={},manifest_hash='two')
        self.assertNotEqual(worker.job_identity(a,'deepseek',{}),worker.job_identity(b,'deepseek',{}))
    def test_changed_prompt_or_protocol_does_not_share_cache(self):
        a=SimpleNamespace(policy=POLICY,manifest_hash='one')
        self.assertNotEqual(worker.job_identity(a,'deepseek',{'prompt':'one'}),worker.job_identity(a,'deepseek',{'prompt':'two'}))
    def test_sampling_deterministic_under_shuffle(self):
        rows=[{'normalized_sha256':sha(str(i))} for i in range(100)]
        def pick(xs):
            heap=[]
            for r in xs:adapter.offer(heap,sha('seed'+r['normalized_sha256']),r,8)
            return sorted(x[1] for x in heap)
        expected=pick(rows);random.Random(12).shuffle(rows);self.assertEqual(expected,pick(rows))
    def test_source_audit_export_and_release_forbidden(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);source=r/'input.jsonl';source.write_text(canonical({k:v for k,v in doc().items() if k not in ['review_payload','lint_flags']})+'\n')
            for name,obj in [('policy.json',POLICY),('schema.json',SCHEMA),('registry.json',REGISTRY)]:
                (r/name).write_text(json.dumps(obj))
            prepare(source,r/'batch',r/'policy.json',r/'schema.json',ROOT/'configs/curation/p0_source_audit_prompt_v4.txt',r/'registry.json')
            batch=Batch(r/'batch')
            try:
                self.assertEqual(batch.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],1)
                with self.assertRaises(ValueError):batch.export(r/'export')
                with self.assertRaises(ValueError):check_release(batch,r/'missing',r/'missing.json')
            finally:batch.close()
    def test_shared_paid_worker_receipt_recovers_without_second_request(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);d=doc();source=r/'input.jsonl'
            source.write_text(canonical({k:v for k,v in d.items() if k not in ['review_payload','lint_flags']})+'\n')
            for name,obj in [('policy.json',POLICY),('schema.json',SCHEMA),('registry.json',REGISTRY)]:
                (r/name).write_text(json.dumps(obj))
            for name in ['batch1','batch2']:
                prepare(source,r/name,r/'policy.json',r/'schema.json',ROOT/'configs/curation/p0_source_audit_prompt_v4.txt',r/'registry.json')
            response={'status':'completed','model':'deepseek-flash','usage':{'input_tokens':100,'output_tokens':20},
                'output':[{'type':'message','content':[{'type':'output_text','text':canonical(label(d))}]}]}
            key='fixture-not-a-real-key'
            with patch.dict(os.environ,{'DEEPSEEK_API_KEY':key}),patch.object(worker,'send',return_value=response) as send:
                for name in ['batch1','batch2']:
                    argv=['worker','--batch',str(r/name),'--judge','deepseek','--allow-api','--ledger',str(r/'campaign.sqlite'),
                        '--budget-cny','1','--expected-key-sha256',sha(key),'--workers','1','--limit','1']
                    with patch.object(sys,'argv',argv):self.assertEqual(worker.main(),0)
                    b=Batch(r/name)
                    try:self.assertEqual(b.report()['status_documents'],{'audit_keep_signal':1})
                    finally:b.close()
                self.assertEqual(send.call_count,1)
    def fixture(self,root,documents=12):
        import pyarrow as pa
        import pyarrow.parquet as pq
        census=root/'census';norm=census/'normalized'/'s';norm.mkdir(parents=True)
        rows=[]
        for i in range(documents):
            text=f'def unique_{i}():\n    return {i}\n'
            rows.append({'document_id':sha('candidate'+str(i)),'source_id':'github_code_clean_supply_v4',
                'source_revision':'b'*40,'source_locator':f'repo/a/x{i}.py','normalized_sha256':sha(text),
                'text':text,'own_text_tokens':50,'domain':'code','language':'Python','family_id':sha('repo/a'),
                'split':'train','policy_bucket':'record_license_whitelisted'})
        rows.append({**rows[0],'document_id':sha('held'),'split':'validation'})
        rows.append({**rows[0],'document_id':sha('license'),'policy_bucket':'record_license_hold'})
        file=norm/'part.parquet';pq.write_table(pa.Table.from_pylist(rows),file)
        receipt={'key':'one','source_id':'github_code_clean_supply_v4','source_file':'data/x.parquet','source_sha256':'c'*64,
            'output_relative_path':'normalized/s/part.parquet','output_sha256':file_sha(file)}
        (norm/'receipt.json').write_text(json.dumps(receipt))
        db=sqlite3.connect(census/'census.sqlite');db.execute('CREATE TABLE shards(key TEXT PRIMARY KEY,receipt TEXT)')
        db.execute('INSERT INTO shards VALUES (?,?)',('one',json.dumps(receipt)));db.commit();db.close()
        plan=root/'plan.json';plan.write_text(json.dumps({'items':[{'source_id':receipt['source_id'],'path':'data/x.parquet','sha256':'c'*64}]}))
        sampling=read_json(ROOT/'configs/curation/p0_source_audit_sampling_v4.json');sampling['calibration_per_stratum']=2
        s=root/'sampling.json';s.write_text(json.dumps(sampling))
        return SimpleNamespace(profile=root/'private.json',census_root=census,download_plan=plan,
            sampling=s,policy=ROOT/'configs/curation/p0_source_audit_policy_v4.json',
            prompt=ROOT/'configs/curation/p0_source_audit_prompt_v4.txt',schema=ROOT/'schemas/p0_teacher_quality_v3.json',
            source=receipt['source_id'],mode='calibration',round_id='one',output=root/'audit',allow_partial_source=False)
    def test_end_to_end_offline_prepare_report_resume_blinded_payload(self):
        with tempfile.TemporaryDirectory() as t:
            args=self.fixture(Path(t))
            with patch.object(adapter,'load_profile',return_value=(PROFILE,'fixture-only',ROOT/'scripts/p0_curation_worker_v3.py')):
                r=adapter.prepare_audit(args);self.assertEqual(r['admitted_documents'],2)
                r2=adapter.prepare_audit(args);self.assertEqual(r2['stage'],'existing_source_audit_reused')
            b=Batch(args.output/'batch')
            try:
                self.assertEqual(b.report()['job_statuses'],{'queued':2})
                for d in b.docs.values():
                    payload=json.loads(d['review_payload'])
                    self.assertNotIn('upstream_score',payload);self.assertNotIn('previous_bucket',payload)
            finally:b.close()
            report=adapter.audit_report(args.output)
            self.assertEqual(report['population_documents'],12);self.assertFalse(report['source_approved'])
            self.assertEqual(report['document_weighted_model_signals']['audit_await_review'],1)
    def test_partial_source_requires_explicit_scope(self):
        with tempfile.TemporaryDirectory() as t:
            args=self.fixture(Path(t));plan=json.loads(args.download_plan.read_text())
            plan['items'].append({**plan['items'][0],'path':'data/extra.parquet'})
            with self.assertRaises(ValueError):adapter.snapshot_files(args.census_root,plan,args.source,False)
            files,n=adapter.snapshot_files(args.census_root,plan,args.source,True);self.assertEqual((len(files),n),(1,2))
    def test_oversize_population_not_silently_sampled_or_truncated(self):
        cfg=read_json(ROOT/'configs/curation/p0_source_audit_sampling_v4.json')
        row={'text':'x'*(cfg['maximum_sample_body_bytes']+1),'own_text_tokens':9000,'language':'Python','domain':'code'}
        self.assertTrue(adapter.stratum(row,cfg).startswith('over_budget_body_size/'))
    def test_review_mode_change_requires_new_run(self):
        with tempfile.TemporaryDirectory() as t:
            args=self.fixture(Path(t))
            with patch.object(adapter,'load_profile',return_value=(PROFILE,'fixture-only',ROOT/'scripts/p0_curation_worker_v3.py')):
                adapter.prepare_audit(args);args.mode='rolling'
                with self.assertRaises(ValueError):adapter.prepare_audit(args)
if __name__=='__main__':unittest.main()
