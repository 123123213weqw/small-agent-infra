"""CPU full-stock routing, resumability and strict READY-only mixture accounting."""
import copy,json,sqlite3,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.full_stock_v9 import route,quota_status,priorities,ReadyQuotaControl,loss_corrected_sample_weights
from small_agent.curation.supply_control import file_sha
from small_agent.curation.release_gates_v5 import sha
from p0_stock_plan_v6 import open_plan
from p0_full_stock_scan_v9 import scan
from p0_full_stock_math_v9 import selection,allocate_one

FILTER=ROOT/'configs/curation/p0_full_stock_filter_v9.json'
RECIPE=json.loads((ROOT/'configs/curation/base_1b_supply_20b_v4.json').read_text())


def row(text='A mathematical explanation with an equation and a proof. '*20,**kw):
    d=dict(text=text,normalized_sha256=sha(text),source_id='finemath_4plus_supply_v4',source_revision='a'*40,
           source_file='math/part.parquet',source_locator='https://unlisted.example.org/lesson',domain='math',language='en',
           own_text_tokens=300,split='train',family_id=sha('family'+text),record_license='',policy_bucket='source_terms_hold')
    d.update(kw);return d


class RoutingTests(unittest.TestCase):
    def setUp(self):self.cfg=json.loads(FILTER.read_text())
    def test_unlisted_host_not_blanket_rejected_or_approved(self):
        self.assertEqual(route(row(),self.cfg),('PENDING_CALIBRATION','whole_body_rule_candidate_not_quality_approval','math_web'))
    def test_subdomain_spoof_not_trusted(self):
        r=row(source_locator='https://math.stackexchange.com.evil.org/article')
        self.assertEqual(route(r,self.cfg)[2],'math_web')
    def test_locked_preview_dropped_and_body_unchanged(self):
        r=row(text='Answer preview. Subscribe to unlock the answer. '*50);before=copy.deepcopy(r)
        self.assertEqual(route(r,self.cfg)[0],'DROP');self.assertEqual(r,before)
    def test_unknown_code_license_quarantined(self):
        self.assertEqual(route(row(domain='code',language='c',record_license='unknown'),self.cfg)[1],'record_license_unresolved')
    def test_nonpython_not_approved_by_python_reference(self):
        r=row(domain='code',language='c',record_license='mit',policy_bucket='record_license_whitelisted')
        self.assertEqual(route(r,self.cfg)[1],'independent_language_reference_required')
    def test_small_python_fragment_not_auto_dropped_or_executed(self):
        r=row(text='print(1)',own_text_tokens=18,domain='code',language='python',record_license='mit',policy_bucket='record_license_whitelisted')
        self.assertEqual(route(r,self.cfg)[0],'DELEGATED')
    def test_missing_split_never_manufactured(self):
        self.assertEqual(route(row(split='unassigned'),self.cfg)[1],'unassigned_split')
    def test_large_doc_quarantined_not_truncated(self):
        self.assertEqual(route(row(text='x'*100001),self.cfg)[1],'whole_body_teacher_context_limit')


class QuotaTests(unittest.TestCase):
    def test_sample_weights_compensate_padding_loss_token_length(self):
        w,m=loss_corrected_sample_weights({'code':.2,'math':.1},{'code':10000,'math':1000},{'code':10,'math':10})
        self.assertAlmostEqual(w['code']*m['code']/(w['math']*m['math']),2)
        self.assertAlmostEqual(w['math']/w['code'],5)
    def test_loss_weight_correction_refuses_missing_domain(self):
        with self.assertRaises(ValueError):loss_corrected_sample_weights({'code':.2,'math':.1},{'code':10000},{'code':10})
    def test_legacy_and_candidates_never_satisfy_ready_goal(self):
        s=quota_status(RECIPE,{}, {'general_en':7000000000},{'math':10064401415})
        self.assertEqual(s['remaining_ready_by_domain']['general_en'],7000000000)
        self.assertFalse(s['target_met']);self.assertEqual(s['recipe_admitted_unique_train_body_tokens'],0)
    def test_abundant_math_is_reserve_not_a_weight_increase(self):
        s=quota_status(RECIPE,{'math':6000000000},{})
        self.assertEqual(s['target_weights']['math'],.1);self.assertEqual(s['reserve_by_domain']['math'],4000000000)
        self.assertEqual(s['recipe_admitted_unique_train_body_tokens'],2000000000)
        self.assertNotIn('math',priorities(RECIPE,{'math':6000000000},set(RECIPE['targets_by_domain'])))
    def test_missing_domain_wait_not_silent_renormalization(self):
        s=quota_status(RECIPE,{'code':4000000000},{})
        self.assertEqual(s['target_weights']['general_en'],.35)
        self.assertEqual(s['missing_domain_action'],'WAIT_REPLENISHMENT_NEVER_SILENT_RENORMALIZE')
    def test_ready_control_keeps_legacy_but_does_not_spend_quota(self):
        with tempfile.TemporaryDirectory() as t:
            c=ReadyQuotaControl(Path(t)/'ledger.sqlite',RECIPE)
            try:
                with c.db:
                    c.db.execute("INSERT INTO batches(batch_id,domain,input_binding,status,created) VALUES ('old','legacy','{}','legacy_internal',0)")
                    c.db.execute("INSERT INTO legacy_imports VALUES ('old','old')")
                    c.db.execute('INSERT INTO documents VALUES (?,?,?,?,?)',('a'*64,'old','train','math',2000000000))
                self.assertEqual(c.status()['remaining_by_domain']['math'],2000000000)
                self.assertEqual(c.status()['legacy_internal_train_body_tokens'],2000000000)
            finally:c.close()


class ScanTests(unittest.TestCase):
    def fixture(self,r):
        import pyarrow as pa,pyarrow.parquet as pq
        orig=r/'orig';db=open_plan(orig);rows=[row(text='Equation and proof one. '*40),row(text='Equation and proof two. '*40,split='validation'),row(text='Equation and proof three. '*40,split='test')]
        p=r/'source.parquet';pq.write_table(pa.Table.from_pylist(rows),p)
        with db:db.execute('INSERT INTO sources VALUES (1,?,?,?,?,1)',(str(p),file_sha(p),'finemath_4plus_supply_v4','a'*40))
        db.close();(orig/'STOCK_REGISTERED.json').write_text(json.dumps(dict(sources=1,documents=3,candidate_train_body_tokens=300,binding={'fixture':True})))
        return orig,r/'out'
    def test_full_coverage_reentry_no_double_count(self):
        with tempfile.TemporaryDirectory() as t:
            orig,out=self.fixture(Path(t));a=scan(orig,out,FILTER);b=scan(orig,out,FILTER)
            self.assertEqual(a,b);self.assertEqual(a['documents'],3);self.assertEqual(a['candidate_train_body_tokens'],300)
            self.assertFalse(a['training_eligible'])
    def test_frozen_config_change_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);orig,out=self.fixture(r);scan(orig,out,FILTER)
            f=r/'changed.json';cfg=json.loads(FILTER.read_text());cfg['maximum_body_bytes']+=1;f.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):scan(orig,out,f)
    def test_deterministic_selection_and_one_batch_allocation(self):
        with tempfile.TemporaryDirectory() as t:
            orig,out=self.fixture(Path(t));scan(orig,out,FILTER);db=open_plan(out)
            try:
                a=selection(db,'math_web',set(),123);b=selection(db,'math_web',set(),123);self.assertEqual(a,b)
            finally:db.close()
            cfg=dict(stock=str(out),batch_train_body_tokens=300)
            scope=dict(status='passed',approved_strata=['short/en'],known_audit_bad_body_sha256=[])
            j=allocate_one(cfg,'math_web',scope,300);self.assertIsNotNone(j)
            self.assertEqual(allocate_one(cfg,'math_web',scope,300),j)
            self.assertIsNone(allocate_one(cfg,'math_web',scope,0))
    def test_scan_does_not_touch_original_metadata(self):
        with tempfile.TemporaryDirectory() as t:
            orig,out=self.fixture(Path(t));before=file_sha(orig/'stock.sqlite');scan(orig,out,FILTER)
            self.assertEqual(before,file_sha(orig/'stock.sqlite'));self.assertNotEqual((out/'stock.sqlite').resolve(),(orig/'stock.sqlite').resolve())
    def test_partial_transaction_cursor_resume_no_skip_or_repeat(self):
        import pyarrow as pa,pyarrow.parquet as pq
        import p0_full_stock_scan_v9 as scanner
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);orig,out=self.fixture(r);p=r/'source.parquet'
            rows=[row(text=f'Equation {i} unique explanation. '*20) for i in range(1000)]
            pq.write_table(pa.Table.from_pylist(rows),p);db=open_plan(orig)
            with db:db.execute('UPDATE sources SET sha=?',(file_sha(p),))
            db.close();(orig/'STOCK_REGISTERED.json').write_text(json.dumps(dict(sources=1,documents=1000,candidate_train_body_tokens=300000,binding={'fixture':True})))
            calls=0;real_route=scanner.route
            def interrupted(*args):
                nonlocal calls
                calls+=1
                if calls==520:raise RuntimeError('controlled crash after first committed batch')
                return real_route(*args)
            with patch.object(scanner,'route',side_effect=interrupted),self.assertRaises(RuntimeError):scan(orig,out,FILTER)
            db=open_plan(out)
            try:self.assertEqual(db.execute('SELECT next_pos FROM scan_progress').fetchone()[0],512)
            finally:db.close()
            done=scan(orig,out,FILTER);self.assertEqual(done['documents'],1000);self.assertEqual(done['candidate_train_body_tokens'],300000)

if __name__=='__main__':unittest.main()
