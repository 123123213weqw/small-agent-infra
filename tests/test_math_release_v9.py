"""Mock quality semantics, REAL physical pack/ledger; fixtures never enter real inventory."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import p0_release_math_batch_v9 as runner
from test_math_release_v6 import MathPipelineTests
from small_agent.curation.supply_control import file_sha,SupplyControl


class FullMathReleaseTests(unittest.TestCase):
    def fixture(self,r):
        argv,out,db,recipe,q,ba=MathPipelineTests().fixture(r)
        cfg=ROOT/'configs/curation/p0_full_stock_filter_v9.json';receipt=r/'scan.json'
        receipt.write_text(json.dumps(dict(identity=dict(config_sha256=file_sha(cfg),router_sha256=file_sha(ROOT/'src/small_agent/curation/full_stock_v9.py')))))
        source=Path(argv[argv.index('--source-audit')+1]);source.mkdir(parents=True,exist_ok=True)
        (source/'SAMPLE_PLAN.json').write_text(json.dumps(dict(request_identity=dict(bucket='math_web',full_stock_scan_sha256=file_sha(receipt)))))
        argv+=['--filter-receipt',str(receipt),'--filter-config',str(cfg),'--filter-bucket','math_web']
        return argv,out,db,recipe,q,ba
    def test_physical_release_and_recovery(self):
        with tempfile.TemporaryDirectory() as t:
            argv,out,db,recipe,q,ba=self.fixture(Path(t))
            with patch.object(sys,'argv',argv),patch.object(runner,'validate_source_cohort',return_value=None),patch.object(runner,'verify_batch_design'),patch.object(runner,'quality_scope',return_value=q),patch.object(runner,'rolling_audit',return_value=ba),patch.object(runner,'route',return_value=('PENDING_CALIBRATION','fixture','math_web')):runner.main()
            with patch.object(sys,'argv',argv),patch.object(runner,'validate_source_cohort',return_value=None),patch.object(runner,'verify_batch_design'):runner.main()
            c=SupplyControl(db,recipe)
            try:self.assertEqual(c.status()['unique_ready_train_body_tokens'],512)
            finally:c.close()
    def test_another_tier_cannot_borrow_quality_approval(self):
        with tempfile.TemporaryDirectory() as t:
            argv,out,*_=self.fixture(Path(t));argv[-1]='math_academic'
            with patch.object(sys,'argv',argv),self.assertRaises(ValueError):runner.main()
            self.assertFalse((out/'READY_INTERNAL.json').exists())
    def test_rule_failure_cannot_be_overridden_by_teacher(self):
        with tempfile.TemporaryDirectory() as t:
            argv,out,db,recipe,q,ba=self.fixture(Path(t))
            with patch.object(sys,'argv',argv),patch.object(runner,'validate_source_cohort',return_value=None),patch.object(runner,'verify_batch_design'),patch.object(runner,'quality_scope',return_value=q),patch.object(runner,'rolling_audit',return_value=ba),patch.object(runner,'route',return_value=('DROP','locked','math_web')),self.assertRaises(ValueError):runner.main()
            self.assertFalse((out/'READY_INTERNAL.json').exists())

if __name__=='__main__':unittest.main()
