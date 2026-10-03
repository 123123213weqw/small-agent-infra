"""Mock semantics; REAL pack and ledger; does not approve natural math corpus."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import p0_release_math_batch_v8 as runner
from test_math_release_v6 import MathPipelineTests
from small_agent.curation.supply_control import file_sha,SupplyControl
class CleanMathReleaseTests(unittest.TestCase):
 def fixture(self,r):
  argv,out,db,recipe,q,ba=MathPipelineTests().fixture(r);config=ROOT/'configs/curation/p0_math_structural_filter_v7.json'
  receipt=r/'filter.json';receipt.write_text(json.dumps(dict(identity=dict(config_sha256=file_sha(config),filter_sha256=file_sha(ROOT/'src/small_agent/curation/math_filter_v7.py')))))
  argv+=['--filter-receipt',str(receipt),'--filter-config',str(config)];return argv,out,db,recipe,q,ba
 def test_real_physical_pack_and_idempotent_release(self):
  with tempfile.TemporaryDirectory() as t:
   argv,out,db,recipe,q,ba=self.fixture(Path(t))
   with patch.object(sys,'argv',argv),patch.object(runner,'validate_source_cohort',return_value=None),patch.object(runner,'quality_scope',return_value=q),patch.object(runner,'rolling_audit',return_value=ba),patch.object(runner,'structural_filter',return_value='structural_candidate_pass_not_quality_approval'):runner.main()
   with patch.object(sys,'argv',argv),patch.object(runner,'validate_source_cohort',return_value=None):runner.main()
   c=SupplyControl(db,recipe)
   try:self.assertEqual(c.status()['unique_ready_train_body_tokens'],512);self.assertEqual(c.db.execute('SELECT COUNT(*) FROM releases').fetchone()[0],1)
   finally:c.close()
 def test_structural_hold_cannot_be_bypassed_by_good_teacher_score(self):
  with tempfile.TemporaryDirectory() as t:
   argv,out,db,recipe,q,ba=self.fixture(Path(t))
   with patch.object(sys,'argv',argv),patch.object(runner,'validate_source_cohort',return_value=None),patch.object(runner,'quality_scope',return_value=q),patch.object(runner,'rolling_audit',return_value=ba),patch.object(runner,'structural_filter',return_value='explicit_locked_answer_preview_hold'),self.assertRaises(ValueError):runner.main()
   self.assertFalse((out/'tokenized').exists());self.assertFalse((out/'READY_INTERNAL.json').exists())
if __name__=='__main__':unittest.main()
