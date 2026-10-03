"""Conservative wire adapter: never upgrade bad decisions; overlapping uniqueness."""
import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from test_evidence_ids_v7 import EvidenceIDTests
from small_agent.curation.evidence_ids_v8 import spans,resolve,extension,wire_schema,request_body,validate_receipt
from small_agent.curation.core import normalize_document,fingerprint
S=json.loads((ROOT/'schemas/p0_teacher_quality_v3.json').read_text());S['properties']['findings']['items']['properties']['quotes']['maxItems']=8
class ConservativeAdapterTests(unittest.TestCase):
 def fixture(self):
  d,j,a=EvidenceIDTests().fixture();j['generation']['evidence_id_transport']=extension();a['anchors']=[spans(d)[0]['id']];return d,j,a
 def test_overlapping_quote_never_admitted(self):
  d=normalize_document(dict(document_text='+1 vote\n'*30+'A distinct mathematical explanation at the end.',source_id='fixture',source_locator='fixture:one',language='en',content_kind='text'))
  for x in spans(d):self.assertEqual(d['document_text'].find(x['text'],x['start']+1),-1)
 def test_conflicting_keep_is_hold_not_success(self):
  d,j,a=self.fixture();a['findings']=[dict(code='factual_or_logical_error',severity='reject',quotes=a['anchors'],explanation='Visible contradictory calculation.')]
  n=resolve(a,d,S,j);self.assertEqual(a['decision'],'keep');self.assertEqual(n['decision'],'uncertain');self.assertEqual(n['quality'],a['quality']);self.assertEqual(n['findings'][0]['severity'],'reject')
 def test_drop_or_uncertain_can_never_be_upgraded(self):
  d,j,a=self.fixture();a['findings']=[dict(code='uncertain',severity='review',quotes=a['anchors'],explanation='Fixture cannot establish usability.')]
  for decision in ['drop','uncertain']:
   a['decision']=decision;self.assertEqual(resolve(a,d,S,j)['decision'],decision)
 def test_eight_real_evidence_ids_not_truncated(self):
  d,j,a=self.fixture();ids=[x['id'] for x in spans(d)[:8]];a['decision']='drop';a['findings']=[dict(code='factual_or_logical_error',severity='reject',quotes=ids,explanation='Fixture multiple contradictory statements.')]
  n=resolve(a,d,S,j);self.assertEqual(len(n['findings'][0]['quotes']),len(ids))
 def test_raw_and_actual_wire_receipt_still_bound(self):
  d,j,a=self.fixture();n=resolve(a,d,S,j);b=request_body(d,'prompt',S,j);r=dict(actual_request_sha256=fingerprint(b),wire_annotation=a,annotation=n,raw_annotation=n)
  self.assertEqual(validate_receipt(r,d,'prompt',S,j),n);r['wire_annotation']['quality']=1
  with self.assertRaises(ValueError):validate_receipt(r,d,'prompt',S,j)
if __name__=='__main__':unittest.main()
