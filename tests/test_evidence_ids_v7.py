"""Lossless wire/protocol tests; no real corpus quality approval."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.evidence_ids_v7 import *
from small_agent.curation.core import normalize_document
from small_agent.curation.store_v7 import protocol
S=json.loads((ROOT/'schemas/p0_teacher_quality_v3.json').read_text())
class EvidenceIDTests(unittest.TestCase):
 def fixture(self):
  d=normalize_document(dict(document_text='A unique formula with NBSP: x\u00a0+\u00a0y = 7.\n\\frac{1}{2} is preserved exactly.\n',source_id='fixture',source_locator='fixture:one',language='en',content_kind='text'))
  j=dict(model='fixture',endpoint=dict(kind='responses'),generation=dict(temperature=0,max_tokens=2048,chat_template_kwargs={},response_format='json_schema',thinking='disabled',require_document_id_match=True,evidence_id_transport=extension()))
  a=dict(document_id=d['document_id'],body_sha256=d['body_sha256'],coverage='full_document',decision='keep',confidence=.99,recommended_bucket='math_en',anchors=[spans(d)[0]['id']],findings=[],quality=4,completeness=4,educational_value=4,format_integrity=4)
  return d,j,a
 def test_lossless_unicode_latex_spans(self):
  d,j,a=self.fixture();n=resolve(a,d,S,j);self.assertEqual(n['anchors'][0],spans(d)[0]['text']);self.assertEqual(n['decision'],a['decision']);self.assertIn('\u00a0',n['anchors'][0])
 def test_unknown_id_invalid_not_filled(self):
  d,j,a=self.fixture();a['anchors']=['e999']
  with self.assertRaises(Exception):resolve(a,d,S,j)
 def test_wrong_document_id_invalid(self):
  d,j,a=self.fixture();a['document_id']='a'*64
  with self.assertRaises(Exception):resolve(a,d,S,j)
 def test_keep_with_adverse_invalid(self):
  d,j,a=self.fixture();a['findings']=[dict(code='format_broken',severity='reject',quotes=a['anchors'],explanation='A contradictory test finding.')]
  with self.assertRaises(ValueError):resolve(a,d,S,j)
 def test_missing_score_invalid(self):
  d,j,a=self.fixture();a.pop('quality')
  with self.assertRaises(Exception):resolve(a,d,S,j)
 def test_truthful_wire_schema_hash_and_receipt_binding(self):
  d,j,a=self.fixture();b=request_body(d,'prompt',S,j);p=protocol(d,'prompt',S,j)
  self.assertEqual(p['wire_schema_sha256'],fingerprint(b['text']['format']['schema']))
  n=resolve(a,d,S,j);r=dict(actual_request_sha256=fingerprint(b),wire_annotation=a,annotation=n,raw_annotation=n);self.assertEqual(validate_receipt(r,d,'prompt',S,j),n)
  r['annotation']['decision']='drop'
  with self.assertRaises(Exception):validate_receipt(r,d,'prompt',S,j)
 def test_changed_adapter_invalid(self):
  d,j,a=self.fixture();j['generation']['evidence_id_transport']['adapter_sha256']='a'*64
  with self.assertRaises(ValueError):request_body(d,'prompt',S,j)
if __name__=='__main__':unittest.main()
