import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from small_agent.curation.math_filter_v7 import structural_filter
C=json.loads((ROOT/'configs/curation/p0_math_structural_filter_v7.json').read_text())
class MathFilterTests(unittest.TestCase):
 def row(self):return dict(source_locator='https://math.stackexchange.com/questions/123/lesson',language='en',own_text_tokens=600,text='The equation defines a polynomial function and its derivative. Consider a matrix of coefficients and calculate the sum.\n\nThis is a distinct explanation of a useful theorem and algebraic proof.')
 def test_candidate_not_approval_and_no_mutation(self):
  r=self.row();old=r['text'];self.assertEqual(structural_filter(r,C),'structural_candidate_pass_not_quality_approval');self.assertEqual(old,r['text'])
 def test_host_suffix_spoof_not_admitted(self):
  r=self.row();r['source_locator']='https://math.stackexchange.com.evil.example/lesson';self.assertEqual(structural_filter(r,C),'host_outside_conservative_subset_hold')
 def test_search_not_mathematical_article(self):
  r=self.row();r['source_locator']='https://math.stackexchange.com/search?q=integral';self.assertEqual(structural_filter(r,C),'directory_search_page_hold')
 def test_locked_preview(self):
  r=self.row();r['text']+=' Unlock the full solution';self.assertEqual(structural_filter(r,C),'explicit_locked_answer_preview_hold')
 def test_replacement_character(self):
  r=self.row();r['text']+=' \ufffd';self.assertEqual(structural_filter(r,C),'replacement_or_null_character_hold')
 def test_repeated_paragraphs(self):
  r=self.row();r['text']=(r['text']+'\n\n')*8;self.assertEqual(structural_filter(r,C),'repeated_paragraphs_hold')
if __name__=='__main__':unittest.main()
