import importlib.util
from pathlib import Path
import random
import unittest
p=Path(__file__).resolve().parents[1]/'scripts/p0_supply_seed_candidates_v4.py'
spec=importlib.util.spec_from_file_location('seed_candidates',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SeedTests(unittest.TestCase):
    def test_hash_selection_does_not_depend_on_input_order(self):
        entries=[(m.digest(str(i)),{'document_id':str(i)}) for i in range(100)]
        def select(xs):
            h=[]
            for r,d in xs:m.offer(h,r,d,10)
            return sorted(x[2]['document_id'] for x in h)
        expected=select(entries);random.Random(42).shuffle(entries);self.assertEqual(expected,select(entries))
    def test_same_rank_tie_is_deterministic(self):
        a=[];b=[]
        for i in ['a','b']:m.offer(a,'0'*64,{'document_id':i},1)
        for i in ['b','a']:m.offer(b,'0'*64,{'document_id':i},1)
        self.assertEqual(a,b)
    def test_code_keeps_indentation(self):
        self.assertEqual(m.normalize('def a():\r\n    return 1\r\n'),'def a():\n    return 1')
    def test_bad_python_not_accepted_as_code(self):
        text=('def bad(:\n    return 1\n'*15)
        self.assertIn('python_parse_failure',m.reasons('ultradata_code_l2_py',{},text))
    def test_code_license_gate_not_guessed(self):
        text='\n'.join('print('+str(i)+')' for i in range(30))
        flags=m.reasons('github_code_clean_new',{'language':'Python','license':'unknown','path':'a.py'},text)
        self.assertIn('license_not_whitelisted',flags)
    def test_math_score_is_source_specific(self):
        self.assertIn('upstream_math_score_or_language',m.reasons('finemath_4plus_new',{'language':'en','int_score':3},'a'*600))

if __name__=='__main__':unittest.main()
