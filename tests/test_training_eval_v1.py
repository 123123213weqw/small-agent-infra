import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
import sys

P = Path(__file__).resolve().parents[1]/'scripts/training_eval_common_v1.py'
spec = importlib.util.spec_from_file_location('training_eval_common_v1', P)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
old_common=sys.modules.get('training_eval_common_v1')
sys.modules['training_eval_common_v1']=m
f_spec=importlib.util.spec_from_file_location('finalize_training_eval_test',P.with_name('finalize_training_eval_v1.py'))
f=importlib.util.module_from_spec(f_spec);f_spec.loader.exec_module(f)
if old_common is None: sys.modules.pop('training_eval_common_v1',None)
else: sys.modules['training_eval_common_v1']=old_common


class TestTrainingEval(unittest.TestCase):
    def test_fixed_subset(self):
        self.assertEqual(m.fixed_subset(range(10),3),[0,3,6])
        self.assertEqual(m.fixed_subset(range(2),3),[0,1])
        self.assertEqual(len(set(m.fixed_subset(range(100),32))),32)
        self.assertEqual(m.fixed_subset(range(100),None),list(range(100)))
    def test_token_weighted_not_batch_mean(self):
        x = m.weighted_loss({'a':{'loss':1.,'predicted_tokens':10},'b':{'loss':3.,'predicted_tokens':30}})
        self.assertEqual(x['loss'],2.5)
        self.assertEqual(x['predicted_tokens'],40)
    def test_no_targets(self):
        with self.assertRaises(ValueError): m.weighted_loss({})
    def test_wilson(self):
        self.assertAlmostEqual(m.wilson(0,16)[0],0.)
        self.assertGreater(m.wilson(0,16)[1],.19)
        self.assertAlmostEqual(m.wilson(16,16)[1],1.)
        with self.assertRaises(ValueError): m.wilson(2,1)
    def test_schedule(self):
        c={'fast_interval_steps':500,'ability_interval_steps':2000}
        self.assertEqual(m.due_checks(100,10000,c),['generation'])
        self.assertIn('validation_quick',m.due_checks(500,10000,c))
        self.assertIn('abilities_pilot',m.due_checks(2000,10000,c))
        self.assertIn('abilities_full',m.due_checks(10000,10000,c))
    def test_input_binding_rejects_changed_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';cp=root/'cp';tok=root/'tok'
            for d in [run,cp,tok]: d.mkdir()
            (tok/'tokenizer.json').write_text('{}')
            tm={'stage':'frozen_tokenizer','artifacts':{'tokenizer.json':{'sha256':m.sha(tok/'tokenizer.json')}}}
            m.write_json(tok/'manifest.json',tm)
            dm={'tokenizer_sha256':m.sha(tok/'tokenizer.json')}
            rm={};compat={}
            for key,content in [('run_config',{}),('model_spec',{}),('data_manifest',dm)]:
                path=root/(key+'.json');m.write_json(path,content)
                rm[key]={'path':str(path),'sha256':m.sha(path)}
                compat[key+'_sha256']=m.sha(path)
            m.write_json(run/'run_manifest.json',rm)
            (cp/'model.safetensors').write_bytes(b'abc')
            cm={'step':1,'compatibility':compat,'artifacts':{'model.safetensors':{'bytes':3,'sha256':m.sha(cp/'model.safetensors')}}}
            m.write_json(cp/'manifest.json',cm)
            self.assertEqual(m.verify_inputs(run,cp,tok)[1]['step'],1)
            (cp/'model.safetensors').write_bytes(b'xyz')
            with self.assertRaises(ValueError): m.verify_inputs(run,cp,tok)
    def test_atomic_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'child/file.json'
            m.write_json(dest,{'text':'中文','n':3})
            self.assertEqual(json.loads(dest.read_text()),{'text':'中文','n':3})
    def test_bad_schedule(self):
        with self.assertRaises(ValueError): m.due_checks(0,100,{'fast_interval_steps':500,'ability_interval_steps':2000})
    def test_category_aggregation(self):
        d={'general_en':{'loss':1.,'predicted_tokens':10,'samples':1},'wave2_general_en':{'loss':3.,'predicted_tokens':30,'samples':2},'math_en':{'loss':2.,'predicted_tokens':4,'samples':1}}
        categories=f.aggregate_domains(d)
        self.assertEqual(categories['en']['loss'],2.5)
        self.assertEqual(categories['en']['samples'],3)
        self.assertEqual(m.weighted_loss(categories),m.weighted_loss(d))
    def test_unknown_category_rejected(self):
        with self.assertRaises(ValueError): f.aggregate_domains({'new_domain':{'loss':1,'predicted_tokens':1,'samples':1}})


if __name__ == '__main__': unittest.main()
