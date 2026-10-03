from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).parents[1]
SCRIPT=ROOT/'scripts/benchmark_perf4_v1.py'
CONFIG=ROOT/'configs/benchmarks/perf4_v1.json'
if not SCRIPT.exists():
    SCRIPT,CONFIG=ROOT/'benchmark_perf4_v1.py',ROOT/'config.json'
spec=importlib.util.spec_from_file_location('perf4',SCRIPT)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Perf4Test(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(CONFIG.read_text())

    def test_fixed_config(self):
        m.validate_config(self.cfg)

    def test_reject_global_batch_change(self):
        c=copy.deepcopy(self.cfg);c['global_batch']=64
        with self.assertRaises(ValueError):m.validate_config(c)

    def test_reject_extra_gpu(self):
        c=copy.deepcopy(self.cfg);c['gpu_indices'].append(7)
        with self.assertRaises(ValueError):m.validate_config(c)

    def test_reject_duplicate_gpu(self):
        c=copy.deepcopy(self.cfg);c['gpu_indices'][3]=0
        with self.assertRaises(ValueError):m.validate_config(c)

    def test_reject_changed_micro(self):
        c=copy.deepcopy(self.cfg);c['variants'][1]['micro_batch']=8
        with self.assertRaises(ValueError):m.validate_config(c)

    def test_reject_variant_path(self):
        c=copy.deepcopy(self.cfg);c['variants'][1]['id']='../other'
        with self.assertRaises(ValueError):m.validate_config(c)

    def test_reject_wrong_compression(self):
        c=copy.deepcopy(self.cfg);c['variants'][1]['communication']='fp8'
        with self.assertRaises(ValueError):m.validate_config(c)

    def test_reject_wrong_checkpoint_policy(self):
        c=copy.deepcopy(self.cfg);c['variants'][1]['checkpointing']='unknown'
        with self.assertRaises(ValueError):m.validate_config(c)

    def test_unrelated_gpu_work_allowed(self):
        g=[{'index':i,'uuid':str(i),'memory_used_mib':42000 if i in (2,3,4) else 0} for i in range(8)]
        p=[{'uuid':'2','pid':123}]
        self.assertEqual(len(m.check_selected_gpus(self.cfg,g,p)),4)

    def test_selected_work_blocks(self):
        g=[{'index':i,'uuid':str(i),'memory_used_mib':0} for i in range(8)]
        with self.assertRaises(RuntimeError):m.check_selected_gpus(self.cfg,g,[{'uuid':'0','pid':123}])

    def test_selected_memory_blocks(self):
        g=[{'index':i,'uuid':str(i),'memory_used_mib':101 if i==5 else 0} for i in range(8)]
        with self.assertRaises(RuntimeError):m.check_selected_gpus(self.cfg,g,[])

    def test_checkpoint_counts(self):
        self.assertEqual([sum(m.checkpoint_mask(p)) for p in ('full','half','none')],[32,16,0])
        self.assertEqual(m.checkpoint_mask('half')[:4],[True,False,True,False])

    def test_checkpoint_flags_on_model(self):
        class Model:
            def __init__(self):self.model=SimpleNamespace(layers=[SimpleNamespace(gradient_checkpointing=False,_gradient_checkpointing_func=True) for _ in range(32)])
            def gradient_checkpointing_enable(self,**kwargs):pass
        model=Model()
        self.assertEqual(m.configure_checkpointing(model,'half'),16)
        self.assertEqual([x.gradient_checkpointing for x in model.model.layers],m.checkpoint_mask('half'))

    def test_validation_spans_whole_pack(self):
        refs=m.validation_refs(5123,128)
        self.assertEqual((refs[0],refs[-1]),(0,5122))
        self.assertEqual(len(set(refs)),128)
        self.assertEqual([len(refs[r::4]) for r in range(4)],[32]*4)

    def test_validation_bad_size(self):
        with self.assertRaises(ValueError):m.validation_refs(10,128)

    def test_global_sample_mean(self):
        refs=list(range(32));gradients=[float(i*i) for i in refs]
        batches=[m.rank_microbatches(refs,r,4,4) for r in range(4)]
        self.assertEqual(sorted(i for rs in batches for batch in rs for i in batch),refs)
        actual=sum(sum(sum(gradients[i] for i in b)/4 for b in rs)/2 for rs in batches)/4
        self.assertEqual(actual,sum(gradients)/32)

    def test_candidate_screen(self):
        good={'status':'measured','worker_exit_code':0,'gradient_check':{'passed':True},'validation_screen':{'passed':True}}
        self.assertTrue(m.eligible(good))
        for key in ('gradient_check','validation_screen'):
            bad=copy.deepcopy(good);bad[key]['passed']=False
            self.assertFalse(m.eligible(bad))
        bad=dict(good,worker_exit_code=1);self.assertFalse(m.eligible(bad))


if __name__=='__main__':unittest.main()
