from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from _benchmark_namespace import import_benchmarks
m,guard=import_benchmarks('benchmark_fusion8_v1','benchmark_fusion8_gpu_guard_v1')


class Fusion8Test(unittest.TestCase):
    def setUp(self):self.cfg=json.loads((ROOT/'configs/benchmarks/fusion8_v1.json').read_text())

    def test_fixed_config(self):m.validate_config(self.cfg)

    def test_reject_core_changes(self):
        for key,value in [('global_batch',64),('world_size',4),('sequence_length',4096),('measurement_steps',10),
                          ('profile_steps',0),('validation_samples',16),('fixed_learning_rate',0.0001),('baseline_id','norm_swi')]:
            c=copy.deepcopy(self.cfg);c[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate_config(c)

    def test_reject_variant_changes(self):
        for key,value in [('id','../code'),('micro_batch',8),('accumulation',2),('bucket_mib',100),
                          ('communication','fp32'),('checkpointing','half'),('loss_backend','liger'),('fuse_swiglu',True)]:
            c=copy.deepcopy(self.cfg);c['variants'][1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate_config(c)

    def test_individual_and_combined_with_controls(self):
        self.assertEqual([(v['fuse_norm'],v['fuse_swiglu']) for v in self.cfg['variants']],
                         [(False,False),(True,False),(False,True),(True,True),(False,False),(True,True)])
        self.assertTrue(all(v['loss_backend']=='torch' for v in self.cfg['variants']))

    def test_reject_gpu_mutations(self):
        for indices in [list(range(7)),[0]*8,list(reversed(range(8)))]:
            c=copy.deepcopy(self.cfg);c['gpu_indices']=indices
            with self.assertRaises(ValueError):m.validate_config(c)

    def test_reject_permuted_group_order(self):
        c=copy.deepcopy(self.cfg);c['variants'][1:3]=reversed(c['variants'][1:3])
        with self.assertRaises(ValueError):m.validate_config(c)

    def test_combined_tps_requires_all_gates(self):
        def row(tps):return {'status':'measured','worker_exit_code':0,'gradient_check':{'passed':True},
                            'validation_screen':{'passed':True},'loss_tokens_per_second':tps}
        self.assertAlmostEqual(m.combined_tps([row(100),row(200)]),400/3)
        r=row(100);r['validation_screen']['passed']=False
        self.assertIsNone(m.combined_tps([row(200),r]));self.assertIsNone(m.combined_tps([]))

    def test_parameter_alias_order_detected(self):
        a,b=SimpleNamespace(shape=(3,4),dtype='float32'),SimpleNamespace(shape=(3,4),dtype='float32')
        model=SimpleNamespace(named_parameters=lambda:[('a',a),('b',b)])
        before=m.backend.parameter_identity(model)
        model.named_parameters=lambda:[('a',b),('b',a)]
        self.assertNotEqual(before,m.backend.parameter_identity(model))

    def test_gradient_thresholds(self):
        self.assertTrue(m.backend.passes_gradient({'relative_l2':0.019,'cosine':0.9995}))
        self.assertFalse(m.backend.passes_gradient({'relative_l2':0.021,'cosine':0.9995}))
        self.assertFalse(m.backend.passes_gradient({'relative_l2':0.001,'cosine':0.998}))

    def test_offset_cast_and_residual_safety(self):
        text=(ROOT/'scripts/benchmark_fusion_backend_v1.py').read_text()
        self.assertIn("self.eps,1.0,'gemma',False,None",text)
        self.assertIn('type(module) is Qwen3NextRMSNorm',text)
        self.assertNotIn('type(module) is Qwen3NextRMSNormGated',text)
        self.assertIn('if not self.training: return original(x)',text)

    def test_full_gradient_gate_before_updates(self):
        text=(ROOT/'scripts/benchmark_fusion8_v1.py').read_text()
        self.assertIn('engine.gradient_passed=fail_closed_gradient',text)
        self.assertIn('full-model gradient/loss gate failed before updates',text)

    def test_old_engine_and_loss_unchanged(self):
        text=(ROOT/'scripts/benchmark_perf8_v1.py').read_text()
        self.assertNotIn('liger',text)
        self.assertEqual(m.engine.OUT,m.OUT)

    def test_runner_queues_bounds_and_private_dependencies(self):
        text=(ROOT/'deploy/p0-wave2-fusion8-v1.sh').read_text()
        self.assertIn('$OUT/deps/vendor:',text);self.assertIn('seq 1 30',text)
        self.assertIn('p0-wave2-base1b-full8-v1.service',text)
        self.assertIn('rm -f "$OUT/tmp/baseline_gradients.pt"',text)
        self.assertNotIn('systemctl --user stop',text);self.assertNotIn('pip install',text)

    def test_guard_exact_own_scope(self):
        group='0::/user.slice/app.slice/'+m.UNIT+'\n'
        guard.validate_parent(100,100,group,group)
        for parent,actual,a,b in [(1,1,group,group),(100,101,group,group),(100,100,group,'other'),
                                 (100,100,group.replace('.service','-other.service'),group.replace('.service','-other.service'))]:
            with self.assertRaises(ValueError):guard.validate_parent(parent,actual,a,b)

    def test_guard_ignores_exited_but_identifies_external(self):
        groups={100:'own',200:'other'}
        result=guard.outside_processes([{'pid':100},{'pid':200}],'own',groups.__getitem__)
        self.assertEqual([r['pid'] for r in result],[200])
        def gone(pid):raise FileNotFoundError
        self.assertEqual(guard.outside_processes([{'pid':201}],'own',gone),[])


if __name__=='__main__':unittest.main()
