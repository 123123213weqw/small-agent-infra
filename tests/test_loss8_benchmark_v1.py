from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from _benchmark_namespace import import_benchmarks
m,guard=import_benchmarks('benchmark_loss8_v1','benchmark_loss8_gpu_guard_v1')


class Loss8Test(unittest.TestCase):
    def setUp(self):self.cfg=json.loads((ROOT/'configs/benchmarks/loss8_v1.json').read_text())

    def test_fixed_config(self):m.validate_config(self.cfg)

    def test_only_loss_varies(self):
        settings=[{k:v for k,v in row.items() if k not in ('id','loss_backend')} for row in self.cfg['variants']]
        self.assertTrue(all(s==settings[0] for s in settings))

    def test_no_fastest_run_selection(self):
        for key,value in [('baseline_id','liger_flce'),('global_batch',64),('world_size',4),('sequence_length',4096),
                          ('measurement_steps',10),('fixed_learning_rate',0.0001)]:
            c=copy.deepcopy(self.cfg);c[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate_config(c)

    def test_reject_variant_mutations(self):
        for key,value in [('id','../code'),('loss_backend','torch'),('micro_batch',8),('accumulation',2),
                          ('bucket_mib',100),('communication','fp32'),('checkpointing','half')]:
            c=copy.deepcopy(self.cfg);c['variants'][1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate_config(c)

    def test_reject_gpu_order(self):
        c=copy.deepcopy(self.cfg);c['gpu_indices']=list(reversed(c['gpu_indices']))
        with self.assertRaises(ValueError):m.validate_config(c)

    def test_equal_work_combined_tps(self):
        def result(tps):return {'status':'measured','worker_exit_code':0,'gradient_check':{'passed':True},
                               'validation_screen':{'passed':True},'loss_tokens_per_second':tps}
        self.assertAlmostEqual(m.combined_tps([result(100),result(200)]),400/3)
        r=result(100);r['gradient_check']['passed']=False
        self.assertIsNone(m.combined_tps([result(200),r]));self.assertIsNone(m.combined_tps([]))

    def test_parameter_identity_detects_order_and_alias(self):
        a,b=SimpleNamespace(shape=(3,4),dtype='float32'),SimpleNamespace(shape=(3,4),dtype='float32')
        model=SimpleNamespace(named_parameters=lambda:[('embed',a),('norm',b)])
        before=m.backend.parameter_identity(model)
        model.named_parameters=lambda:[('embed',b),('norm',a)]
        self.assertNotEqual(before,m.backend.parameter_identity(model))

    def test_prior_engine_not_edited(self):
        text=(ROOT/'scripts/benchmark_perf8_v1.py').read_text()
        self.assertNotIn('liger',text)
        self.assertEqual(m.engine.OUT,m.OUT)

    def test_runner_pinned_dependency_and_bounded(self):
        text=(ROOT/'deploy/p0-wave2-loss8-v1.sh').read_text()
        self.assertIn('$OUT/deps/vendor:',text)
        self.assertIn('seq 1 30',text)
        self.assertIn('rm -f "$OUT/tmp/baseline_gradients.pt"',text)
        self.assertNotIn('systemctl --user stop',text)
        self.assertNotIn('pip install',text)

    def test_guard_only_exact_own_scope(self):
        group='0::/user.slice/app.slice/'+m.UNIT+'\n'
        guard.validate_parent(100,100,group,group)
        for parent,actual,a,b in [(1,1,group,group),(100,101,group,group),(100,100,group,'other'),
                                 (100,100,group.replace('.service','-other.service'),group.replace('.service','-other.service'))]:
            with self.assertRaises(ValueError):guard.validate_parent(parent,actual,a,b)

    def test_guard_never_targets_external_pid(self):
        groups={100:'own',200:'other'}
        result=guard.outside_processes([{'pid':100},{'pid':200}],'own',groups.__getitem__)
        self.assertEqual([r['pid'] for r in result],[200])
        def gone(pid):raise FileNotFoundError
        self.assertEqual(guard.outside_processes([{'pid':201}],'own',gone),[])


if __name__=='__main__':unittest.main()
