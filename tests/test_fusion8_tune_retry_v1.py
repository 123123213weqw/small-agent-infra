import copy
import json
from pathlib import Path
import unittest
from _benchmark_namespace import import_benchmarks

ROOT=Path(__file__).parents[1]
m,guard=import_benchmarks('benchmark_fusion8_tune_retry_v1','benchmark_fusion8_tune_retry_gpu_guard_v1')


class Fusion8TuneRetryTest(unittest.TestCase):
    def setUp(self): self.cfg=json.loads((ROOT/'configs/benchmarks/fusion8_tune_v1.json').read_text())

    def test_fixed_geometry_and_original_ce(self):
        m.validate_config(self.cfg)
        self.assertEqual(m.engine.OUT,m.OUT)
        self.assertNotEqual(m.OUT,m.PREREQUISITE)
        self.assertEqual(m.OUT.name,'fusion8_tune_v1_retry1')
        self.assertEqual(m.UNIT,'p0-wave2-fusion8-tune-retry1-v1.service')
        self.assertTrue(all(v['micro_batch']*v['accumulation']*8==32 for v in self.cfg['variants']))

    def test_predeclared_two_factors_only(self):
        self.assertEqual([(v['checkpointing'],v['bucket_mib']) for v in self.cfg['variants']],
                         [('full',25),('half',25),('full',100),('half',100),('full',25)])
        self.assertTrue(all(v['fuse_norm'] and not v['fuse_swiglu'] for v in self.cfg['variants']))

    def test_reject_math_geometry_and_budget_changes(self):
        for key,value in [('world_size',4),('global_batch',64),('seed',1),('measurement_steps',100),('phase','other')]:
            c=copy.deepcopy(self.cfg);c[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate_config(c)

    def test_reject_unplanned_variants(self):
        for key,value in [('micro_batch',8),('fuse_swiglu',True),('fuse_norm',False),('loss_backend','liger'),
                          ('bucket_mib',200),('checkpointing','none')]:
            c=copy.deepcopy(self.cfg);c['variants'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate_config(c)

    def test_activating_prerequisite_must_block_queue(self):
        s=(ROOT/'deploy/p0-wave2-fusion8-tune-retry1-v1.sh').read_text()
        self.assertIn('active|activating|reloading|deactivating)',s)
        self.assertIn('sleep 5; continue',s)
        self.assertIn('seq 1 720',s)
        self.assertIn('prerequisite_check.log',s)
        self.assertIn('run_variant winner_confirmation',s)
        self.assertNotIn('systemctl --user stop',s)

    def test_fail_closed_before_next_trial(self):
        s=(ROOT/'deploy/p0-wave2-fusion8-tune-retry1-v1.sh').read_text()
        self.assertIn('if [[ "$rc" != 0 ]]; then exit "$rc"; fi',s)
        self.assertIn("r['validation_screen']['passed']",s)
        self.assertIn('--nproc_per_node=8',s)

    def test_no_production_math_change(self):
        s=(ROOT/'scripts/benchmark_fusion8_tune_retry_v1.py').read_text()
        self.assertIn("'production_config_changed': False",s)
        self.assertIn("'capability_or_long_convergence_certified': False",s)
        self.assertIn("p['fusion8_scripts'][script.name]",s)
        self.assertNotIn('def install_fusion',s)

    def test_guard_exact_unit(self):
        cg='0::/user.slice/app.slice/'+m.UNIT+'\n'
        guard.validate_parent(100,100,cg,cg)
        wrong=cg.replace('tune','confirm1')
        with self.assertRaises(ValueError):guard.validate_parent(100,100,wrong,wrong)


if __name__=='__main__': unittest.main()
