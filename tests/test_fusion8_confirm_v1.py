from __future__ import annotations
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from _benchmark_namespace import import_benchmarks

ROOT = Path(__file__).parents[1]
m, guard = import_benchmarks('benchmark_fusion8_confirm_v1', 'benchmark_fusion8_confirm_gpu_guard_v1')


class Fusion8ConfirmTest(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / 'configs/benchmarks/fusion8_confirm_v1.json').read_text())

    def test_fixed_eight_geometry(self):
        m.validate_config(self.cfg)
        self.assertTrue(all(v['micro_batch'] * v['accumulation'] * 8 == 32 for v in self.cfg['variants']))
        self.assertNotEqual(m.OUT, m.PARENT)
        self.assertEqual(m.family.engine.OUT, m.OUT)

    def test_two_repeats_each(self):
        self.assertEqual([v['id'] for v in self.cfg['variants']],
            ['torch_full', 'rmsnorm_only', 'norm_swi', 'torch_full_repeat', 'rmsnorm_only_repeat', 'norm_swi_repeat'])
        for i in range(3):
            a, b = copy.deepcopy(self.cfg['variants'][i]), copy.deepcopy(self.cfg['variants'][i+3])
            a.pop('id'); b.pop('id')
            self.assertEqual(a, b)

    def test_reject_other_geometry(self):
        for key, value in [('world_size',4), ('global_batch',64), ('gpu_indices',[0,1,5,6]),
                           ('measurement_steps',10), ('fixed_learning_rate',1e-4), ('phase','four_card')]:
            c = copy.deepcopy(self.cfg); c[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): m.validate_config(c)

    def test_reject_changed_implementation(self):
        for key, value in [('micro_batch',8), ('accumulation',2), ('loss_backend','liger'),
                           ('checkpointing','half'), ('fuse_norm',False), ('communication','fp32')]:
            c = copy.deepcopy(self.cfg); c['variants'][1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): m.validate_config(c)

    def test_reject_swapped_order(self):
        c = copy.deepcopy(self.cfg); c['variants'][1:3] = reversed(c['variants'][1:3])
        with self.assertRaises(ValueError): m.validate_config(c)

    def test_any_selected_busy_blocks(self):
        c = self.cfg
        for index in range(8):
            snap = {'gpus':[{'index':i, 'uuid':str(i), 'memory_used_mib':0} for i in range(8)],
                    'processes':[{'uuid':str(index), 'pid':123}]}
            with patch.object(m.family.engine,'read',return_value=c), patch.object(m.family.engine,'gpu_snapshot',return_value=snap):
                with self.assertRaises(RuntimeError): m.preflight()

    def test_all_free_passes(self):
        snap = {'gpus':[{'index':i, 'uuid':str(i), 'memory_used_mib':0} for i in range(8)], 'processes':[]}
        with patch.object(m.family.engine,'read',return_value=self.cfg), patch.object(m.family.engine,'gpu_snapshot',return_value=snap):
            self.assertEqual(m.preflight(),snap)

    def test_repeat_uses_total_time_not_average_tps(self):
        def row(tps):
            return {'status':'measured','worker_exit_code':0,'gradient_check':{'passed':True},
                    'validation_screen':{'passed':True},'loss_tokens_per_second':tps}
        stats = m.repeat_stats([row(100),row(200)])
        self.assertAlmostEqual(stats['combined_tokens_per_second'],400/3)
        self.assertFalse(stats['stable_within_10pct'])
        bad = row(100); bad['gradient_check']['passed'] = False
        self.assertIsNone(m.repeat_stats([row(100),bad])['combined_tokens_per_second'])

    def test_guard_exact_unit(self):
        group = '0::/user.slice/app.slice/' + m.UNIT + '\n'
        guard.validate_parent(100,100,group,group)
        for wrong in [group.replace('confirm1','confirm2'), group.replace('.service','-other.service')]:
            with self.assertRaises(ValueError): guard.validate_parent(100,100,wrong,wrong)

    def test_bounded_queue_and_fail_closed_runner(self):
        text = (ROOT / 'deploy/p0-wave2-fusion8-confirm-v1.sh').read_text()
        self.assertIn('seq 1 120',text)
        self.assertIn('maximum_queue_seconds":3600',text)
        self.assertIn('--nproc_per_node=8',text)
        self.assertIn('if [[ "$rc" != 0 ]]; then exit "$rc"; fi',text)
        self.assertIn("r['validation_screen']['passed']",text)
        self.assertNotIn('systemctl --user stop',text)

    def test_numerical_adapter_untouched(self):
        text = (ROOT / 'scripts/benchmark_fusion8_confirm_v1.py').read_text()
        self.assertIn("p['fusion8_scripts'][script.name]",text)
        self.assertIn("parent['sample_plan_sha256']",text)
        self.assertIn("'capability_or_long_convergence_certified': False",text)
        self.assertNotIn('def install_fusion',text)


if __name__ == '__main__': unittest.main()
