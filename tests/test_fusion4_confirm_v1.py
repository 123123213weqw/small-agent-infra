from __future__ import annotations
import copy,json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from _benchmark_namespace import import_benchmarks
m,guard=import_benchmarks('benchmark_fusion4_confirm_v1','benchmark_fusion4_confirm_gpu_guard_v1')


class Fusion4ConfirmTest(unittest.TestCase):
    def setUp(self):self.cfg=json.loads((ROOT/'configs/benchmarks/fusion4_confirm_v1.json').read_text())

    def test_fixed_configuration(self):
        m.validate_config(self.cfg)
        self.assertTrue(all(v['micro_batch']*v['accumulation']*4==32 for v in self.cfg['variants']))

    def test_reject_changed_geometry(self):
        for key,value in [('world_size',8),('global_batch',64),('gpu_indices',[0,1,2,3]),('phase','other')]:
            cfg=copy.deepcopy(self.cfg);cfg[key]=value
            with self.assertRaises(ValueError):m.validate_config(cfg)

    def test_reject_micro_or_accumulation_change(self):
        for key,value in [('micro_batch',8),('accumulation',1),('loss_backend','liger'),('fuse_norm',False)]:
            cfg=copy.deepcopy(self.cfg);cfg['variants'][1][key]=value
            with self.assertRaises(ValueError):m.validate_config(cfg)

    def test_busy_unselected_card_does_not_block(self):
        snap={'gpus':[{'index':i,'uuid':str(i),'memory_used_mib':42000 if i==3 else 0} for i in range(8)],
              'processes':[{'uuid':'3','pid':123}]}
        with patch.object(m,'read',return_value=self.cfg),patch.object(m,'gpu_snapshot',return_value=snap):
            self.assertEqual(m.preflight(),snap)

    def test_busy_selected_card_blocks(self):
        snap={'gpus':[{'index':i,'uuid':str(i),'memory_used_mib':0} for i in range(8)],
              'processes':[{'uuid':'5','pid':123}]}
        with patch.object(m,'read',return_value=self.cfg),patch.object(m,'gpu_snapshot',return_value=snap):
            with self.assertRaises(RuntimeError):m.preflight()

    def test_guard_and_runner_use_only_selected_four(self):
        text=(ROOT/'scripts/benchmark_fusion4_confirm_gpu_guard_v1.py').read_text()
        self.assertIn('bench.GPU_INDICES',text);self.assertIn("if p['uuid'] in uuids",text)
        text=(ROOT/'deploy/p0-wave2-fusion4-confirm-v1.sh').read_text()
        self.assertIn('CUDA_VISIBLE_DEVICES=0,1,5,6',text);self.assertIn('--nproc_per_node=4',text)

    def test_own_scope_exact(self):
        group='0::/user.slice/app.slice/'+m.UNIT+'\n'
        guard.validate_parent(100,100,group,group)
        wrong=group.replace('fusion4-confirm1','fusion8')
        with self.assertRaises(ValueError):guard.validate_parent(100,100,wrong,wrong)

    def test_four_card_not_eight_claim(self):
        text=(ROOT/'scripts/benchmark_fusion4_confirm_v1.py').read_text()
        self.assertIn("'eight_card_repeat_completed':False",text)
        self.assertIn('Not an eight-card repeat',text)


if __name__=='__main__':unittest.main()
