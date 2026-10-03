from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).parents[1]
SCRIPT=ROOT/'scripts/benchmark_perf8_v1.py'
CONFIG=ROOT/'configs/benchmarks/perf8_v1.json'
if not SCRIPT.exists():
    SCRIPT,CONFIG=ROOT/'benchmark_perf8_v1.py',ROOT/'config.json'
spec=importlib.util.spec_from_file_location('perf8',SCRIPT)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Perf8Test(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(CONFIG.read_text())

    def test_fixed_config(self):
        m.validate_config(self.cfg)

    def test_reject_global_batch_change(self):
        c=copy.deepcopy(self.cfg);c['global_batch']=64
        with self.assertRaises(ValueError):m.validate_config(c)

    def test_reject_extra_gpu(self):
        c=copy.deepcopy(self.cfg);c['gpu_indices'].append(8)
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

    def test_any_existing_cuda_work_blocks(self):
        g=[{'index':i,'uuid':str(i),'memory_used_mib':0} for i in range(8)]
        for i in range(8):
            with self.subTest(gpu=i):
                with self.assertRaises(RuntimeError):
                    m.check_selected_gpus(self.cfg,g,[{'uuid':str(i),'pid':123}])

    def test_all_eight_free_allowed(self):
        g=[{'index':i,'uuid':str(i),'memory_used_mib':0} for i in range(8)]
        self.assertEqual(len(m.check_selected_gpus(self.cfg,g,[])),8)

    def test_missing_gpu_blocks(self):
        g=[{'index':i,'uuid':str(i),'memory_used_mib':0} for i in range(7)]
        with self.assertRaises(RuntimeError):m.check_selected_gpus(self.cfg,g,[])

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
        self.assertEqual([len(refs[r::8]) for r in range(8)],[16]*8)

    def test_validation_bad_size(self):
        with self.assertRaises(ValueError):m.validation_refs(10,128)

    def test_global_sample_mean(self):
        refs=list(range(32));gradients=[float(i*i) for i in refs]
        batches=[m.rank_microbatches(refs,r,8,4) for r in range(8)]
        self.assertEqual(sorted(i for rs in batches for batch in rs for i in batch),refs)
        actual=sum(sum(sum(gradients[i] for i in b)/4 for b in rs) for rs in batches)/8
        self.assertEqual(actual,sum(gradients)/32)

    def test_candidate_screen(self):
        good={'status':'measured','worker_exit_code':0,'gradient_check':{'passed':True},'validation_screen':{'passed':True}}
        self.assertTrue(m.eligible(good))
        for key in ('gradient_check','validation_screen'):
            bad=copy.deepcopy(good);bad[key]['passed']=False
            self.assertFalse(m.eligible(bad))
        bad=dict(good,worker_exit_code=1);self.assertFalse(m.eligible(bad))


    def test_no_accumulation_in_eight_card_configuration(self):
        self.assertTrue(all(v['accumulation']==1 for v in self.cfg['variants']))
        self.assertTrue(all(v['micro_batch']*v['accumulation']*8==32 for v in self.cfg['variants']))

    def test_reject_accumulation_change(self):
        c=copy.deepcopy(self.cfg);c['variants'][1]['accumulation']=2
        with self.assertRaises(ValueError):m.validate_config(c)

    def test_reject_rank_count_change(self):
        c=copy.deepcopy(self.cfg);c['world_size']=4
        with self.assertRaises(ValueError):m.validate_config(c)

    def test_baseline_repeat_and_combined_candidate_present(self):
        by_id={v['id']:v for v in self.cfg['variants']}
        a=by_id[self.cfg['baseline_id']];b=by_id[self.cfg['baseline_id']+'_repeat']
        self.assertEqual({k:v for k,v in a.items() if k!='id'}, {k:v for k,v in b.items() if k!='id'})
        c=by_id['bf16_b25_halfgc']
        self.assertEqual((c['communication'],c['checkpointing']),('bf16','half'))
        self.assertFalse(any(v['checkpointing']=='none' for v in self.cfg['variants']))


    def test_profiler_excludes_duplicate_gpu_annotations(self):
        def event(name, ms, annotation=False):
            return SimpleNamespace(name=name, device_type='DeviceType.CUDA',is_user_annotation=annotation,
                                   time_range=SimpleNamespace(elapsed_us=lambda:ms*1000))
        report=m.summarize_cuda_activity([event('nccl:all_reduce',90,True),
                                         event('ncclDevKernel_AllReduce',90),event('gemm',30)])
        self.assertEqual(report['total_cuda_activity_duration_ms'],120)
        self.assertEqual(report['nccl_cuda_activity_duration_share'],0.75)

    def test_profiler_missing_annotation_flag_rejected(self):
        with self.assertRaises(ValueError):
            m.summarize_cuda_activity([SimpleNamespace(device_type='DeviceType.CUDA')])


    def test_runner_waits_for_whole_prior_campaign(self):
        script=ROOT/'deploy/p0-wave2-perf8-v1.sh'
        if not script.exists():script=ROOT/'run.sh'
        runner=script.read_text()
        self.assertIn('systemctl --user is-active --quiet perf4-g237.service',runner)
        self.assertIn('[[ "$prior_active" == 0 ]]',runner)
        self.assertLess(runner.index('prior_active=0'),runner.index('benchmark_perf8_v1.py prepare'))
        self.assertNotIn('systemctl --user stop',runner)


    def test_retry_output_stays_in_benchmark_root(self):
        self.assertEqual(m.benchmark_output('perf8_v1_retry1'),m.ROOT/'benchmarks/perf8_v1_retry1')
        for name in ('../run','perf8_v1/../code','perf8_v1_retry0','perf8_v1_retry-1','production'):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):m.benchmark_output(name)


if __name__=='__main__':unittest.main()
