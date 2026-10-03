from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / 'scripts/benchmark_microbatch_v1.py'
CONFIG = ROOT / 'configs/benchmarks/microbatch_v1.json'
# Also test the isolated server bundle, without copying the production repository.
if not SCRIPT.exists():
    SCRIPT, CONFIG = ROOT / 'benchmark_microbatch_v1.py', ROOT / 'config.json'
SPEC = importlib.util.spec_from_file_location('microbench', SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class MicrobatchBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(CONFIG.read_text())

    def test_config(self):
        module.validate_config(self.cfg)

    def test_reject_global_batch_change(self):
        c = copy.deepcopy(self.cfg)
        c['variants'][1]['accumulation'] = 4
        with self.assertRaises(ValueError):
            module.validate_config(c)

    def test_reject_duplicate(self):
        c = copy.deepcopy(self.cfg)
        c['variants'].append(c['variants'][0])
        with self.assertRaises(ValueError):
            module.validate_config(c)

    def test_all_samples_once(self):
        for micro in (1, 2, 4):
            samples = [r for rank in range(8) for batch in module.rank_microbatches(list(range(32)),rank,8,micro) for r in batch]
            self.assertEqual(sorted(samples),list(range(32)))

    def test_same_rank_ownership(self):
        for rank in range(8):
            expected = list(range(32))[rank::8]
            for micro in (1, 2, 4):
                self.assertEqual([r for b in module.rank_microbatches(list(range(32)),rank,8,micro) for r in b],expected)

    def test_gradient_mean(self):
        # Analytic toy gradients: micro mean / accumulation, then DDP rank mean.
        gradients = [float((i-12)**2) for i in range(32)]
        expected = sum(gradients)/32
        for micro in (1,2,4):
            actual = 0.0
            for rank in range(8):
                batches = module.rank_microbatches(list(range(32)),rank,8,micro)
                actual += sum(sum(gradients[i] for i in b)/len(b) for b in batches)/len(batches)/8
            self.assertEqual(actual,expected)

    def test_gradient_limits(self):
        limits = self.cfg['gradient_check']
        self.assertTrue(module.gradient_passed({'relative_l2':0.001,'cosine':0.99999},2.6,2.6001,limits))
        self.assertFalse(module.gradient_passed({'relative_l2':0.1,'cosine':0.99999},2.6,2.6001,limits))
        self.assertFalse(module.gradient_passed({'relative_l2':0.001,'cosine':float('nan')},2.6,2.6001,limits))

    def test_invalid_partition(self):
        for rank,world,micro,refs in ((8,8,1,list(range(32))),(0,8,3,list(range(32))),(0,0,1,[0])):
            with self.assertRaises(ValueError):
                module.rank_microbatches(refs,rank,world,micro)

    def test_profiler_excludes_duplicate_annotation(self):
        def event(name, milliseconds, annotation=False, device='DeviceType.CUDA'):
            return SimpleNamespace(name=name, device_type=device, is_user_annotation=annotation,
                                   time_range=SimpleNamespace(elapsed_us=lambda: milliseconds * 1000))
        result = module.summarize_cuda_activity([
            event('nccl:all_reduce', 100, annotation=True),
            event('ncclDevKernel_AllReduce', 100),
            event('DistributedDataParallel.forward', 50, annotation=True),
            event('gemm', 50), event('Memcpy HtoD', 10),
            event('cpu_activity', 1000, device='DeviceType.CPU')])
        self.assertEqual(result['total_cuda_activity_duration_ms'],160)
        self.assertEqual(result['nccl_cuda_activity_duration_ms'],100)
        self.assertAlmostEqual(result['nccl_cuda_activity_duration_share'],100/160)

    def test_profiler_requires_annotation_flag(self):
        event = SimpleNamespace(name='nccl:all_reduce',device_type='DeviceType.CUDA')
        with self.assertRaises(ValueError):
            module.summarize_cuda_activity([event])


if __name__ == '__main__':
    unittest.main()
