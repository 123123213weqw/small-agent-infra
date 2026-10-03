from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from small_agent.training import kernel_state as state


def dummy_kernel():pass


class Config:
    def __init__(self,warps=4):
        self.kwargs={'BK':128};self.num_warps=warps;self.num_stages=2;self.num_ctas=1;self.maxnreg=None


class Tuner:
    def __init__(self):self.base_fn=dummy_kernel;self.configs=[Config(2),Config(4)];self.cache={}


class KernelTests(unittest.TestCase):
    def setUp(self):
        try:import triton
        except ImportError:self.skipTest('remote Triton runtime required')

    def test_selected_declared_config_roundtrip(self):
        tuner=Tuner();tuner.cache[(2048,'torch.bfloat16')]=tuner.configs[1]
        with mock.patch.object(state,'registry',return_value={'kernel':tuner}):
            saved=state.capture_kernel_state();tuner.cache.clear();state.restore_kernel_state(saved)
        self.assertIs(tuner.cache[(2048,'torch.bfloat16')],tuner.configs[1])

    def test_unknown_choice_rejected_before_mutation(self):
        tuner=Tuner();tuner.cache[(1,)]=tuner.configs[0]
        with mock.patch.object(state,'registry',return_value={'kernel':tuner}):
            saved=state.capture_kernel_state();tuner.cache.clear();saved['entries']['kernel']['choices'][0]['config']['num_warps']=128
            with self.assertRaises(ValueError):state.restore_kernel_state(saved)
        self.assertEqual(tuner.cache,{})

    def test_kernel_source_drift_rejected(self):
        tuner=Tuner();tuner.cache[(1,)]=tuner.configs[0]
        with mock.patch.object(state,'registry',return_value={'kernel':tuner}):
            saved=state.capture_kernel_state();saved['entries']['kernel']['source_sha256']='different';tuner.cache.clear()
            with self.assertRaises(ValueError):state.restore_kernel_state(saved)
        self.assertEqual(tuner.cache,{})

    def test_legacy_checkpoint_without_tuning_state_supported(self):state.restore_kernel_state(None)


if __name__=='__main__':unittest.main()
