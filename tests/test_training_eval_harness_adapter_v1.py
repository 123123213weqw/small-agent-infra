import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

try:
    import torch
    from tokenizers import Tokenizer,models,pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from lm_eval.models.utils_hf import stop_sequences_criteria
    AVAILABLE=True
except ImportError:
    AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'run integration tests in isolated remote eval environment')
class TestHarnessGenerationAdapter(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('training_eval_harness_adapter_v1',Path(__file__).resolve().parents[1]/'scripts/training_eval_harness_adapter_v1.py')
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        raw=Tokenizer(models.WordLevel({'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3,'Question':4,':':5},unk_token='<unk>'))
        raw.pre_tokenizer=pre_tokenizers.Whitespace()
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='<pad>',bos_token='<bos>',eos_token='<eos>')
        self.kw=None
        outer=self
        class Fake:
            def generate(self,**kw): outer.kw=kw;return kw['input_ids']
        self.lm=SimpleNamespace(tokenizer=tokenizer,model=Fake(),device=torch.device('cpu'),mixed_precision_dtype=torch.bfloat16)
        self.inputs=torch.ones(1,4,dtype=torch.long)
    def test_no_duplicate_cache_keyword(self):
        self.module.generate(self.lm,self.inputs,6,['Question:'],False,{'use_cache':True})
        self.assertIs(self.kw['use_cache'],False)
        self.assertEqual(self.kw['max_length'],6)
    def test_explicit_cache_allowed(self):
        self.module.generate(self.lm,self.inputs,6,[],True,{})
        self.assertIs(self.kw['use_cache'],True)
    def test_greedy_temperature_removed(self):
        self.module.generate(self.lm,self.inputs,6,[],False,{'temperature':0.0})
        self.assertNotIn('temperature',self.kw)
        self.assertIs(self.kw['do_sample'],False)
    def test_mask_and_stop_criteria_preserved(self):
        mask=torch.ones_like(self.inputs)
        self.module.generate(self.lm,self.inputs,6,['Question:'],False,{'attention_mask':mask})
        self.assertIs(self.kw['attention_mask'],mask)
        self.assertEqual(len(self.kw['stopping_criteria']),1)


if __name__=='__main__': unittest.main()
