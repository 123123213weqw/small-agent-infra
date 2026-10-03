#!/usr/bin/env python3
"""Instance-only Qwen3Next dense training-loss adapter; eval stays unfused."""
from __future__ import annotations

from contextlib import nullcontext
import gc
import math
from types import MethodType


def parameter_identity(model):
    return [(n, id(p), tuple(p.shape), str(p.dtype)) for n, p in model.named_parameters()]


def install_fused_loss(model):
    import torch
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    from transformers.modeling_outputs import MoeCausalLMOutputWithPast

    if model.__class__.__name__ != 'Qwen3NextForCausalLM':
        raise ValueError('adapter only supports the frozen dense Qwen3Next model')
    if model.config.num_experts != 0 or model.config.output_router_logits:
        raise ValueError('MoE/router losses are deliberately unsupported')
    if getattr(model, '_loss8_adapter_installed', False):
        raise ValueError('adapter already installed')
    if model.lm_head.bias is not None:
        raise ValueError('frozen output head is bias-free')
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError('must preserve the frozen tied embedding/head')
    before = parameter_identity(model)
    original = model.forward
    loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction='mean', accum_dtype=torch.float32)

    def fused_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                      past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                      output_router_logits=None, cache_position=None, logits_to_keep=0, **kwargs):
        call = dict(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                    past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                    use_cache=use_cache, output_router_logits=output_router_logits,
                    cache_position=cache_position, logits_to_keep=logits_to_keep, **kwargs)
        # Independent reference CE for every heldout probe and ordinary generation.
        if not self.training or labels is None:
            return original(**call)
        if use_cache is not False or past_key_values is not None or output_router_logits:
            raise ValueError('fused training requires no cache and no router loss')
        if not isinstance(logits_to_keep, int) or logits_to_keep != 0 or kwargs:
            raise ValueError('unsupported loss kwargs or sliced logits; never silently change reduction')
        if labels.ndim != 2 or labels.shape[1] < 2:
            raise ValueError('causal targets must have at least two positions')
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=None,
                             inputs_embeds=inputs_embeds, use_cache=False,
                             output_router_logits=False, cache_position=cache_position)
        hidden = outputs.last_hidden_state
        if hidden.shape[:2] != labels.shape:
            raise ValueError('hidden/label shape mismatch')
        x = hidden[:, :-1, :].contiguous().view(-1, hidden.shape[-1])
        targets = labels[:, 1:].contiguous().view(-1)
        # This trial has full train sequences. Kernel preflight also tests masked targets.
        loss = loss_fn(self.lm_head.weight, x, targets)
        return MoeCausalLMOutputWithPast(loss=loss, logits=None,
                    past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions, router_logits=outputs.router_logits)

    model.forward = MethodType(fused_forward, model)
    model._loss8_adapter_installed = True
    if parameter_identity(model) != before or model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError('adapter changed parameter identity, order, dtype or aliasing')
    return {'backend': 'liger_flce_0.6.4', 'parameter_identity_preserved': True,
            'tied_embedding_preserved': True, 'training_returns_logits': False,
            'evaluation_and_generation': 'original HuggingFace forward, FP32 CE',
            'causal_shift': 'hidden[:, :-1] against labels[:, 1:]',
            'reduction': 'mean over non-ignore targets', 'weight_gradient_accumulation': 'float32',
            'other_modules_patched': False}


def compare_gradients(reference, candidate):
    import torch
    aa = bb = ab = dd = maximum = 0.0
    if len(reference) != len(candidate):
        raise ValueError('gradient coverage mismatch')
    for a, b in zip(reference, candidate):
        a, b = a.float(), b.float()
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('invalid gradient')
        aa += float(a.square().sum()); bb += float(b.square().sum())
        ab += float((a*b).sum()); dd += float((a-b).square().sum())
        maximum = max(maximum, float((a-b).abs().max()))
    return {'relative_l2': math.sqrt(dd/max(aa, 1e-30)),
            'cosine': ab/math.sqrt(max(aa*bb, 1e-60)), 'maximum_absolute_difference': maximum}


def kernel_preflight():
    """Real GPU operator, AMP, tied-head adapter, eval fallback and masking checks."""
    import torch
    import torch.nn.functional as F
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    torch.cuda.set_device(0)
    torch.manual_seed(20261002)
    fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction='mean', accum_dtype=torch.float32)
    cases = []
    # The last case exercises actual production projection dimensions without a model checkpoint.
    for amp, bt, h, v in [(False,37,96,257), (True,37,96,257), (True,8188,1536,32768)]:
        w0 = torch.randn(v,h,device='cuda',dtype=torch.float32)/math.sqrt(h)
        x0 = torch.randn(bt,h,device='cuda',dtype=torch.float32)
        target = torch.randint(v,(bt,),device='cuda'); target[::7] = -100
        results = []
        for backend in ('torch','liger'):
            x = x0.detach().clone().requires_grad_(); w = w0.detach().clone().requires_grad_()
            context = torch.autocast('cuda',dtype=torch.bfloat16) if amp else nullcontext()
            with context:
                loss = F.cross_entropy((x@w.t()).float(),target,ignore_index=-100) if backend=='torch' else fn(w,x,target)
            (loss*0.37).backward()  # Non-unit upstream derivative is intentional.
            results.append((float(loss), [x.grad.detach(),w.grad.detach()]))
            del x,w,loss
        metrics = compare_gradients(results[0][1],results[1][1])
        metrics.update({'amp_bfloat16':amp,'shape':[bt,h,v],
                        'loss_difference':results[1][0]-results[0][0]})
        metrics['passed'] = (abs(metrics['loss_difference'])<=0.01 and metrics['relative_l2']<=0.02 and metrics['cosine']>=0.999)
        cases.append(metrics)
        del results,w0,x0,target
        gc.collect(); torch.cuda.empty_cache()

    config = Qwen3NextConfig(vocab_size=257,hidden_size=96,intermediate_size=192,
              num_hidden_layers=1,layer_types=['full_attention'],num_attention_heads=2,
              num_key_value_heads=1,head_dim=48,num_experts=0,num_experts_per_tok=0,
              tie_word_embeddings=True,attention_dropout=0.0,use_cache=False,
              linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=48,
              linear_value_head_dim=48,output_router_logits=False)
    model = Qwen3NextForCausalLM(config).cuda().train()
    ids = torch.randint(3,257,(2,17),device='cuda')
    labels = ids.clone(); labels[0,-3:] = -100; labels[1,5] = -100
    mask = torch.ones_like(ids)
    before = parameter_identity(model)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        ref = model(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False).loss
    ref.backward()
    grads = [p.grad.detach().clone() for p in model.parameters()]
    ref_loss = float(ref); model.zero_grad(set_to_none=True)
    adapter = install_fused_loss(model)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out = model(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False)
    assert out.logits is None
    out.loss.backward()
    metrics = compare_gradients(grads,[p.grad for p in model.parameters()])
    metrics['loss_difference'] = float(out.loss)-ref_loss
    metrics['passed'] = abs(metrics['loss_difference'])<=0.01 and metrics['relative_l2']<=0.02 and metrics['cosine']>=0.999
    assert parameter_identity(model)==before
    model.eval()
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        fallback = model(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False)
        generation = model(input_ids=ids,attention_mask=mask,use_cache=False)
    metrics['eval_fallback_loss_difference'] = float(fallback.loss)-ref_loss
    metrics['eval_returns_logits'] = tuple(fallback.logits.shape)==(2,17,257)
    metrics['generation_returns_logits'] = tuple(generation.logits.shape)==(2,17,257)
    metrics['passed'] &= metrics['eval_fallback_loss_difference']==0.0 and metrics['eval_returns_logits'] and metrics['generation_returns_logits']
    return {'status':'passed' if all(c['passed'] for c in cases) and metrics['passed'] else 'failed',
            'operator_cases':cases,'tied_model_adapter':metrics,'adapter':adapter,
            'torch':torch.__version__,'cuda':torch.version.cuda,
            'not_full_model_gradient_check':True}
