#!/usr/bin/env python3
"""Training-only, instance-level RMSNorm/SiLU-mul fusion with original eval."""
from __future__ import annotations
import copy
import gc
import math
from types import MethodType


def parameter_identity(model):
    return [(n,id(p),tuple(p.shape),str(p.dtype)) for n,p in model.named_parameters()]


def gradient_metrics(reference,candidate):
    import torch
    if len(reference)!=len(candidate): raise ValueError('gradient coverage changed')
    aa=bb=ab=dd=maximum=0.0
    for a,b in zip(reference,candidate):
        if a is None or b is None or a.shape!=b.shape: raise ValueError('invalid gradient coverage')
        a,b=a.float(),b.float()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all(): raise ValueError('nonfinite gradient')
        aa+=float(a.square().sum());bb+=float(b.square().sum());ab+=float((a*b).sum())
        dd+=float((a-b).square().sum());maximum=max(maximum,float((a-b).abs().max()))
    return {'relative_l2':math.sqrt(dd/max(aa,1e-30)), 'cosine':ab/math.sqrt(max(aa*bb,1e-60)),
            'maximum_absolute_difference':maximum}


def tensor_metrics(reference,candidate):
    return gradient_metrics([reference],[candidate])


def passes_gradient(metrics):
    return metrics['relative_l2']<=0.02 and metrics['cosine']>=0.999


def install_fusion(model, fuse_norm, fuse_swiglu):
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm,Qwen3NextMLP
    if model.__class__.__name__!='Qwen3NextForCausalLM' or model.config.num_experts!=0:
        raise ValueError('only the frozen dense Qwen3Next model is supported')
    if model.config.hidden_act!='silu': raise ValueError('only the existing SiLU activation is supported')
    if getattr(model,'_fusion8_installed',False): raise ValueError('already patched')
    before=parameter_identity(model)
    if model.lm_head.weight is not model.model.embed_tokens.weight: raise ValueError('tied head required')
    norms=[];mlps=[]

    def patch_norm(module):
        original=module.forward
        def forward(self,x):
            if not self.training: return original(x)
            # Qwen3Next is zero-centered: (x/rms)*(1+w), all FP32 math then cast.
            # No in-place upstream-gradient overwrite: residual branches share it.
            return LigerRMSNormFunction.apply(x,self.weight,self.eps,1.0,'gemma',False,None)
        module.forward=MethodType(forward,module)

    def patch_mlp(module):
        original=module.forward
        def forward(self,x):
            if not self.training: return original(x)
            # The Liger backward overwrites only these private projection outputs,
            # never the shared residual input, parameter, or upstream gradient.
            a=self.gate_proj(x);b=self.up_proj(x)
            return self.down_proj(LigerSiLUMulFunction.apply(a,b))
        module.forward=MethodType(forward,module)

    for name,module in model.named_modules():
        if type(module) is Qwen3NextRMSNorm and fuse_norm:
            patch_norm(module);norms.append(name)
        elif type(module) is Qwen3NextMLP and fuse_swiglu:
            patch_mlp(module);mlps.append(name)
    expected_norms=2*model.config.num_hidden_layers+1+2*model.config.layer_types.count('full_attention')
    if len(norms)!=(expected_norms if fuse_norm else 0) or len(mlps)!=(model.config.num_hidden_layers if fuse_swiglu else 0):
        raise ValueError('unexpected fusion module coverage')
    if before!=parameter_identity(model) or model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError('parameter identity, order, dtype or alias changed')
    model._fusion8_installed=True
    return {'norm_module_count':len(norms),'swiglu_module_count':len(mlps),
            'norm_module_names':norms,'swiglu_module_names':mlps,
            'norm_offset':1.0,'norm_casting_mode':'gemma','norm_in_place':False,
            'gdn_gated_norm_patched':False,'gdn_or_attention_math_changed':False,
            'loss_backend':'original HF CE, unfused','output_linear_layers_changed':False,
            'parameter_identity_preserved':True,'tied_embedding_preserved':True,
            'eval_and_generation':'original module forwards for all patched modules',
            'training_only_patch':True,'liger_version':'0.6.4'}


def kernel_preflight():
    import torch
    import torch.nn.functional as F
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    from transformers import Qwen3NextConfig,Qwen3NextForCausalLM
    torch.cuda.set_device(0);torch.manual_seed(20261002)
    cases=[]
    # Learned, nonzero, FP32 norm weights; residual and shared-upstream checks.
    for dtype,h,rows,strided in [(torch.float32,1536,257,False),
             (torch.bfloat16,1536,8192,False),(torch.bfloat16,96,4096,True)]:
        x0=torch.randn(rows,h*(2 if strided else 1),device='cuda',dtype=dtype)
        if strided:x0=x0[:,::2]
        w0=torch.randn(h,device='cuda')*0.1
        dy=torch.randn(rows,h,device='cuda',dtype=dtype)
        dy_original=dy.clone();runs=[];untouched=True
        for kind in ('torch','liger'):
            # Preserve a genuine noncontiguous view for the small head-norm test.
            x=x0.detach().requires_grad_();w=w0.detach().clone().requires_grad_()
            snapshot=x.detach().clone()
            y=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*(1+w.float())).to(dtype) if kind=='torch' else LigerRMSNormFunction.apply(x,w,1e-6,1.0,'gemma',False,None)
            # Branches must still see unmodified x and dy after backward.
            ((y.float()*dy.float()).mean()+0.19*x.float().square().mean()).backward()
            runs.append((y.detach().clone(),[x.grad.detach().clone(),w.grad.detach().clone()]))
            untouched &= torch.equal(x.detach(),snapshot) and torch.equal(dy,dy_original)
            del x,w,y,snapshot
        forward=tensor_metrics(runs[0][0],runs[1][0]);grad=gradient_metrics(runs[0][1],runs[1][1])
        cases.append({'operator':'rmsnorm','input_dtype':str(dtype),'shape':[rows,h],
                      'noncontiguous_input':strided,'offset':1.0,'in_place':False,
                      'forward':forward,'gradient':grad,'residual_and_upstream_unchanged':bool(untouched),
                      'passed':forward['relative_l2']<=0.01 and passes_gradient(grad) and bool(untouched)})
        del runs,x0,w0,dy,dy_original
        gc.collect();torch.cuda.empty_cache()

    for dtype,rows,cols in [(torch.float32,257,192),(torch.bfloat16,8192,4096)]:
        a0=torch.randn(rows,cols,device='cuda',dtype=dtype);b0=torch.randn_like(a0)
        dy=torch.randn_like(a0);runs=[]
        for kind in ('torch','liger'):
            a=a0.detach().clone().requires_grad_();b=b0.detach().clone().requires_grad_()
            y=F.silu(a)*b if kind=='torch' else LigerSiLUMulFunction.apply(a,b)
            y.backward(dy)
            runs.append((y.detach().clone(),[a.grad.detach().clone(),b.grad.detach().clone()]))
            del a,b,y
        forward=tensor_metrics(runs[0][0],runs[1][0]);grad=gradient_metrics(runs[0][1],runs[1][1])
        cases.append({'operator':'silu_mul','input_dtype':str(dtype),'shape':[rows,cols],
                      'forward':forward,'gradient':grad,
                      'passed':forward['relative_l2']<=0.01 and passes_gradient(grad)})
        del runs,a0,b0,dy
        gc.collect();torch.cuda.empty_cache()

    config=Qwen3NextConfig(vocab_size=257,hidden_size=96,intermediate_size=192,
            num_hidden_layers=4,layer_types=['linear_attention']*3+['full_attention'],
            num_attention_heads=2,num_key_value_heads=1,head_dim=48,num_experts=0,
            num_experts_per_tok=0,tie_word_embeddings=True,attention_dropout=0.0,use_cache=False,
            linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=48,
            linear_value_head_dim=48,output_router_logits=False,dtype=torch.float32)
    base=Qwen3NextForCausalLM(config).cuda()
    for module in base.modules():
        if module.__class__.__name__=='Qwen3NextRMSNorm':torch.nn.init.uniform_(module.weight,-0.1,0.1)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    ids=torch.randint(3,257,(2,65),device='cuda');labels=ids.clone();labels[0,-3:]=-100
    inputs={'input_ids':ids,'labels':labels,'attention_mask':torch.ones_like(ids),'use_cache':False}
    base.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):eval_ref=base(**inputs)
    base.train()
    with torch.autocast('cuda',dtype=torch.bfloat16):ref=base(**inputs)
    ref.loss.backward();grads=[p.grad.detach().clone() for p in base.parameters()]
    adapters=[]
    for norm,swi in [(True,False),(False,True),(True,True)]:
        model=copy.deepcopy(base);model.zero_grad(set_to_none=True)
        info=install_fusion(model,norm,swi);model.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**inputs)
        out.loss.backward()
        grad=gradient_metrics(grads,[p.grad for p in model.parameters()])
        model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):fallback=model(**inputs)
        eval_delta=float(fallback.loss-eval_ref.loss)
        metrics={'fuse_norm':norm,'fuse_swiglu':swi,'adapter':info,'gradient':grad,
                 'train_loss_difference':float(out.loss-ref.loss),'eval_loss_difference':eval_delta,
                 'eval_logits_maximum_difference':float((fallback.logits-eval_ref.logits).abs().max()),
                 'checkpointing':'nonreentrant, all four mixed GDN/GQA layers',
                 'passed':passes_gradient(grad) and abs(float(out.loss-ref.loss))<=0.01 and abs(eval_delta)<=1e-5}
        adapters.append(metrics)
        del model,out,fallback
        gc.collect();torch.cuda.empty_cache()
    return {'status':'passed' if all(c['passed'] for c in cases+adapters) else 'failed',
            'operator_cases':cases,'tiny_hybrid_adapter_cases':adapters,'torch':torch.__version__,
            'cuda':torch.version.cuda,'not_full_model_gradient_check':True,
            'long_convergence_or_quality_certified':False}
