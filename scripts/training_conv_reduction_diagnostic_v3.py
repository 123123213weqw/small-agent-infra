#!/usr/bin/env python3
"""Isolated deterministic dweight oracle; NEVER installed in formal training.

Keep the installed CUDA convolution's forward and dx unchanged. Replace only
its dweight reduction by a fixed PyTorch sum of the analytic derivative. This
is a numerical reference for checkpoint tests, not a claim of bitwise native
kernel equivalence, and is intentionally limited to our no-cache training call.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from training_eval_common_v1 import write_json


def weight_gradient(x,weight,bias,dout,activation):
    if activation not in (False,True): raise ValueError('boolean SiLU flag required')
    if x.ndim!=3 or weight.ndim!=2 or dout.shape!=x.shape or x.shape[1]!=weight.shape[0]:
        raise ValueError('unsupported causal convolution dimensions')
    dtype=torch.float64 if x.dtype==weight.dtype==torch.float64 else torch.float32
    batch,channels,length=x.shape; width=weight.shape[1]
    padded=F.pad(x.to(dtype),(width-1,0))
    pre=torch.zeros((batch,channels,length),dtype=dtype,device=x.device)
    if bias is not None:pre+=bias.to(dtype)[None,:,None]
    for k in range(width):pre+=padded[:,:,k:k+length]*weight[:,k].to(dtype)[None,:,None]
    grad=dout.to(dtype)
    if activation:
        sigma=torch.sigmoid(pre)
        grad=grad*sigma*(1+pre*(1-sigma))
    return torch.stack([(grad*padded[:,:,k:k+length]).sum(dim=(0,2)) for k in range(width)],dim=1).to(weight.dtype)


class _SavedTensorContext:
    """Backward-local view: unpack the real autograd context exactly once.

    Non-reentrant activation checkpointing consumes saved-tensor holders on
    access. Passing the real ctx to native backward after reading it ourselves
    would unpack twice. This view shares the SAME tensor objects, delegates all
    other attributes, and is discarded at the end of this backward call. It
    must not cache tensors on the real ctx or bypass saved-tensor hooks.
    """
    def __init__(self, ctx, saved):
        self._ctx = ctx
        self.saved_tensors = saved

    def __getattr__(self, name):
        return getattr(self._ctx, name)


def backward_with_reference(native, ctx, dout, *args):
    saved = ctx.saved_tensors
    x, weight, bias, seq_idx, initial_states = saved
    if seq_idx is not None or initial_states is not None or ctx.return_final_states or args:
        raise ValueError('diagnostic supports no-cache training convolution only')
    results = list(native(_SavedTensorContext(ctx, saved), dout, *args))
    results[1] = weight_gradient(x, weight, bias, dout, ctx.activation)
    return tuple(results)


def install():
    from causal_conv1d.causal_conv1d_interface import CausalConv1dFn
    if getattr(CausalConv1dFn,'_quality_dweight_reference',False):raise RuntimeError('diagnostic installed twice')
    native=CausalConv1dFn.backward
    def backward(ctx,dout,*args):
        return backward_with_reference(native,ctx,dout,*args)
    CausalConv1dFn.backward=staticmethod(backward)
    CausalConv1dFn._quality_dweight_reference=True
    return {'scope':'isolated_checkpoint_diagnostic_only','forward':'unchanged_installed_CUDA',
        'dx':'unchanged_installed_CUDA','dweight':'FP32_fixed_analytic_sum',
        'saved_tensor_access':'single_unpack_shared_backward_local_tuple',
        'original_backend_bitwise_equivalence_certified':False,
        'helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def preflight(root):
    import importlib.metadata
    from causal_conv1d import causal_conv1d_fn
    from torch.utils.checkpoint import checkpoint
    torch.cuda.set_device(0);torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261003);torch.cuda.manual_seed_all(20261003)
    rows=[]
    # Actual Qwen GDN convolution geometry, both original and candidate batch.
    tensors=[]
    for batch in (1,4):
        x=(torch.randn(batch,2048,4608,device='cuda')*.5).transpose(1,2).to(torch.bfloat16).requires_grad_()
        weight=(torch.randn(4608,4,device='cuda')*.1).requires_grad_()
        dout=(torch.randn_like(x.float())*.01).to(x.dtype)
        y=causal_conv1d_fn(x,weight,activation='silu')
        dx,dw=torch.autograd.grad(y,(x,weight),dout)
        oracle=weight_gradient(x,weight,None,dout,True)
        delta=(dw-oracle).double()
        rows.append({'batch':batch,'native_oracle_max_abs':float(delta.abs().max()),
            'native_oracle_relative_l2':float(delta.norm()/dw.double().norm()),
            'native_oracle_close':bool(torch.allclose(dw,oracle,atol=5e-6,rtol=2e-5))})
        tensors.append((x,weight,dout,y.detach(),dx.detach()))
    policy=install()
    for row,(x,weight,dout,reference,native_dx) in zip(rows,tensors):
        # Require unchanged CUDA output AND exactly repeatable dweight.
        gradients=[];outputs_exact=[];dx_exact=[]
        for _ in range(3):
            y=causal_conv1d_fn(x,weight,activation='silu')
            outputs_exact.append(torch.equal(y,reference))
            dx,dw=torch.autograd.grad(y,(x,weight),dout)
            dx_exact.append(torch.equal(dx,native_dx));gradients.append(dw)
        row['forward_exact']=all(outputs_exact);row['dx_exact']=all(dx_exact)
        row['dweight_repeat_exact']=all(torch.equal(gradients[0],g) for g in gradients[1:])
        # The original preflight missed this production execution path. Both
        # input and weight require gradients, with the actual non-reentrant
        # checkpoint flag, dtype, stride and channel/sequence geometry.
        checkpoint_gradients=[];checkpoint_outputs=[];checkpoint_dx=[]
        for _ in range(2):
            y=checkpoint(lambda a,b:causal_conv1d_fn(a,b,activation='silu'),x,weight,use_reentrant=False)
            dx,dw=torch.autograd.grad(y,(x,weight),dout)
            checkpoint_outputs.append(torch.equal(y,reference))
            checkpoint_dx.append(torch.equal(dx,native_dx));checkpoint_gradients.append(dw)
        row['nonreentrant_checkpoint_forward_exact']=all(checkpoint_outputs)
        row['nonreentrant_checkpoint_dx_exact']=all(checkpoint_dx)
        row['nonreentrant_checkpoint_dweight_exact']=all(torch.equal(gradients[0],g) for g in checkpoint_gradients)
    required=('native_oracle_close','forward_exact','dx_exact','dweight_repeat_exact',
        'nonreentrant_checkpoint_forward_exact','nonreentrant_checkpoint_dx_exact','nonreentrant_checkpoint_dweight_exact')
    passed=all(all(r[key] for key in required) for r in rows)
    report={'passed':passed,'rows':rows,'policy':policy,
        'causal_conv1d_version':importlib.metadata.version('causal-conv1d'),
        'formal_training_changed':False,'automatic_adoption_allowed':False}
    write_json(root/'reports/conv_preflight.json',report)
    print(json.dumps(report),flush=True)
    if not passed:raise RuntimeError('convolution diagnostic numeric preflight failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    preflight(p.parse_args().root)
