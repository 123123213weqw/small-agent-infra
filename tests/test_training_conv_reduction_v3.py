import sys
from pathlib import Path
import unittest
import tempfile
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import torch
    from training_conv_reduction_diagnostic_v3 import weight_gradient, backward_with_reference
except ImportError:torch=None


@unittest.skipIf(torch is None,'remote torch environment required')
class ConvTests(unittest.TestCase):
    def test_fixed_ddp_plan_is_diagnostic_only_and_fails_closed_without_reference_scope(self):
        from training_quality_campaign_v2 import resume_ddp_options
        cfg={'resume_fixed_ddp_bucket_plan':True,'resume_conv_reduction':'deterministic_weight_reference'}
        self.assertEqual(resume_ddp_options(cfg,True),{'find_unused_parameters':True,'static_graph':False})
        self.assertEqual(resume_ddp_options(cfg,False),{})
        self.assertEqual(resume_ddp_options({},True),{})
        cfg['resume_conv_reduction']='native'
        with self.assertRaises(ValueError):resume_ddp_options(cfg,True)

    def test_real_context_saved_tensors_is_read_once_and_native_attributes_delegate(self):
        from types import SimpleNamespace
        x=torch.randn(1,2,9);weight=torch.randn(2,4);dout=torch.randn_like(x)
        saved=(x,weight,None,None,None)
        class OneReadContext:
            activation=True
            return_final_states=False
            return_dinitial_states=False
            reads=0
            @property
            def saved_tensors(self):
                self.reads+=1
                if self.reads>1:raise RuntimeError('second unpack forbidden')
                return saved
        ctx=OneReadContext();sentinels=[object() for _ in range(8)]
        def native(view,gradient):
            self.assertIs(view.saved_tensors,saved)
            self.assertIs(view.saved_tensors[0],x)
            self.assertIs(gradient,dout)
            self.assertFalse(view.return_dinitial_states)
            return tuple(sentinels)
        result=backward_with_reference(native,ctx,dout)
        self.assertEqual(ctx.reads,1)
        self.assertTrue(torch.equal(result[1],weight_gradient(x,weight,None,dout,True)))
        for i in (0,2,3,4,5,6,7):self.assertIs(result[i],sentinels[i])
        self.assertNotIn('saved_tensors',ctx.__dict__)

    def test_nonreentrant_checkpoint_reproduces_old_double_unpack_and_accepts_fixed_wrapper(self):
        from torch.utils.checkpoint import checkpoint, CheckpointError
        import torch.nn.functional as F
        def native(ctx,dout):
            x,weight,bias,_,_=ctx.saved_tensors
            pre=F.conv1d(F.pad(x,(weight.shape[1]-1,0)),weight[:,None,:],groups=x.shape[1])
            sigma=torch.sigmoid(pre)
            dy=dout*sigma*(1+pre*(1-sigma))
            dx=F.conv_transpose1d(dy,weight[:,None,:],groups=x.shape[1])[:,:,weight.shape[1]-1:]
            return dx,weight_gradient(x,weight,bias,dout,True)
        class Conv(torch.autograd.Function):
            @staticmethod
            def forward(ctx,x,weight,legacy):
                ctx.save_for_backward(x,weight,None,None,None)
                ctx.activation=True;ctx.return_final_states=False;ctx.legacy=legacy
                return F.silu(F.conv1d(F.pad(x,(weight.shape[1]-1,0)),weight[:,None,:],groups=x.shape[1]))
            @staticmethod
            def backward(ctx,dout):
                if ctx.legacy:
                    _=ctx.saved_tensors
                    dx,dw=native(ctx,dout)
                else:dx,dw=backward_with_reference(native,ctx,dout)
                return dx,dw,None
        torch.manual_seed(17)
        x=torch.randn(2,3,21,dtype=torch.float64,requires_grad=True)
        weight=torch.randn(3,4,dtype=torch.float64,requires_grad=True)
        dout=torch.randn_like(x)
        failed=checkpoint(lambda a,b:Conv.apply(a,b,True),x,weight,use_reentrant=False)
        with self.assertRaisesRegex(CheckpointError,'already unpacked once'):
            torch.autograd.grad(failed,(x,weight),dout)
        expected=Conv.apply(x,weight,False)
        dx,dw=torch.autograd.grad(expected,(x,weight),dout)
        for _ in range(2):
            actual=checkpoint(lambda a,b:Conv.apply(a,b,False),x,weight,use_reentrant=False)
            actual_dx,actual_dw=torch.autograd.grad(actual,(x,weight),dout)
            self.assertTrue(torch.equal(actual,expected))
            self.assertTrue(torch.equal(actual_dx,dx));self.assertTrue(torch.equal(actual_dw,dw))

    def test_unsupported_cache_paths_fail_before_native_backward(self):
        from types import SimpleNamespace
        x=torch.ones(1,2,9);weight=torch.ones(2,4);dout=torch.ones_like(x)
        for seq,initial,final,args in ((torch.zeros(1,9),None,False,()),(None,torch.zeros(1,2,3),False,()),
                                      (None,None,True,()),(None,None,False,(torch.ones(1),))):
            ctx=SimpleNamespace(saved_tensors=(x,weight,None,seq,initial),return_final_states=final,activation=True)
            native=mock.Mock()
            with self.assertRaises(ValueError):backward_with_reference(native,ctx,dout,*args)
            native.assert_not_called()

    def test_analytic_dw_matches_double_autograd_for_silu_and_linear(self):
        for batch in (1,4):
            for width in (2,4):
                for activation in (False,True):
                    torch.manual_seed(3)
                    x=torch.randn(batch,7,31,dtype=torch.float64)
                    weight=torch.randn(7,width,dtype=torch.float64,requires_grad=True)
                    bias=torch.randn(7,dtype=torch.float64)
                    y=torch.nn.functional.conv1d(x,weight[:,None,:],bias,padding=width-1,groups=7)[:,:,:31]
                    if activation:y=torch.nn.functional.silu(y)
                    dout=torch.randn_like(y)
                    expected=torch.autograd.grad(y,weight,dout)[0]
                    actual=weight_gradient(x,weight,bias,dout,activation)
                    self.assertTrue(torch.allclose(actual,expected,atol=1e-12,rtol=1e-12))

    def test_float_repeat_and_weight_dtype_preserved(self):
        x=torch.randn(1,4,17).bfloat16();weight=torch.randn(4,4);dout=torch.randn_like(x)
        a=weight_gradient(x,weight,None,dout,True);b=weight_gradient(x,weight,None,dout,True)
        self.assertTrue(torch.equal(a,b));self.assertEqual(a.dtype,weight.dtype)

    def test_future_inputs_do_not_influence_past_weight_derivative(self):
        x=torch.randn(1,2,12);weight=torch.randn(2,4);dout=torch.zeros_like(x);dout[:,:,:5]=1
        a=weight_gradient(x,weight,None,dout,True)
        x[:,:,5:]=torch.randn_like(x[:,:,5:])*100
        self.assertTrue(torch.equal(a,weight_gradient(x,weight,None,dout,True)))

    def test_invalid_shapes_fail_closed(self):
        with self.assertRaises(ValueError):weight_gradient(torch.ones(1,3,7),torch.ones(4,4),None,torch.ones(1,3,7),True)

    def test_state_comparison_rejects_dtype_and_any_tensor_value_drift(self):
        from training_quality_campaign_v2 import same_state
        left={'state':{0:{'exp_avg':torch.zeros(10000)}}};right={'state':{0:{'exp_avg':torch.zeros(10000)}}}
        self.assertTrue(same_state(left,right));right['state'][0]['exp_avg'][-1]=1
        self.assertFalse(same_state(left,right))
        self.assertFalse(same_state(torch.zeros(2),torch.zeros(2,dtype=torch.bfloat16)))

    def test_only_rank_zero_creates_shared_case_directory(self):
        from training_quality_campaign_v2 import create_case_directory
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'case'
            with mock.patch('torch.distributed.broadcast_object_list'):
                create_case_directory(out,0)
                create_case_directory(out,1)
                self.assertTrue(out.is_dir())
                with self.assertRaises(RuntimeError):create_case_directory(out,0)


if __name__=='__main__':unittest.main()
