#!/usr/bin/env python3
"""Separate RMSNorm, SwiGLU and combined eight-card trials, without production edits."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import benchmark_perf8_v1 as engine
import benchmark_fusion_backend_v1 as backend

ROOT,RUN,CHECKPOINT,CODE=engine.ROOT,engine.RUN,engine.CHECKPOINT,engine.CODE
OUT=ROOT/'benchmarks/fusion8_v1'
UNIT='p0-wave2-fusion8-v1.service'
WHEEL_SHA='77b041f97cca2afaaef3b71f2e4d2978bf12b75c75f1de64ca557acfc4e3dc55'
sha,read,write=engine.sha,engine.read,engine.write
gpu_snapshot=engine.gpu_snapshot
EXPECTED_VARIANTS=[('torch_full',False,False),('rmsnorm_only',True,False),
                   ('swiglu_only',False,True),('norm_swi',True,True),
                   ('torch_full_repeat',False,False),('norm_swi_repeat',True,True)]


def validate_config(cfg):
    if cfg['version']!='fusion8_benchmark_v1':raise ValueError('invalid version')
    if (cfg['world_size'],cfg['global_batch'],cfg['sequence_length'])!=(8,32,2048):
        raise ValueError('fixed eight ranks, global32 and context2048 required')
    if cfg['gpu_indices']!=list(range(8)):raise ValueError('all eight ordered GPUs required')
    if (cfg['warmup_steps'],cfg['measurement_steps'],cfg['profile_steps'],cfg['validation_samples'])!=(10,30,2,128):
        raise ValueError('fixed work and validation budget required')
    actual=[(v['id'],v['fuse_norm'],v['fuse_swiglu']) for v in cfg['variants']]
    if actual!=EXPECTED_VARIANTS or cfg['baseline_id']!='torch_full':
        raise ValueError('predeclared individual and joint groups with repeats required')
    for v in cfg['variants']:
        if (v['micro_batch'],v['accumulation'],v['communication'],v['bucket_mib'],v['checkpointing'])!=(4,1,'bf16',25,'full'):
            raise ValueError('only norm and SiLU-mul implementations may vary')
        if v['loss_backend']!='torch':raise ValueError('retain original unfused CE in every group')
    if cfg['fixed_learning_rate']!=2e-5 or cfg['maximum_validation_loss_difference']!=0.01:
        raise ValueError('fixed learning rate and validation tolerance required')
    if cfg['gradient_check']!={'maximum_relative_l2':0.02,'minimum_cosine':0.999,'maximum_loss_difference':0.01}:
        raise ValueError('fixed diagnostic gradient tolerances required')


engine.OUT=OUT
engine.validate_config=validate_config


def dependency_identity():
    wheels=sorted((OUT/'deps/wheels').glob('*.whl'))
    if len(wheels)!=1 or sha(wheels[0])!=WHEEL_SHA:raise ValueError('pinned wheel changed')
    vendor=OUT/'deps/vendor';h=hashlib.sha256()
    files=sorted(p for p in vendor.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    if not files:raise ValueError('missing isolated dependency')
    for p in files:h.update(str(p.relative_to(vendor)).encode()+b'\0'+sha(p).encode()+b'\n')
    return {'wheel_sha256':WHEEL_SHA,'vendor_tree_sha256':h.hexdigest(),'vendor_file_count':len(files),
            'installation':'--no-deps --no-compile --target, production environment untouched'}


def scripts_identity():
    return {p.name:sha(p) for p in [Path(__file__),Path(backend.__file__),Path(engine.__file__),
            Path(engine.common.__file__),OUT/'benchmark_fusion8_gpu_guard_v1.py',OUT/'run.sh']}


def verify_experiment():
    prepared=read(OUT/'PREPARED.json')
    if prepared['fusion8_scripts']!=scripts_identity() or prepared['fusion8_dependency']!=dependency_identity():
        raise ValueError('experiment code or dependency changed')
    for name,key in [('config.json','config_sha256'),('sample_plan.json','sample_plan_sha256'),
                     ('validation_plan.json','validation_plan_sha256'),('kernel_preflight.json','kernel_preflight_sha256')]:
        if sha(OUT/name)!=prepared[key]:raise ValueError('experiment input changed: '+name)
    return prepared


def preflight():return engine.gpu_is_free()


def kernel_check():
    preflight()
    try:report=backend.kernel_preflight()
    except Exception as error:
        write(OUT/'kernel_preflight.json',{'status':'failed','error':repr(error)});raise
    report['dependency']=dependency_identity()
    write(OUT/'kernel_preflight.json',report)
    print(json.dumps(report),flush=True)
    if report['status']!='passed':raise RuntimeError('operator/adapter gate failed; eight-card trials forbidden')


def prepare():
    if read(OUT/'kernel_preflight.json')['status']!='passed':raise ValueError('kernel gate not passed')
    engine.prepare()
    prepared=read(OUT/'PREPARED.json')
    prepared.update({'fusion8_scripts':scripts_identity(),'fusion8_dependency':dependency_identity(),
                     'kernel_preflight_sha256':sha(OUT/'kernel_preflight.json'),
                     'fusion8_design':'A, norm-only, swi-only, joint, A-repeat, joint-repeat; original CE/eval always'})
    write(OUT/'PREPARED.json',prepared)


def worker(variant_id):
    import small_agent.models as models
    verify_experiment()
    cfg=read(OUT/'config.json');validate_config(cfg)
    variant=engine.resolve_variant(cfg,variant_id)
    original_factory=models.build_model

    def factory(spec):
        model=original_factory(spec)
        info=backend.install_fusion(model,variant['fuse_norm'],variant['fuse_swiglu'])
        info['parameter_count']=sum(p.numel() for p in model.parameters())
        if info['parameter_count']!=1_005_213_696:raise ValueError('model size changed')
        if int(os.environ['RANK'])==0:write(OUT/variant_id/'adapter.json',info)
        return model

    def fail_closed_gradient(metrics,base_loss,loss,limits):
        good=engine.common.gradient_passed(metrics,base_loss,loss,limits)
        if not good:
            write(OUT/variant_id/'gradient_check.json',dict(metrics,passed=False,
                  note='Full-model gate failed; no optimizer steps or timing performed'))
            raise ValueError('full-model gradient/loss gate failed before updates')
        return True

    models.build_model=factory
    engine.gradient_passed=fail_closed_gradient
    try:engine.worker(variant_id)
    finally:models.build_model=original_factory


def combined_tps(results):
    if not results or not all(engine.eligible(r) for r in results):return None
    return len(results)/sum(1/r['loss_tokens_per_second'] for r in results)


def summarize():
    prepared=verify_experiment();source=engine.verify_source()
    if source['identity']!=prepared['identity']:raise ValueError('production source changed')
    cfg=read(OUT/'config.json');results=engine.collect_results(cfg)
    by_id={r['variant']['id']:r for r in results}
    controls=[by_id['torch_full'],by_id['torch_full_repeat']]
    joints=[by_id['norm_swi'],by_id['norm_swi_repeat']]
    base_tps,joint_tps=combined_tps(controls),combined_tps(joints)
    def drift(rows):
        return rows[1]['loss_tokens_per_second']/rows[0]['loss_tokens_per_second']-1 if combined_tps(rows) else None
    bd,jd=drift(controls),drift(joints)
    noise=any(d is None or abs(d)>0.1 for d in [bd,jd])
    ratio=joint_tps/base_tps if base_tps and joint_tps else None
    scratch=OUT/'tmp/baseline_gradients.pt';scratch.unlink(missing_ok=True)
    individual={name:by_id[name]['loss_tokens_per_second']/base_tps if base_tps and engine.eligible(by_id[name]) else None
                for name in ['rmsnorm_only','swiglu_only']}
    report={'status':'fusion8_trial_completed_production_checkpoint_unchanged',
            'generated_at':datetime.now(timezone.utc).isoformat(),'source_identity':source['identity'],
            'results':results,'combined_control_tokens_per_second':base_tps,
            'combined_joint_tokens_per_second':joint_tps,'joint_speedup_ratio':ratio,
            'individual_speedup_ratios':individual,'individual_variants_repeated':False,
            'control_repeat_drift_fraction':bd,'joint_repeat_drift_fraction':jd,
            'shared_host_noise_warning':noise,'all_diagnostic_gates_passed':all(engine.eligible(r) for r in results),
            'joint_speedup_verified':ratio is not None and ratio>1.02 and not noise,
            'temporary_full_gradient_deleted':not scratch.exists(),
            'dependency':prepared['fusion8_dependency'],'gpu_snapshot_after':gpu_snapshot(),
            'formal_training_resumed':False,'production_config_changed':False,
            'capability_or_long_convergence_certified':False,'full_validation_rerun':False}
    write(OUT/'COMPLETED.json',report)
    lines=['# RMSNorm／SwiGLU融合：八卡拆分与联合实验','',
           '八张L40，micro4×acc1，全局32，2048上下文；BF16通信，25MiB桶，32层完整重计算。',
           'FP32主参数/Adam、BF16计算；相同checkpoint/Adam/训练样本/学习率2e-5。',
           '仅训练期局部算子变化；每组都用原始输出层/CE，验证与生成都恢复原始模块forward。',
           'RMSNorm精确保留1+weight及FP32计算后cast，in_place=False；不改GDN gated norm。',
           '各预热10步、计时30步、额外2步profiler，训练前后128条固定heldout探针。','',
           '|组|状态/退出码|loss-token/s|峰值分配GiB|全参数梯度相对L2|40步后验证loss差|',
           '|---|---|---:|---:|---:|---:|']
    for r in results:
        if r['status']=='measured':
            lines.append(f"|{r['variant']['id']}|{r['status']}/{r['worker_exit_code']}|{r['loss_tokens_per_second']:.0f}|{r['peak_allocated_gib']:.2f}|{r['gradient_check']['relative_l2']:.6f}|{r['validation_screen']['post40_step_loss_difference']:+.6f}|")
        else:lines.append(f"|{r['variant']['id']}|{r['status']}/{r['worker_exit_code']}|—|—|—|—|")
    lines+=['',f'联合两次重复 / 两次基线合并吞吐：{ratio:.4f}。' if ratio else '联合组没有完整有效重复对照。',
            f'联合组提速门槛通过：{report["joint_speedup_verified"]}；完整数值门禁通过：{report["all_diagnostic_gates_passed"]}。',
            '单项组只测一次，归因属于诊断线索；联合增益不能简单相加或直接归给某个算子。',
            '短程梯度/探针不证明长期收敛或训练质量完全一致，不直接切进正式训练。',
            '正式模型/Adam/source再次hash核对，GPU worker退出，临时完整梯度删除。',
            'Profiler只计非annotation CUDA活动，活动时长之和不是墙钟占比。']
    (OUT/'report.zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ['status','combined_control_tokens_per_second',
                     'combined_joint_tokens_per_second','joint_speedup_ratio','all_diagnostic_gates_passed']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['preflight','kernel','prepare','worker','summarize']);p.add_argument('--variant')
    args=p.parse_args()
    if args.mode=='preflight':print(json.dumps(preflight()),flush=True)
    elif args.mode=='kernel':kernel_check()
    elif args.mode=='prepare':prepare()
    elif args.mode=='worker':
        if not args.variant:p.error('worker requires --variant')
        worker(args.variant)
    else:summarize()


if __name__=='__main__':main()
