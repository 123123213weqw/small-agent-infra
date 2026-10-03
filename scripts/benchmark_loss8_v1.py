#!/usr/bin/env python3
"""Eight-card fused-loss ABAB, using an immutable copy of the verified perf8 engine."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import benchmark_perf8_v1 as engine
import benchmark_loss_backend_v1 as backend

ROOT, RUN, CHECKPOINT, CODE = engine.ROOT, engine.RUN, engine.CHECKPOINT, engine.CODE
OUT = ROOT/'benchmarks/loss8_v1'
UNIT = 'p0-wave2-loss8-v1.service'
WHEEL_SHA = '77b041f97cca2afaaef3b71f2e4d2978bf12b75c75f1de64ca557acfc4e3dc55'
sha, read, write = engine.sha, engine.read, engine.write
gpu_snapshot = engine.gpu_snapshot


def validate_config(cfg):
    if cfg['version']!='loss8_benchmark_v1': raise ValueError('invalid version')
    if (cfg['world_size'],cfg['global_batch'],cfg['sequence_length'])!=(8,32,2048):
        raise ValueError('fixed ranks, global batch and context required')
    if cfg['gpu_indices']!=list(range(8)): raise ValueError('all eight GPUs required')
    if (cfg['warmup_steps'],cfg['measurement_steps'],cfg['profile_steps'],cfg['validation_samples'])!=(10,30,2,128):
        raise ValueError('fixed work and probe budget required')
    if cfg['baseline_id']!='torch_full' or [v['id'] for v in cfg['variants']]!=[
            'torch_full','liger_flce','torch_full_repeat','liger_flce_repeat']:
        raise ValueError('predeclared ABAB repeats required, no fastest-run selection')
    for index,v in enumerate(cfg['variants']):
        if (v['micro_batch'],v['accumulation'],v['communication'],v['bucket_mib'],v['checkpointing'])!=(4,1,'bf16',25,'full'):
            raise ValueError('loss backend must be the only changed optimization')
        if v['loss_backend']!=('torch' if index%2==0 else 'liger'):
            raise ValueError('invalid backend allocation')
    if cfg['fixed_learning_rate']!=2e-5 or cfg['maximum_validation_loss_difference']!=0.01:
        raise ValueError('fixed learning rate and diagnostic tolerance required')
    if cfg['gradient_check']!={'maximum_relative_l2':0.02,'minimum_cosine':0.999,'maximum_loss_difference':0.01}:
        raise ValueError('fixed gradient tolerances required')


engine.OUT = OUT
engine.validate_config = validate_config


def dependency_identity():
    wheels = sorted((OUT/'deps/wheels').glob('*.whl'))
    if len(wheels)!=1 or sha(wheels[0])!=WHEEL_SHA: raise ValueError('pinned wheel mismatch')
    h = hashlib.sha256()
    vendor = OUT/'deps/vendor'
    files = sorted(p for p in vendor.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    if not files: raise ValueError('missing isolated dependency')
    for p in files: h.update(str(p.relative_to(vendor)).encode()+b'\0'+sha(p).encode()+b'\n')
    return {'wheel_sha256':WHEEL_SHA,'vendor_tree_sha256':h.hexdigest(),'vendor_file_count':len(files),
            'installation':'--no-deps --no-compile --target, production environment untouched'}


def scripts_identity():
    return {p.name:sha(p) for p in [Path(__file__),Path(backend.__file__),Path(engine.__file__),
            Path(engine.common.__file__),OUT/'benchmark_loss8_gpu_guard_v1.py',OUT/'run.sh']}


def verify_experiment():
    prepared = read(OUT/'PREPARED.json')
    if prepared['loss8_scripts']!=scripts_identity() or prepared['loss8_dependency']!=dependency_identity():
        raise ValueError('experiment script or dependency changed')
    for name,key in [('config.json','config_sha256'),('sample_plan.json','sample_plan_sha256'),
                     ('validation_plan.json','validation_plan_sha256')]:
        if sha(OUT/name)!=prepared[key]: raise ValueError('experiment input changed: '+name)
    return prepared


def preflight():
    return engine.gpu_is_free()


def kernel_check():
    preflight()
    report = backend.kernel_preflight()
    report['dependency'] = dependency_identity()
    write(OUT/'kernel_preflight.json',report)
    print(json.dumps(report),flush=True)
    if report['status']!='passed': raise RuntimeError('operator or adapter check failed; no eight-card trial')


def prepare():
    if read(OUT/'kernel_preflight.json')['status']!='passed': raise ValueError('kernel gate not passed')
    engine.prepare()
    prepared = read(OUT/'PREPARED.json')
    prepared.update({'loss8_scripts':scripts_identity(),'loss8_dependency':dependency_identity(),
                     'kernel_preflight_sha256':sha(OUT/'kernel_preflight.json'),
                     'loss8_design':'ABAB, no winner selection; evaluation CE is always original HF'})
    write(OUT/'PREPARED.json',prepared)


def worker(variant_id):
    import small_agent.models as models
    verify_experiment()
    cfg = read(OUT/'config.json'); validate_config(cfg)
    variant = engine.resolve_variant(cfg,variant_id)
    if sha(OUT/'kernel_preflight.json')!=read(OUT/'PREPARED.json')['kernel_preflight_sha256']:
        raise ValueError('kernel gate changed')
    original_factory = models.build_model

    def factory(spec):
        model = original_factory(spec)
        info = backend.install_fused_loss(model) if variant['loss_backend']=='liger' else {
                 'backend':'original_transformers','other_modules_patched':False}
        info['parameter_count'] = sum(p.numel() for p in model.parameters())
        if info['parameter_count']!=1_005_213_696: raise ValueError('model size changed')
        if int(os.environ['RANK'])==0: write(OUT/variant_id/'adapter.json',info)
        return model

    models.build_model = factory
    try: engine.worker(variant_id)
    finally: models.build_model = original_factory


def combined_tps(results):
    if not results or not all(engine.eligible(r) for r in results): return None
    return len(results)/sum(1/r['loss_tokens_per_second'] for r in results)


def summarize():
    prepared = verify_experiment()
    source = engine.verify_source()
    if source['identity']!=prepared['identity']: raise ValueError('production source changed')
    results = engine.collect_results(read(OUT/'config.json'))
    controls, fused = results[::2],results[1::2]
    base_tps, fused_tps = combined_tps(controls),combined_tps(fused)
    def drift(rows):
        return rows[1]['loss_tokens_per_second']/rows[0]['loss_tokens_per_second']-1 if combined_tps(rows) else None
    base_drift, fused_drift = drift(controls),drift(fused)
    noise = any(d is None or abs(d)>0.1 for d in [base_drift,fused_drift])
    ratio = fused_tps/base_tps if base_tps and fused_tps else None
    scratch = OUT/'tmp/baseline_gradients.pt'
    scratch.unlink(missing_ok=True)
    snap = gpu_snapshot()  # Other work can start after our workers exit; do not stop it.
    report = {'status':'loss8_abab_completed_production_checkpoint_unchanged',
              'generated_at':datetime.now(timezone.utc).isoformat(),'source_identity':source['identity'],
              'results':results,'combined_control_tokens_per_second':base_tps,
              'combined_fused_tokens_per_second':fused_tps,'fused_speedup_ratio':ratio,
              'control_repeat_drift_fraction':base_drift,'fused_repeat_drift_fraction':fused_drift,
              'shared_host_noise_warning':noise,'all_diagnostic_gates_passed':all(engine.eligible(r) for r in results),
              'speedup_verified':ratio is not None and ratio>1.02 and not noise,
              'temporary_full_gradient_deleted':not scratch.exists(),
              'dependency':prepared['loss8_dependency'],'gpu_snapshot_after':snap,
              'formal_training_resumed':False,'production_config_changed':False,
              'capability_or_long_convergence_certified':False,'full_validation_rerun':False}
    write(OUT/'COMPLETED.json',report)
    lines = ['# 融合输出层＋交叉熵：八卡 ABAB 实验','',
             '八张 L40，每卡 micro4、累积1、全局32，2048上下文；BF16通信、桶25MiB、32层完整重计算。',
             'FP32主参数和Adam，BF16计算；各组读取相同最终checkpoint与Adam，固定LR=2e-5。',
             '仅变训练期输出层/CE；原始HF前向用于全部验证和生成，不替换GDN、GQA、RMSNorm或MLP。',
             'ABAB预先固定，两组各复跑一次；各预热10步、计时30步，随后额外2步profiler。',
             '全参数梯度对比及128条heldout探针不计入吞吐；这是数值/工程诊断，不是能力测评。','',
             '|组|状态/退出码|loss-token/s|峰值分配GiB|梯度相对L2|40步后验证loss差|',
             '|---|---|---:|---:|---:|---:|']
    for r in results:
        if r['status']=='measured':
            lines.append(f"|{r['variant']['id']}|{r['status']}/{r['worker_exit_code']}|{r['loss_tokens_per_second']:.0f}|{r['peak_allocated_gib']:.2f}|{r['gradient_check']['relative_l2']:.6f}|{r['validation_screen']['post40_step_loss_difference']:+.6f}|")
        else: lines.append(f"|{r['variant']['id']}|{r['status']}/{r['worker_exit_code']}|—|—|—|—|")
    lines += ['',f'合并吞吐比（等量工作/总耗时）：{ratio:.4f}。' if ratio else '未获得有效合并吞吐比。',
              f'实测提速结论成立：{report["speedup_verified"]}（要求两次重复无>10%漂移且增益>2%）。',
              '峰值只取训练计时窗口，后续原始CE验证的大logits不会计入训练显存。',
              '临时完整梯度已删除；原始checkpoint/source全部hash再次校验，正式训练和环境未修改。',
              'Profiler kernel时长合计不是墙钟占比；Liger在前向预计算梯度，前向/反向耗时不能直接互比。']
    (OUT/'report.zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ['status','combined_control_tokens_per_second',
                     'combined_fused_tokens_per_second','fused_speedup_ratio','all_diagnostic_gates_passed']}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','kernel','prepare','worker','summarize'])
    parser.add_argument('--variant')
    args = parser.parse_args()
    if args.mode=='preflight': print(json.dumps(preflight()),flush=True)
    elif args.mode=='kernel': kernel_check()
    elif args.mode=='prepare': prepare()
    elif args.mode=='worker':
        if not args.variant: parser.error('worker requires --variant')
        worker(args.variant)
    else: summarize()


if __name__=='__main__': main()
