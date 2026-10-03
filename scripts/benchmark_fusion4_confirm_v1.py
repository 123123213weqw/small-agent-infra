#!/usr/bin/env python3
"""Four-card matched-global-batch confirmation; never claims an eight-card repeat."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import benchmark_fusion8_v1 as family

PARENT=family.OUT
OUT=family.ROOT/'benchmarks/fusion4_v1_confirm1'
UNIT='p0-wave2-fusion4-confirm1-v1.service'
GPU_INDICES=[0,1,5,6]
family.OUT=OUT;family.UNIT=UNIT;family.engine.OUT=OUT
family.EXPECTED_VARIANTS=[('torch_full',False,False),('rmsnorm_only',True,False),
                          ('norm_swi',True,True),('torch_full_repeat',False,False)]
sha,read,write=family.sha,family.read,family.write
gpu_snapshot=family.gpu_snapshot


def scripts_identity():
    return {p.name:sha(p) for p in [Path(__file__),Path(family.__file__),Path(family.backend.__file__),
            Path(family.engine.__file__),Path(family.engine.common.__file__),
            OUT/'benchmark_fusion4_confirm_gpu_guard_v1.py',OUT/'run.sh']}


family.scripts_identity=scripts_identity


def validate_config(cfg):
    if cfg['version']!='fusion4_confirm_benchmark_v1':raise ValueError('four-card version required')
    if (cfg['world_size'],cfg['global_batch'],cfg['sequence_length'])!=(4,32,2048) or cfg['gpu_indices']!=GPU_INDICES:
        raise ValueError('fixed four GPUs and global32 required')
    if (cfg['warmup_steps'],cfg['measurement_steps'],cfg['profile_steps'],cfg['validation_samples'])!=(10,30,2,128):
        raise ValueError('fixed work budget required')
    if cfg['baseline_id']!='torch_full' or [(v['id'],v['fuse_norm'],v['fuse_swiglu']) for v in cfg['variants']]!=family.EXPECTED_VARIANTS:
        raise ValueError('predeclared groups required')
    for v in cfg['variants']:
        if (v['micro_batch'],v['accumulation'],v['communication'],v['bucket_mib'],v['checkpointing'],v['loss_backend'])!=(4,2,'bf16',25,'full','torch'):
            raise ValueError('same per-forward shape, fixed global batch and original loss required')
    if cfg['fixed_learning_rate']!=2e-5 or cfg['maximum_validation_loss_difference']!=0.01:
        raise ValueError('fixed learning rate and diagnostic tolerance required')
    if cfg['gradient_check']!={'maximum_relative_l2':0.02,'minimum_cosine':0.999,'maximum_loss_difference':0.01}:
        raise ValueError('fixed gradient tolerances required')
    if cfg.get('phase')!='confirmation_after_external_contention':raise ValueError('explicit confirmation phase required')


family.validate_config=validate_config
family.engine.validate_config=validate_config


def preflight():
    cfg=read(OUT/'config.json');validate_config(cfg)
    snap=gpu_snapshot();selected=[g for g in snap['gpus'] if g['index'] in GPU_INDICES]
    uuids={g['uuid'] for g in selected}
    busy=[p for p in snap['processes'] if p['uuid'] in uuids]
    if len(selected)!=4 or busy or any(g['memory_used_mib']>100 for g in selected):
        raise RuntimeError('selected four GPUs not free; other GPUs are never stopped')
    return snap


family.engine.gpu_is_free=preflight


def parent_identity():
    p=read(PARENT/'PREPARED.json');interrupted=read(PARENT/'INTERRUPTED.json')
    if interrupted['status']!='interrupted_by_external_gpu_contention' or interrupted['completed_groups']!=5:
        raise ValueError('parent campaign is not the expected interrupted trial')
    if p['identity']!=interrupted['source_identity']:raise ValueError('parent source identity mismatch')
    for script in [Path(family.__file__),Path(family.backend.__file__),Path(family.engine.__file__),Path(family.engine.common.__file__)]:
        if sha(script)!=p['fusion8_scripts'][script.name]:raise ValueError('numerical implementation changed across campaigns')
    return {'prepared_sha256':sha(PARENT/'PREPARED.json'),'interrupted_sha256':sha(PARENT/'INTERRUPTED.json'),
            'source_identity':p['identity'],'sample_plan_sha256':p['sample_plan_sha256'],
            'validation_plan_sha256':p['validation_plan_sha256']}


def prepare():
    parent=parent_identity()
    family.prepare()
    p=read(OUT/'PREPARED.json')
    if p['identity']!=parent['source_identity'] or p['sample_plan_sha256']!=parent['sample_plan_sha256'] or p['validation_plan_sha256']!=parent['validation_plan_sha256']:
        raise ValueError('checkpoint, Adam, train or validation sample plans differ across campaigns')
    p['confirmation_parent']=parent
    p['fusion8_design']='Four-card A, norm-only, joint, A-repeat; same checkpoint/global32/micro4, accumulation2'
    p['geometry_note']='Parent is eight ranks/accumulation1; this stage is four ranks/accumulation2. Not an eight-card repeat.'
    write(OUT/'PREPARED.json',p)


def summarize():
    p=family.verify_experiment();parent=parent_identity();source=family.engine.verify_source()
    if p['confirmation_parent']!=parent or source['identity']!=p['identity']:raise ValueError('source or parent changed')
    results=family.engine.collect_results(read(OUT/'config.json'))
    old=read(PARENT/'INTERRUPTED.json')['results'];old_by={r['variant']['id']:r for r in old}
    by={r['variant']['id']:r for r in results}
    control=family.combined_tps([by['torch_full'],by['torch_full_repeat']])
    old_control=family.combined_tps([old_by['torch_full'],old_by['torch_full_repeat']])
    drift=by['torch_full_repeat']['loss_tokens_per_second']/by['torch_full']['loss_tokens_per_second']-1 if control else None
    candidates={}
    for name in ['rmsnorm_only','norm_swi']:
        current,previous=by[name],old_by[name]
        ratio=current['loss_tokens_per_second']/control if control and family.engine.eligible(current) else None
        old_ratio=previous['loss_tokens_per_second']/old_control if old_control and family.engine.eligible(previous) else None
        candidates[name]={'confirmation_tokens_per_second':current.get('loss_tokens_per_second'),
                          'confirmation_speedup_ratio':ratio,'parent_speedup_ratio':old_ratio,
                          'gain_seen_in_both_geometries_over_2pct':ratio is not None and old_ratio is not None and min(ratio,old_ratio)>1.02 and drift is not None and abs(drift)<=0.1,
                          'long_convergence_or_quality_certified':False}
    scratch=OUT/'tmp/baseline_gradients.pt';scratch.unlink(missing_ok=True)
    report={'status':'fusion4_confirmation_completed_production_checkpoint_unchanged',
            'generated_at':datetime.now(timezone.utc).isoformat(),'source_identity':source['identity'],
            'confirmation_parent':parent,'results':results,'parent_results':old,
            'combined_confirmation_control_tokens_per_second':control,
            'combined_parent_control_tokens_per_second':old_control,
            'control_repeat_drift_fraction':drift,'candidates':candidates,
            'all_diagnostic_gates_passed':all(family.engine.eligible(r) for r in results),
            'gpu_snapshot_after':gpu_snapshot(),'temporary_full_gradient_deleted':not scratch.exists(),
            'production_config_changed':False,'formal_training_resumed':False,
            'capability_or_long_convergence_certified':False,'full_validation_rerun':False,
            'world_size':4,'global_batch':32,'micro_batch':4,'accumulation':2,
            'eight_card_repeat_completed':False}
    write(OUT/'COMPLETED.json',report)
    lines=['# RMSNorm／SwiGLU：四卡固定global32补测','',
           '八卡首轮保留五组有效结果，第六组因外部GPU任务被保护机制中断。本目录只用空闲的0/1/5/6四卡，不修改首轮归档。',
           '独立跑A、N、NS、A-repeat；相同checkpoint/Adam、43个训练batch及128条验证探针、原始CE、global32。',
           '每卡micro4不变，累积从1改2保持global32。四卡结果不冒称八卡重复确认，不直接对比两个阶段的绝对吞吐。',
           '生产模型、训练配方与环境未改；本结论只属于40步工程和数值诊断，不保证长期质量无损。','',
           '|阶段|组|token/s|峰值allocated GiB|梯度相对L2|40步后验证loss差|',
           '|---|---|---:|---:|---:|---:|']
    for stage,rows in [('八卡首轮',old),('四卡补测',results)]:
        for r in rows:
            if r['status']=='measured':
                lines.append(f"|{stage}|{r['variant']['id']}|{r['loss_tokens_per_second']:.0f}|{r['peak_allocated_gib']:.4f}|{r['gradient_check']['relative_l2']:.6f}|{r['validation_screen']['post40_step_loss_difference']:+.7f}|")
            else:lines.append(f"|{stage}|{r['variant']['id']}|失败，exit{r['worker_exit_code']}|—|—|—|")
    lines+=['',f'复跑两次基线合并吞吐：{control:.2f}。' if control else '复跑无有效基线。']
    for name,metrics in candidates.items():
        ratio=metrics['confirmation_speedup_ratio']
        lines.append(f'{name}：四卡相对该阶段基线 {ratio:.4f}，两种卡数下均见>2%增益：{metrics["gain_seen_in_both_geometries_over_2pct"]}。' if ratio else f'{name}：无有效四卡结论。')
    lines+=['SwiGLU单项首轮没有明显增益；优先考虑更简单的RMSNorm-only，而不是为联合版增加无明显收益的改动。',
            '两阶段控制分别计算，不能把旧阶段绝对吞吐当作新阶段无噪声基线。',
            '正式采用前仍需更长同数据对照与分领域验证，不直接恢复正式训练。']
    (OUT/'report.zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':report['status'],'control_tps':control,'candidates':candidates}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','kernel','prepare','worker','summarize']);parser.add_argument('--variant')
    args=parser.parse_args()
    if args.mode=='preflight':print(json.dumps(preflight()),flush=True)
    elif args.mode=='kernel':family.kernel_check()
    elif args.mode=='prepare':prepare()
    elif args.mode=='worker':
        if not args.variant:parser.error('worker requires --variant')
        validate_config(read(OUT/'config.json'));family.worker(args.variant)
    else:summarize()


if __name__=='__main__':main()
