#!/usr/bin/env python3
"""Publish a verified training-pack catalog and an overnight delivery report.

Read-only observer: no teacher requests, no release edits, no model training.
Only ledger-committed READY packs with checked physical files enter delivery.
"""
import argparse,fcntl,json,os,sqlite3,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p0_ready_catalog_v5 import catalog
from p0_training_bundle_v5 import check_documents
from small_agent.curation.release_gates_v5 import write
from small_agent.curation.supply_control import file_sha,digest

LABELS=dict(general_en='英文通用',general_zh='中文通用',technical='技术文档',code='代码',math='数学')

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);p=path.with_name('.'+path.name+'.tmp')
    p.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    with p.open('rb') as f:os.fsync(f.fileno())
    os.replace(p,path)

def load(path):
    return json.loads(path.read_text()) if path.exists() else {}

def render(report):
    current=report['verified_ready_train_body_tokens'];delta=report['new_ready_since_baseline']
    out=['# 生产清洗交付', '',time.strftime('更新时间：%Y-%m-%d %H:%M:%S %Z',time.localtime(report['verified_unix'])),
        '',f'物理校验通过的新管线 READY：**{current:,} train 正文 token**，相对本轮基线新增 **{delta:,}**。',
        f'可训练分片组：{report["verified_pack_groups"]}；继承旧库存单列，不在此重新认证质量。',
        '', '| 领域 | 新管线已校验 READY token | 本轮新增 |','|---|---:|---:|']
    for domain,label in LABELS.items():
        out.append(f'| {label} | {report["ready_by_domain"].get(domain,0):,} | {report["new_by_domain"].get(domain,0):,} |')
    out += ['',f'旧库存账本 train 正文：{report["inherited_legacy_train_body_tokens"]:,}（沿用原内部训练口径）。',
        f'旧库存账本 + 本次物理验证 READY：{report["inherited_plus_verified_ready_train_body_tokens"]:,}；候选池、EOS、验证和测试 token 不计入。',
        '',f'CPU 扫描：{report["cpu_files_complete"]}/{report["cpu_files_registered"]} 文件；规则阶段处理 {report["cpu_rows_processed"]:,} 行。',
        f'审核队列阶段：`{report["queue_stage"]}`；全局 API HOLD：{report["api_campaign_hold"]}。',
        f'预算上限：1000 元；最近教师状态：{report["latest_teacher_budget"]}（估算/预留，不是结算账单）。',
        '', '## 使用', '', '`catalog.json` 列出冻结的 manifest 和 READY 收据路径及 SHA-256。',
        '训练时固定其中具体的不可变 manifest，不追踪变化中的报告；这些分片仅允许内部训练。',
        '目前可用领域比例不代表已完成均衡 200 亿配方。来源校准/滚动抽检不等于逐篇双模型事实认证。',
        '', '## 隔离与阻塞']
    for h in report['cohort_holds']:out.append('- '+h['cohort']+'：'+h['failure_type']+'；'+h['reason'])
    if not report['cohort_holds']:out.append('- 当前未登记 cohort HOLD。')
    out.append('');return '\n'.join(out)

def deliver(cfg):
    root=Path(cfg['root']);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'delivery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c=catalog(Path(cfg['database']))
        if not c['packs']:raise ValueError('No physically verified READY packs; never substitute candidates')
        totals,domains,docs=check_documents(c['packs'])
        if totals.get('train',0)!=c['unique_train_body_tokens']:raise ValueError('Catalog/document counters differ')
        baseline=out/'BASELINE.json'
        if not baseline.exists():write(baseline,dict(created_unix=time.time(),ready_train_body_tokens=c['unique_train_body_tokens'],ready_by_domain=domains,training_started=False))
        previous=load(baseline)
        if c['unique_train_body_tokens']<previous['ready_train_body_tokens']:raise ValueError('READY stock shrank; inspect before publishing')
        d=sqlite3.connect('file:'+str(Path(cfg['database']).resolve())+'?mode=ro',uri=True)
        try:legacy=d.execute("SELECT COALESCE(SUM(d.tokens),0) FROM documents d JOIN legacy_imports l ON l.batch_id=d.batch_id WHERE d.split='train'").fetchone()[0]
        finally:d.close()
        d=sqlite3.connect('file:'+str(root/'stock.sqlite')+'?mode=ro',uri=True)
        try:files=d.execute('SELECT state,next_pos FROM files').fetchall()
        finally:d.close()
        holds=[]
        for path in sorted((root/'holds').glob('*')):
            h=load(path);holds.append(dict(cohort=h.get('cohort','unknown'),failure_type=h.get('failure_type','unknown'),reason=str(h.get('reason',''))[:250]))
        recent=list(root.rglob('ADAPTIVE_STATUS.json'));budget={}
        if recent:
            p=max(recent,key=lambda x:x.stat().st_mtime);s=load(p);a=s.get('ledger',{})
            budget={k:a.get(k) for k in ('cap_cny','estimated_or_reserved_cny','headroom_cny','cap_exceeded','not_a_settled_bill')};budget['as_of_unix']=p.stat().st_mtime
        q=load(root/'QUEUE_STATUS.json')
        frozen_catalog=out/'catalogs'/('catalog-'+digest(c)+'.json');write(frozen_catalog,c)
        report=dict(version=12,status='PHYSICALLY_VERIFIED_INTERNAL_TRAINING_PACKS_AVAILABLE',verified_unix=time.time(),
            verified_pack_groups=len(c['packs']),verified_ready_train_body_tokens=c['unique_train_body_tokens'],ready_by_domain=domains,
            new_ready_since_baseline=c['unique_train_body_tokens']-previous['ready_train_body_tokens'],
            new_by_domain={k:v-previous['ready_by_domain'].get(k,0) for k,v in domains.items()},
            inherited_legacy_train_body_tokens=legacy,inherited_plus_verified_ready_train_body_tokens=legacy+c['unique_train_body_tokens'],
            legacy_not_recertified=True,documents_all_splits=docs,body_tokens_all_splits=totals,
            cpu_files_complete=sum(x[0]=='complete' for x in files),cpu_files_registered=len(files),cpu_rows_processed=sum(x[1] for x in files),
            queue_stage=q.get('stage','unknown'),api_campaign_hold=(root/'API_HOLD.json').exists(),latest_teacher_budget=budget,cohort_holds=holds,
            catalog_path=str(frozen_catalog.resolve()),checker_sha256=file_sha(Path(__file__)),
            candidate_tokens_counted_as_ready=False,balanced_20b_training_mix_claimed=False,distribution_eligible=False,training_started=False)
        # Catalog is written only after all its inputs pass. Bind the exact bytes in the report.
        atomic(out/'catalog.json',c);report['catalog_sha256']=file_sha(frozen_catalog)
        atomic(out/'latest.json',report);atomic(out/'latest.zh.md',render(report))
        if (out/'ERROR.json').exists():(out/'ERROR.json').unlink()
        return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();cfg=load(a.config)
    try:print(json.dumps(deliver(cfg),ensure_ascii=False))
    except BlockingIOError:print('DELIVERY_OBSERVER_ALREADY_RUNNING')
    except Exception as exc:
        atomic(Path(cfg['output'])/'ERROR.json',dict(stage='DELIVERY_VERIFICATION_HOLD_NO_FRESH_APPROVAL',observed_unix=time.time(),reason=type(exc).__name__+':'+str(exc)[-500:],training_started=False));raise
