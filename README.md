# Small Agent Infra

面向小型混合语言模型的训练、数据清洗、评测与性能实验基础设施。

本仓库是经过脱敏的源码快照，不含原项目 Git 历史、个人目录、真实服务器地址、认证配置、数据正文、模型权重或运行回执。

## 组件

| 目录/入口 | 用途 |
| --- | --- |
| `src/small_agent/models/` | 混合 GDN/GQA 模型配置与构建 |
| `src/small_agent/data/` | 分片 token 数据、确定性混合和 checkpointable sampler |
| `src/small_agent/training/` | 单机/DDP 训练、Adam/checkpoint、调度器、健康与参考集统计 |
| `src/small_agent/curation/` | 清洗控制面、SQLite 台账、去重、质量与放行门禁 |
| `scripts/trainctl.py` | 环境、数据、模型和训练启动前检查 |
| `scripts/p0_curation_v3.py` 及 `p0_*` | 数据供应与分区、筛选、审计和 token 打包工具 |
| `scripts/training_eval_queue_v1.py` | checkpoint 评测队列、冻结评测输入与完成回执 |
| `scripts/eval_training_*.py` | 验证 NLL、选择题、生成、缓存一致性与隔离代码执行 |
| `scripts/benchmark_*.py` | micro-batch、通信、checkpointing、融合算子性能实验 |
| `scripts/training_quality_campaign_v2.py` | 恢复诊断与双种子原版/加速版长对照 |
| `scripts/finalize_training_quality_recovery_v6.py` | 仅 CPU 的评测兼容性修复与汇总，不训练模型 |
| `configs/`、`schemas/` | 配置示例、评分提示词、JSON Schema |
| `deploy/` | 脱敏部署模板；不是可直接执行的生产部署 |
| `tests/`、`experiments/` | 回归测试和早期对照原型 |

## 使用前

1. 在自己的 GPU/Linux 环境配置 Python 与 CUDA 依赖。训练、数据准备、数学原型依赖不同，已有 `requirements-*.txt` 是分组件依赖，不是完整训练环境锁文件。
2. 将模板中的 `/srv/small-agent`、`/opt/small-agent/venvs`、`/path/to/...` 改成自己的路径；`CPU_NODE`、`GPU_NODE` 是自定义 SSH 别名占位符，文档网段地址不是实际服务器。
3. 自行准备 tokenizer、数据与所需第三方依赖。权重、语料、上游许可审计和内部授权记录不随源码发布。
4. API 认证仅从环境或仓库外的私有配置加载。不要提交 `.env`、密钥、Cookie、运行生成的 manifest、日志或 checkpoint。
5. 不直接运行 `deploy/` 中的历史实验模板。先检查所有路径、预算、GPU 门禁及版本，再创建自己的新实验目录。

在配置好的环境查看接口：

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts"
python scripts/trainctl.py --help
python scripts/p0_curation_v3.py --help
python scripts/training_eval_queue_v1.py --help
```

模型实验曾使用 Torch 2.5.1、Transformers 4.57.1、Triton 3.1.0。GDN、卷积和融合路径还依赖匹配的 FLA、causal-conv1d、Liger；这些版本记录不是跨环境兼容性保证。本仓库没有打包或重新分发这些第三方依赖。

## 验收语义

- 保存成功、服务退出 0、测量全部完成、质量门禁通过是不同状态。
- 失败目录和回执保留，新修复使用新隔离版本；不自动重跑失败目录。
- 诊断的确定性卷积/固定 DDP 桶不能冒充原快速执行路径的逐位证书。
- 加速版必须接受数值、能力与生成对照；不能用降阈值或删除样本凑通过。
- 两版均为零分的数学/代码指标是地板效应，不能据此证明能力未退化。
- 不自动采用优化、不自动恢复正式训练，不停止无关进程。

详见 [训练质量测量说明](docs/TRAINING_QUALITY.zh.md) 和 [公开快照范围](docs/PUBLIC_EXPORT.zh.md)。

## 许可

本项目源码沿用 Apache-2.0，见 `LICENSE`。数据集、模型、依赖及外部基准的许可分别适用；本项目许可不授予这些内容的使用或再发布权。
