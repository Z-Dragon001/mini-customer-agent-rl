# GRPO 修复实验导航

本目录保存 2026-10-13 至 2026-10-14 的后续实验；原始实验在 `../retail-3b/`，不覆盖旧结论。新的任务成功率须以完整评测为准，不能由训练奖励或概率偏差推导。

已完成：40个测试任务×2个评测种子，SFT 13/80（16.25%），F 18/80（22.50%），差值+6.25个百分点；任务聚类95% bootstrap区间[-5.00,+17.50]个百分点。完整结果与限制见 `attention_report.md`。42项运行时测试及实际GPU前向/反向验证通过，F训练和冻结测试的412项证据检查通过；证据检查数不是单元测试数。

最终模型：`.art/service-agent-retail/models/qwen25-3b-repair-attention-selected/checkpoints/0008/`（项目根目录下）。由开发集选择，对应7次策略更新；整轮训练共21次更新，不要用末轮step22替换该模型。评测配置为 `experiments/retail-grpo-repair/attention-test42.json`，密钥仍从环境变量读取。原始基准已知，本轮仅一个训练种子。

- `protocol.md`：对照条件、奖励公式、预算、开发集选型及测试门槛。
- `journal.md`：诊断证据、运行故障、方案修订与实际过程。
- `status.json` / `events.jsonl`：当前阶段及完整调度历史。
- `report.md` / `results.csv`：从轨迹重建的评测表；每行检查覆盖数。
- `training_summary.json`：已完成组的采样量、实际策略更新和优化批次数。
- `training_metrics_summary.json`：真实优化批次的损失、梯度、KL及概率偏差范围，保留异常最大值。
- `numerical_comparison.json`：固定输入下的采样/训练概率偏差对比。
- `termination_diagnostics.json`：历史 train/dev 预算耗尽轨迹的离线数据库重放；仅验证 DB 分量。
- `progress_diagnostics.json`：train-only 的目标操作覆盖率重放，用于诊断全失败组是否丢失学习信号。
- `dev_task_changes.json`：各完整开发检查点相对 SFT 的逐任务变化。
- `selection.json`：原五组的开发选型记录；这些组最终仅作为诊断对照，不进入测试。
- `selection-attention.json` / `attention_report.md`：后续掩码修复组F的独立选型和结果，保留原五组结论。
- `attention_forwarding_diagnostic.json` / `attention_gpu_probe.json`：已安装模型入口覆盖掩码的复现，以及实际GPU分支隔离与反向验证。
- `paired_test-attention.csv` / `paired_summary-attention.json`：SFT与F的逐任务配对，以及全部测试结束后的汇总和区间。
- `audit-training.json` / `audit-attention.json`：原五组训练证据检查，以及F训练与完整测试的最终证据检查。
- `source*.json` / `source*.zip`：源码及起始权重的哈希与归档；运行中修复另见 journal。

项目根目录运行：

```bash
# 原五组仅完成开发对照并冻结；已完成阶段跳过，失败阶段按记录恢复
uv run --locked --extra retail python scripts/run_retail_grpo_repair.py --dev-only

# 不启动模型，只从现有轨迹重建报告
uv run --locked --extra retail python scripts/report_retail_grpo_repair.py

# 五个训练组都完成后，检查训练证据
uv run --locked --extra retail python scripts/audit_retail_grpo_repair.py --training-only

# 只查看精简进度，不加载模型
python scripts/status_retail_grpo_repair.py

# 原五组结束后，运行已登记的注意力修复组；不要与其他调度器同时启动
uv run --locked --extra retail python scripts/run_retail_attention_repair.py
uv run --locked --extra retail python scripts/report_retail_attention.py
# 训练结束、测试尚未完成时，可追加 --training-only
uv run --locked --extra retail python scripts/audit_retail_grpo_repair.py --attention

# 从完整开发评测绘制曲线；绘图库使用临时uv环境，不改训练依赖
uv run --locked --extra retail --with matplotlib==3.10.6 python scripts/plot_retail_grpo_repair.py
```

不要同时启动两个调度器。训练仅使用 GPU 0；API 密钥仍放在 `.env`，不进入归档。原始交互与损失在 `traces/retail/grpo-repair/`，适配器与优化器状态在 `.art/service-agent-retail/models/qwen25-3b-repair-*/`。

## 可以支持的结论边界

本实验区分“代码数值正确性”“训练行为”“开发选型”和“冻结测试”。log-prob 偏差降低只能证明数值一致性改善；过程奖励增加不等于官方任务成功率增加。重复调用是可计算代理指标，工具错误不等于全面业务违规。两个评测种子不是两个独立训练种子，40 个测试任务上的小幅变化仍可能受采样波动影响。
