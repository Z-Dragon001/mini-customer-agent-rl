# GRPO 改进实验结果

每行覆盖必须达到 dev 15 或 test 40 才能作为该次完整评测。过程奖励不替代官方成功率。

| 实验 | 划分/种子/轮次 | 成功 | 成功率 | 工具错误任务比例 | 平均调用 | 重复调用 | 预算耗尽 |
|---|---|---:|---:|---:|---:|---:|---:|
| sft-dev42 | dev/42/-1 | 3/15 | 20.0% | 60.0% | 9.87 | 18 | 9 |
| kl | dev/42/8 | 3/15 | 20.0% | 60.0% | 11.87 | 42 | 9 |
| kl | dev/42/16 | 2/15 | 13.3% | 46.7% | 9.67 | 24 | 9 |
| kl | dev/42/24 | 2/15 | 13.3% | 46.7% | 9.27 | 11 | 6 |
| outcome | dev/42/8 | 2/15 | 13.3% | 80.0% | 11.33 | 22 | 11 |
| outcome | dev/42/16 | 3/15 | 20.0% | 60.0% | 10.07 | 17 | 7 |
| outcome | dev/42/24 | 1/15 | 6.7% | 46.7% | 10.47 | 32 | 11 |
| consistent | dev/42/8 | 3/15 | 20.0% | 40.0% | 10.73 | 22 | 9 |
| consistent | dev/42/16 | 1/15 | 6.7% | 46.7% | 11.60 | 17 | 9 |
| consistent | dev/42/24 | 2/15 | 13.3% | 46.7% | 9.73 | 20 | 5 |
| shaped | dev/42/8 | 1/15 | 6.7% | 33.3% | 11.47 | 45 | 10 |
| shaped | dev/42/16 | 0/15 | 0.0% | 66.7% | 9.87 | 20 | 7 |
| shaped | dev/42/24 | 2/15 | 13.3% | 53.3% | 8.87 | 10 | 5 |
| progress | dev/42/8 | 3/15 | 20.0% | 46.7% | 9.87 | 27 | 8 |
| progress | dev/42/16 | 3/15 | 20.0% | 53.3% | 10.00 | 26 | 8 |
| progress | dev/42/24 | 1/15 | 6.7% | 66.7% | 9.87 | 23 | 9 |
| sft-dev43 | dev/43/-1 | 2/15 | 13.3% | 40.0% | 10.60 | 35 | 8 |
| kl-dev43 | dev/43/-1 | 1/15 | 6.7% | 73.3% | 10.33 | 22 | 9 |
| outcome-dev43 | dev/43/-1 | 2/15 | 13.3% | 66.7% | 10.07 | 19 | 6 |
| consistent-dev43 | dev/43/-1 | 1/15 | 6.7% | 46.7% | 10.87 | 34 | 11 |
| shaped-dev43 | dev/43/-1 | 4/15 | 26.7% | 66.7% | 11.20 | 19 | 10 |
| progress-dev43 | dev/43/-1 | 1/15 | 6.7% | 46.7% | 10.53 | 34 | 7 |
| attention | dev/42/8 | 3/15 | 20.0% | 73.3% | 12.00 | 32 | 10 |
| attention | dev/42/16 | 2/15 | 13.3% | 40.0% | 10.20 | 12 | 8 |
| attention | dev/42/24 | 0/15 | 0.0% | 46.7% | 9.93 | 30 | 8 |
| attention-dev43 | dev/43/-1 | 3/15 | 20.0% | 60.0% | 8.53 | 17 | 6 |
| sft-test42 | test/42/-1 | 7/40 | 17.5% | 47.5% | 9.62 | 72 | 20 |
| attention-test42 | test/42/-1 | 9/40 | 22.5% | 45.0% | 9.18 | 64 | 18 |
| sft-test43 | test/43/-1 | 6/40 | 15.0% | 52.5% | 10.88 | 127 | 24 |
| attention-test43 | test/43/-1 | 9/40 | 22.5% | 52.5% | 10.00 | 94 | 20 |

## 开发集选型（15个任务 × 2个评测种子）

| 模型 | 检查点 | seed42成功 | seed43成功 | 合计成功率 |
|---|---:|---:|---:|---:|
| SFT | 1 | 3/15 | 2/15 | 16.7% |
| kl | 6 | 3/15 | 1/15 | 13.3% |
| outcome | 6 | 3/15 | 2/15 | 16.7% |
| consistent | 5 | 3/15 | 1/15 | 13.3% |
| shaped | 9 | 2/15 | 4/15 | 20.0% |
| progress | 14 | 3/15 | 1/15 | 13.3% |

冻结候选：shaped step 9；相对SFT合计成功数变化+1/30。
开发集用于选择，不能作为独立测试结果。两次评测复用相同15个任务。
原五组已按修订协议冻结为诊断对照；因确认训练掩码问题，不启动其测试，完整测试留给后续F组。

## 实际训练

```json
{
  "kl": {
    "groups": 24,
    "selected_trajectories": 96,
    "policy_updates": 9,
    "skipped_groups": 15,
    "effective_update_rate": 0.375,
    "all_failure_updates": 0,
    "optimizer_batches": 33,
    "train_successes": 11
  },
  "outcome": {
    "groups": 24,
    "selected_trajectories": 96,
    "policy_updates": 8,
    "skipped_groups": 16,
    "effective_update_rate": 0.3333333333333333,
    "all_failure_updates": 0,
    "optimizer_batches": 30,
    "train_successes": 12
  },
  "consistent": {
    "groups": 24,
    "selected_trajectories": 96,
    "policy_updates": 9,
    "skipped_groups": 15,
    "effective_update_rate": 0.375,
    "all_failure_updates": 0,
    "optimizer_batches": 34,
    "train_successes": 13
  },
  "shaped": {
    "groups": 24,
    "selected_trajectories": 96,
    "policy_updates": 8,
    "skipped_groups": 16,
    "effective_update_rate": 0.3333333333333333,
    "all_failure_updates": 0,
    "optimizer_batches": 28,
    "train_successes": 10
  },
  "progress": {
    "groups": 24,
    "selected_trajectories": 96,
    "policy_updates": 21,
    "skipped_groups": 3,
    "effective_update_rate": 0.875,
    "all_failure_updates": 13,
    "optimizer_batches": 78,
    "train_successes": 12
  },
  "attention": {
    "groups": 24,
    "selected_trajectories": 96,
    "policy_updates": 21,
    "skipped_groups": 3,
    "effective_update_rate": 0.875,
    "all_failure_updates": 16,
    "optimizer_batches": 76,
    "train_successes": 8
  }
}
```

## 基础设施失败尝试

以下尝试保留在原始轨迹并重试，不作为模型失败奖励；上表只统计完成的任务，每个任务去重一次。
```json
{
  "sft-dev42": {
    "dev:SimulatorGroundingError": 1
  },
  "outcome": {
    "dev:SimulatorGroundingError": 1
  },
  "consistent": {
    "dev:SimulatorGroundingError": 1
  },
  "shaped": {
    "dev:Timeout": 1
  },
  "progress": {
    "dev:SimulatorGroundingError": 6
  },
  "progress-dev43": {
    "dev:SimulatorGroundingError": 1
  },
  "attention": {
    "train:SimulatorGroundingError": 1,
    "train:InternalServerError": 12
  },
  "attention-dev43": {
    "dev:SimulatorGroundingError": 1
  }
}
```

## 固定输入概率重放

```json
{
  "old_policy_logprob_mae": {
    "original": 0.10792122036218643,
    "corrected": 0.044421471655368805
  },
  "old_policy_logprob_bias": {
    "original": -0.055621035397052765,
    "corrected": -0.012495363131165504
  },
  "old_policy_ratio_outside_clip": {
    "original": 0.13929961621761322,
    "corrected": 0.056031130254268646
  }
}
```
相同首批输入、首次正式更新前的只读重放；ART 冷启动已执行 lr=1e-9 warmup。
该批无位置重置，不能将改善归因给 packed position 修复。数值一致性不是任务成功率。

旧测试任务已在前次实验分析中出现；本次使用开发集选型，报告为已知基准上的后续改进。
尚未实现全面政策违规率验证。两次评测种子不等于两个独立训练种子。
计划和过程分别见 protocol.md、journal.md；最终候选选择见 selection.json（选型后生成）。
