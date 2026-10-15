"""Build experiment tables only from saved measurements; safe to run for progress."""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/retail-3b"


def read_rows(path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if text and not text.endswith("\n"):
        lines = lines[:-1]  # Only a writer's unfinished final row is skipped.
    return [json.loads(line) for line in lines if line]


def failure_labels(row):
    if row.get("success"):
        return []
    labels = []
    if row.get("termination") == "invalid_model_output":
        labels.append("模型输出格式错误")
    if row.get("tool_errors", 0):
        labels.append("工具执行错误")
    if row.get("termination") in {"max_steps", "context_length_exceeded"}:
        labels.append("交互预算耗尽")
    reward = row.get("simulation", {}).get("reward_info", {}) or {}
    breakdown = reward.get("reward_breakdown", {}) or {}
    if breakdown.get("NL_ASSERTION") == 0:
        labels.append("自然语言断言未满足")
    if breakdown.get("DB") == 0 or (not labels and row.get("official_reward") == 0):
        labels.append("业务目标未完成")
    return labels


def main():
    runs = json.loads((OUT / "runs.json").read_text(encoding="utf-8"))
    splits = json.loads((ROOT / "data/tau2/splits.json").read_text(encoding="utf-8"))
    tables, tasks, progress, by_stage, failure_counts = [], [], {}, {}, []
    for stage, folder in runs.items():
        if folder is None:
            progress[stage] = {"status": "not_started"}
            continue
        raw = read_rows(ROOT / folder / "traces.jsonl")
        valid = {(r["split"], r["task_id"], r["seed"]): r for r in raw if not r.get("error")}
        progress[stage] = {"folder": folder, "completed_episodes": len(valid),
                           "successful_episodes": sum(r["success"] for r in valid.values()),
                           "error_attempts": dict(Counter(r["error"] for r in raw if r.get("error")))}
        if stage == "sft":
            metrics = read_rows(ROOT / folder / "sft_metrics.jsonl")
            progress[stage]["training_batches"] = len(metrics)
            if metrics:
                progress[stage].update(first_loss=metrics[0]["loss"], latest_loss=metrics[-1]["loss"],
                                       supervised_tokens=sum(r["num_trainable_tokens"] for r in metrics),
                                       mean_loss=statistics.mean(r["loss"] for r in metrics),
                                       finite_metrics=all(math.isfinite(r[k]) for r in metrics for k in ("loss", "grad_norm")))
                if len(metrics) == 116:
                    progress[stage]["epoch_mean_losses"] = [statistics.mean(r["loss"] for r in metrics[i:i+58]) for i in (0, 58)]
        if stage.startswith("grpo") and stage != "grpo_eval":
            # Four stochastic rollouts deliberately share task/seed. They are
            # distinct RL samples, unlike an evaluation retry of one task.
            # A fully resampled incomplete group supersedes its older partial
            # attempts. Raw rows and resumption events remain available.
            rollouts = list({(r.get("group_id", 0), r.get("rollout_index", i)): r
                             for i, r in enumerate(raw) if not r.get("error")}.values())
            progress[stage].update(completed_episodes=len(rollouts),
                                   successful_episodes=sum(r["success"] for r in rollouts))
            updates = read_rows(ROOT / folder / "updates.jsonl")
            progress[stage].update(rollout_groups=len(updates), policy_updates=sum(r["updated"] for r in updates),
                                   optimizer_batches=len(read_rows(ROOT / folder / "grpo_batch_metrics.jsonl")),
                                   zero_reward_groups=sum(not any(r["rewards"]) for r in updates),
                                   all_success_groups=sum(all(v == 1 for v in r["rewards"]) for r in updates),
                                   resumptions=len(read_rows(ROOT / folder / "resumptions.jsonl")))
        if stage not in {"base", "sft_eval", "grpo_eval"}:
            continue
        by_stage[stage] = valid
        for part in ("dev", "test"):
            rows = [r for r in valid.values() if r["split"] == part]
            if not rows:
                continue
            success = sum(r["success"] for r in rows)
            record = {"stage": stage, "split": part, "evaluated": len(rows), "expected": len(splits[part]),
                      "successes": success, "success_rate": success / len(rows),
                      "tool_error_episode_rate": sum(r["tool_errors"] > 0 for r in rows) / len(rows),
                      "avg_tool_calls": statistics.mean(r["tool_calls"] for r in rows),
                      "avg_tool_calls_successful": statistics.mean(r["tool_calls"] for r in rows if r["success"]) if success else None,
                      "avg_seconds": statistics.mean(r["elapsed_ms"] / 1000 for r in rows),
                      "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in rows),
                      "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in rows),
                      "grounding_rejections": sum(len(r.get("user_grounding_events", [])) for r in rows),
                      "policy_step": rows[0]["policy_step"], "source": folder}
            tables.append(record)
            counts = Counter(label for row in rows for label in failure_labels(row))
            failure_counts.extend({"stage": stage, "split": part, "failure": key, "episodes": value}
                                  for key, value in counts.items())
            for row in rows:
                tasks.append({"stage": stage, "split": part, "task_id": row["task_id"],
                              "success": row["success"], "tool_calls": row["tool_calls"],
                              "tool_errors": row["tool_errors"], "termination": row["termination"],
                              "policy_step": row["policy_step"], "failure_labels": " / ".join(failure_labels(row))})
    for filename, rows in (("results.csv", tables), ("task_results.csv", tasks), ("failure_counts.csv", failure_counts)):
        if rows:
            with (OUT / filename).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    progress["updated_at"] = datetime.now(timezone.utc).isoformat()
    (OUT / "status.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 客服 Agent 实验结果", "", f"更新：{progress['updated_at']}", "",
             "表格仅使用实际完成的任务。覆盖未达到15/15或40/40时为中间进度，不能当作完整对照。", "",
             "| 阶段 | 划分 | 覆盖 | 成功数 | 成功率 | 工具错误任务比例 | 平均工具调用 | 检查点 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    test_results = {r["stage"]: r for r in tables if r["split"] == "test" and r["evaluated"] == r["expected"]}
    if set(test_results) == {"base", "sft_eval", "grpo_eval"}:
        base_rate, sft_rate, grpo_rate = (test_results[k]["success_rate"] for k in ("base", "sft_eval", "grpo_eval"))
        lines[4:4] = [f"本轮完成：测试集Base {base_rate:.1%} → SFT {sft_rate:.1%} → SFT＋GRPO {grpo_rate:.1%}。"
                      f"SFT相对Base变化{100*(sft_rate-base_rate):+.1f}个百分点，GRPO相对SFT变化{100*(grpo_rate-sft_rate):+.1f}个百分点。"
                      "结论仅限本次单种子、小规模实验；[详细解读与下一轮工作](analysis.md)。", ""]
    for r in tables:
        lines.append(f"| {r['stage']} | {r['split']} | {r['evaluated']}/{r['expected']} | {r['successes']} | "
                     f"{r['success_rate']:.1%} | {r['tool_error_episode_rate']:.1%} | {r['avg_tool_calls']:.2f} | {r['policy_step']} |")
    lines += ["", "## 实际训练记录", ""]
    sft = progress.get("sft", {})
    if sft.get("training_batches"):
        lines.append(f"SFT：已记录{sft['training_batches']}/116个优化批次，监督目标token累计{int(sft['supervised_tokens']):,}；loss/梯度有限值检查：{sft['finite_metrics']}。")
        if "epoch_mean_losses" in sft:
            lines.append("两个epoch的批次平均训练loss分别为" + "、".join(f"{x:.4f}" for x in sft["epoch_mean_losses"]) + "。这是训练集拟合指标，不能替代独立任务评测。")
    grpo = progress.get("grpo", {})
    if grpo.get("rollout_groups"):
        lines.append(f"GRPO：{grpo['rollout_groups']}组采样、{grpo['policy_updates']}轮策略更新（保存检查点）、{grpo['optimizer_batches']}个优化批次；同奖励组不更新。奖励使用官方任务成功0/1，本轮未加入额外惩罚项。")
        lines.append(f"其中{grpo['zero_reward_groups']}组全部失败、{grpo['all_success_groups']}组全部成功。")
        if grpo.get("resumptions"):
            lines.append(f"发生{grpo['resumptions']}次中途续跑，LoRA权重保留、优化器动量重置；具体边界和分类修复见过程日志。不能称为严格等价的不中断训练。")
    lines += ["", "## 配对任务变化", ""]
    for before, after in (("base", "sft_eval"), ("sft_eval", "grpo_eval")):
        left, right = by_stage.get(before, {}), by_stage.get(after, {})
        shared = [key for key in left.keys() & right.keys() if key[0] == "test"]
        if not shared:
            continue
        gained = sorted(key[1] for key in shared if not left[key]["success"] and right[key]["success"])
        lost = sorted(key[1] for key in shared if left[key]["success"] and not right[key]["success"])
        lines.append(f"- {before} → {after}：共同测试任务{len(shared)}个；失败转成功{len(gained)}个{gained}；成功转失败{len(lost)}个{lost}。")
    lines += ["", "## 记录位置与解释", "",
              "- 协议：[protocol.md](protocol.md)；过程：[journal.md](journal.md)；详细指标：[results.csv](results.csv)；逐任务：[task_results.csv](task_results.csv)。",
              "- 原始数据位置：[runs.json](runs.json)；当前阶段状态及错误：[status.json](status.json)。",
              "- 最终一致性核验：[audit.json](audit.json)；阶段耗时与采样显存：[resources.json](resources.json)；代码验证：[verification.json](verification.json)。",
              "- [具体行为案例](cases.md)；[SFT训练曲线](sft_training.png)；[GRPO采样奖励](grpo_sampling.png)；[测试成功率对照图](test_success.png)。",
              "- 单次种子结果，不能据此宣称统计显著或稳定提升。工具错误率不是全面政策违规率。",
              "- 平均调用数包含提前失败的任务；CSV另列成功任务的平均调用数。不同阶段成功任务集合不同，不能仅凭这两列判断效率优劣。",
              "- [失败分类](failure_counts.csv)按可观测事件统计，允许一条轨迹有多个标签，不等价于完整因果归因。",
              "- 模拟用户增加了可见编号约束；仍可能产生其他语义偏差。云端API别名可能变化。",
              "- 费用和全面政策违规率未可靠测量，未填入数字。负面结果与故障保留。", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(progress, ensure_ascii=False))


if __name__ == "__main__":
    main()
