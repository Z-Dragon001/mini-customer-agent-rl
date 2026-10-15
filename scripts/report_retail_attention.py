"""Report the attention follow-up independently of the original five-arm choice."""
import csv
import json
import random
import statistics
from pathlib import Path
import report_retail_grpo_repair as base_report

ROOT,OUT=base_report.ROOT,base_report.OUT


def main():
    base_report.main()
    runs=json.loads((OUT/'runs.json').read_text())
    lines=['# 注意力掩码修复：后续实验','',
           'F从同一SFT起点训练，与E仅增加打包注意力掩码修复；原五组的选择记录保留。',
           '官方任务成功率与数值测试分开报告。']
    selection_path=OUT/'selection-attention.json'
    if selection_path.exists():
        decision=json.loads(selection_path.read_text());candidate=decision['selected']
        sft=decision['sft_combined_score'][0];success=candidate['combined_score'][0]
        lines+=['',f"开发选型：F step {candidate['step']}（第{candidate['round']}组）；"
                f"seed42={candidate['seed42_score'][0]}/15，seed43={candidate['seed43_score'][0]}/15。",
                f"该检查点对应{candidate['round']*4}条已采样轨迹、{candidate['step']-1}次策略更新；"
                'F整轮共采样96条、更新21次。最终测试使用开发集选定的早期检查点，不使用最后一步权重。',
                f'两种子合计：SFT {sft}/30={sft/30:.1%}；F {success}/30={success/30:.1%}。',
                '开发集参与选型，两个种子复用同一批15个任务，不能作为独立测试性能。',
                '通过开发门槛，测试见下方完整记录。' if success>sft else
                '未通过开发门槛，未运行F测试，不宣称任务成功率提升。']
    else:
        lines+=['','尚未完成开发选型。']
    pairs=[]
    combined_rows={'sft':[],'attention':[]}
    lines+=['','每行覆盖40个任务才算该次完整测试；不足40的行仅为中间进度。','',
            '| 模型 | 测试种子 | 成功/覆盖 | 成功率 | 工具错误任务比例 | 平均调用 |',
            '|---|---:|---:|---:|---:|---:|']
    for seed in (42,43):
        rows_by_model={}
        for model in ('sft','attention'):
            name=f'{model}-test{seed}'
            if name not in runs:continue
            rows={r['task_id']:r for r in base_report.records(ROOT/runs[name]/'traces.jsonl')
                  if r['split']=='test' and not r.get('error')}
            rows_by_model[model]=rows
            combined_rows[model].extend(rows.values())
            if rows:
                metrics=base_report.summarize(list(rows.values()))
                lines.append(f"| {model} | {seed} | {metrics['successes']}/{metrics['tasks']} | "
                    f"{metrics['success_rate']:.1%} | {metrics['tool_error_episode_rate']:.1%} | {metrics['avg_tool_calls']:.2f} |")
        before,after=rows_by_model.get('sft',{}),rows_by_model.get('attention',{})
        if len(before)==40 and set(before)==set(after):
            pairs.extend({'seed':seed,'task_id':task,'sft':int(before[task]['success']),
                          'attention':int(after[task]['success'])} for task in sorted(before))
    if pairs:
        with (OUT/'paired_test-attention.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    if len(pairs)==80:
        clusters={}
        for row in pairs:clusters.setdefault(row['task_id'],[]).append(row['attention']-row['sft'])
        deltas=[statistics.mean(v) for v in clusters.values()]
        rng=random.Random(42)
        boot=sorted(statistics.mean(rng.choices(deltas,k=40)) for _ in range(10000))
        result={'episodes':80,'sft_success':sum(r['sft'] for r in pairs),
                'attention_success':sum(r['attention'] for r in pairs),
                'gains':sum(r['attention']>r['sft'] for r in pairs),
                'losses':sum(r['attention']<r['sft'] for r in pairs),
                'delta':statistics.mean(deltas),'cluster_bootstrap_95ci':[boot[250],boot[9749]],
                'sft_metrics':base_report.summarize(combined_rows['sft']),
                'attention_metrics':base_report.summarize(combined_rows['attention']),
                'note':'40 task clusters, two evaluation seeds; one training seed; known benchmark follow-up'}
        (OUT/'paired_summary-attention.json').write_text(json.dumps(result,indent=2)+'\n')
        baseline=result['sft_metrics'];repaired=result['attention_metrics']
        low,high=result['cluster_bootstrap_95ci']
        lines+=['','## 完整配对结果','',
                '每个模型在相同40个任务上分别执行seed42、43，共80次任务执行；不是80个独立任务。','',
                '| 指标 | SFT | SFT＋修复后的GRPO |',
                '|---|---:|---:|',
                f"| 任务成功 | {baseline['successes']}/80（{baseline['success_rate']:.2%}） | "
                f"{repaired['successes']}/80（{repaired['success_rate']:.2%}） |",
                f"| 出现工具错误的任务执行比例 | {baseline['tool_error_episode_rate']:.2%} | {repaired['tool_error_episode_rate']:.2%} |",
                f"| 平均工具调用次数 | {baseline['avg_tool_calls']:.3f} | {repaired['avg_tool_calls']:.3f} |",
                f"| 重复调用总数 | {baseline['repeated_calls']} | {repaired['repeated_calls']} |",
                f"| 预算耗尽次数 | {baseline['budget_exhausted']} | {repaired['budget_exhausted']} |",'',
                f"配对差值为{result['delta']*100:+.2f}个百分点；{result['gains']}次由失败变成功，"
                f"{result['losses']}次由成功变失败。按40个任务聚类、重采样10000次的95% bootstrap区间为"
                f"[{low*100:+.2f}, {high*100:+.2f}]个百分点。",
                '区间包含0，本次结果不足以证明稳定提升。' if low<=0<=high else
                '该区间仅反映本次任务样本的不确定性，未覆盖独立训练种子或外部用户分布。',
                '本次只有一个训练种子，测试集是已知基准的后续复测；工具错误比例不代表全面业务违规率。',
                '旧实验7.5%→20%→10%保留在retail-3b目录，本表使用本轮重新测量的SFT基线。',
                '机器可读结果见paired_summary-attention.json，逐任务配对见paired_test-attention.csv。']
    replay_path=ROOT/'.art/service-agent-retail/models/qwen25-3b-repair-attention/attention_audit.json'
    if replay_path.exists():
        replay=json.loads(replay_path.read_text())
        lines+=['','## 相同3B权重与输入的只读掩码对照','',
                '| 序列 | 打包位置偏离比例 | 原MAE | 保留掩码MAE | 原超出裁剪区间比例 | 保留掩码比例 |',
                '|---|---:|---:|---:|---:|---:|']
        for sequence in replay['sequences']:
            old=sequence['variants']['overwritten'];new=sequence['variants']['preserved']
            lines.append(f"| {sequence['index']} | {old['packed_position_reset_fraction']:.1%} | "
                f"{old['old_policy_logprob_mae']:.5f} | {new['old_policy_logprob_mae']:.5f} | "
                f"{old['old_policy_ratio_outside_clip']:.1%} | {new['old_policy_ratio_outside_clip']:.1%} |")
        lines+=['',replay['note'],'这是概率一致性证据，不是任务成功率。']
        if all(s['variants']['overwritten']['packed_position_reset_fraction']==0 for s in replay['sequences']):
            lines+=['这份冻结的3B批次没有分支位置重置，不能用它证明分支隔离改善；'
                    '分支隔离证据来自实际Unsloth/xFormers随机小Qwen2的GPU前向/反向测试。']
    lines+=['','完整训练计数、逐轮开发评测和异常尝试见report.md；配置、协议和过程见protocol.md及journal.md。']
    (OUT/'attention_report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
