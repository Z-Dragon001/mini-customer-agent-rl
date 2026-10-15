"""Derive comparison tables from saved real trajectories, including partial runs."""
from collections import Counter
import csv
import json
from pathlib import Path
import random
import statistics

from mini_customer_agent.retail.rewards import process_signals

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/retail-grpo-repair'


def records(path):
    if not path.exists():
        return []
    content = path.read_text()
    lines = content.splitlines()
    if content and not content.endswith('\n'):
        lines = lines[:-1]
    return [json.loads(line) for line in lines]


def summarize(rows):
    n = len(rows)
    signals = [process_signals(row) for row in rows]
    return dict(tasks=n, successes=sum(r['success'] for r in rows),
                success_rate=sum(r['success'] for r in rows)/n,
                tool_error_episode_rate=sum(r['tool_errors']>0 for r in rows)/n,
                avg_tool_calls=statistics.mean(r['tool_calls'] for r in rows),
                repeated_calls=sum(s['repeated_calls'] for s in signals),
                budget_exhausted=sum(s['budget_exhausted'] for s in signals))


def main():
    runs = json.loads((OUT/'runs.json').read_text())
    tables, training, evaluations, infrastructure, metric_distributions = [], {}, {}, {}, {}
    for name, path in runs.items():
        folder = ROOT/path
        rows = records(folder/'traces.jsonl')
        infrastructure[name] = dict(Counter(f"{row['split']}:{row['error']}"
                                           for row in rows if row.get('error')))
        grouped = {}
        for row in rows:
            if row.get('error') or row['split']=='train':
                continue
            group = (row['split'], row.get('evaluation_round', -1), row['seed'], row.get('policy_step'))
            grouped.setdefault(group, {})[row['task_id']] = row
        for (split, round_id, seed, step), selected in sorted(grouped.items()):
            valid = list(selected.values())
            tables.append(dict(run=name, split=split, round=round_id, seed=seed,
                               step=step, **summarize(valid)))
            evaluations[name,split,round_id] = selected
        updates = records(folder/'updates.jsonl')
        if updates:
            selected = {(r['group_id'],r['rollout_index']):r for r in rows
                        if r['split']=='train' and not r.get('error')
                        and r['group_id'] < len(updates)}
            counts = Counter('updated' if r['updated'] else 'skipped' for r in updates)
            metrics = records(folder/'grpo_batch_metrics.jsonl')
            metric_distributions[name]={}
            for key in ('loss','grad_norm','kl_div','old_policy_logprob_mae','old_policy_ratio_outside_clip'):
                values=[record['metrics'][key] for record in metrics
                        if isinstance(record['metrics'].get(key),(int,float))]
                if values:
                    metric_distributions[name][key]={'batches':len(values),'min':min(values),
                                                     'median':statistics.median(values),'max':max(values)}
            failed_updates=sum(group['updated'] and
                               not any(row['success'] for (g,_),row in selected.items() if g==group['iteration'])
                               for group in updates)
            training[name] = dict(groups=len(updates), selected_trajectories=len(selected),
                                  policy_updates=counts['updated'], skipped_groups=counts['skipped'],
                                  effective_update_rate=counts['updated']/len(updates),
                                  all_failure_updates=failed_updates,
                                  optimizer_batches=len(metrics),
                                  train_successes=sum(r['success'] for r in selected.values()))
    if tables:
        with (OUT/'results.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(tables[0]))
            writer.writeheader()
            writer.writerows(tables)
    (OUT/'training_summary.json').write_text(json.dumps(training,indent=2)+'\n')
    (OUT/'infrastructure_summary.json').write_text(json.dumps(infrastructure,indent=2)+'\n')
    (OUT/'training_metrics_summary.json').write_text(json.dumps({
        'arms':metric_distributions,
        'note':'Descriptive distributions over actual optimizer batches, not matched-input causal comparisons; '
               'later batches may follow prior optimizer updates. See numerical_comparison.json for the fixed-input replay.'
    },indent=2)+'\n')
    lines=['# GRPO 改进实验结果', '',
           '每行覆盖必须达到 dev 15 或 test 40 才能作为该次完整评测。过程奖励不替代官方成功率。', '',
           '| 实验 | 划分/种子/轮次 | 成功 | 成功率 | 工具错误任务比例 | 平均调用 | 重复调用 | 预算耗尽 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in tables:
        lines.append(f"| {row['run']} | {row['split']}/{row['seed']}/{row['round']} | "
                     f"{row['successes']}/{row['tasks']} | {row['success_rate']:.1%} | "
                     f"{row['tool_error_episode_rate']:.1%} | {row['avg_tool_calls']:.2f} | "
                     f"{row['repeated_calls']} | {row['budget_exhausted']} |")
    selection_path=OUT/'selection.json'
    if selection_path.exists():
        diagnostic_only=any(event.get('stage')=='selection' and event.get('state')=='development_complete'
                            for event in records(OUT/'events.jsonl'))
        selection=json.loads(selection_path.read_text())
        chosen=selection['selected']
        baseline=selection['sft_combined_score'][0]
        gain=chosen['combined_score'][0]-baseline
        lines+=['', '## 开发集选型（15个任务 × 2个评测种子）', '',
                '| 模型 | 检查点 | seed42成功 | seed43成功 | 合计成功率 |',
                '|---|---:|---:|---:|---:|']
        first=next(r for r in tables if r['run']=='sft-dev42')
        second=next(r for r in tables if r['run']=='sft-dev43')
        lines.append(f"| SFT | 1 | {first['successes']}/15 | {second['successes']}/15 | {baseline/30:.1%} |")
        for candidate in selection['candidates']:
            lines.append(f"| {candidate['arm']} | {candidate['step']} | "
                         f"{candidate['seed42_score'][0]}/15 | {candidate['seed43_score'][0]}/15 | "
                         f"{candidate['combined_score'][0]/30:.1%} |")
        lines+=['', f"冻结候选：{chosen['arm']} step {chosen['step']}；相对SFT合计成功数变化{gain:+d}/30。",
                '开发集用于选择，不能作为独立测试结果。两次评测复用相同15个任务。',
                ('原五组已按修订协议冻结为诊断对照；因确认训练掩码问题，不启动其测试，完整测试留给后续F组。' if diagnostic_only else
                 '通过预先规定的开发门槛；测试结果以本报告完整测试行和配对表为准。' if gain>0 else
                 '未通过开发门槛，本轮停止在开发评测，不运行测试、不宣称任务成功率提升。')]
    lines+=['', '## 实际训练', '', '```json', json.dumps(training,ensure_ascii=False,indent=2), '```']
    failures={name:counts for name,counts in infrastructure.items() if counts}
    lines+=['','## 基础设施失败尝试','',
            '以下尝试保留在原始轨迹并重试，不作为模型失败奖励；上表只统计完成的任务，每个任务去重一次。',
            '```json',json.dumps(failures,ensure_ascii=False,indent=2),'```']
    dev_baseline=evaluations.get(('sft-dev42','dev',-1),{})
    dev_changes=[]
    for (name,split,round_id),candidate in evaluations.items():
        if split!='dev' or round_id<0 or set(candidate)!=set(dev_baseline):
            continue
        for task in sorted(dev_baseline):
            before,after=dev_baseline[task],candidate[task]
            old,new=bool(before['success']),bool(after['success'])
            change='gained' if new and not old else 'lost' if old and not new else 'both_success' if new else 'both_fail'
            def detail(row):
                return {'success':row['success'],'termination':row.get('termination'),
                        'tool_calls':row['tool_calls'],**process_signals(row)}
            dev_changes.append({'arm':name,'round':round_id,'task_id':task,'change':change,
                                'sft':detail(before),'candidate':detail(after)})
    (OUT/'dev_task_changes.json').write_text(json.dumps(dev_changes,indent=2)+'\n')
    numerical_path=ROOT/'.art/service-agent-retail/models/qwen25-3b-repair-consistent/numerical_audit.json'
    if numerical_path.exists():
        numerical=json.loads(numerical_path.read_text())
        control=records(ROOT/runs['outcome']/'grpo_batch_metrics.jsonl')[0]['metrics']
        corrected=numerical['variants']['bf16_fp32']
        comparison={key:{'original':control[key],'corrected':corrected[key]}
                    for key in ('old_policy_logprob_mae','old_policy_logprob_bias','old_policy_ratio_outside_clip')}
        (OUT/'numerical_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
        lines+=['','## 固定输入概率重放','', '```json',json.dumps(comparison,indent=2),'```',
                '相同首批输入、首次正式更新前的只读重放；ART 冷启动已执行 lr=1e-9 warmup。',
                '该批无位置重置，不能将改善归因给 packed position 修复。数值一致性不是任务成功率。']
    pairs=[]
    for seed in [42,43]:
        base=evaluations.get((f'sft-test{seed}','test',-1),{})
        candidate=evaluations.get((f'selected-test{seed}','test',-1),{})
        if len(base)==40 and set(base)==set(candidate):
            for task in sorted(base):
                pairs.append(dict(seed=seed,task_id=task,sft=int(base[task]['success']),
                                  grpo=int(candidate[task]['success'])))
    if pairs:
        with (OUT/'paired_test.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(pairs[0]))
            writer.writeheader();writer.writerows(pairs)
    if len(pairs)==80:
        gains=sum(p['grpo']>p['sft'] for p in pairs)
        losses=sum(p['grpo']<p['sft'] for p in pairs)
        clusters={}
        for p in pairs:
            clusters.setdefault(p['task_id'],[]).append(p['grpo']-p['sft'])
        deltas=[statistics.mean(v) for v in clusters.values()]
        rng=random.Random(42)
        boot=sorted(statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(10000))
        result=dict(episodes=80,sft_success=sum(p['sft'] for p in pairs),
                    grpo_success=sum(p['grpo'] for p in pairs),gains=gains,losses=losses,
                    delta=statistics.mean(deltas),cluster_bootstrap_95ci=[boot[250],boot[9749]],
                    note='40 task clusters × 2 evaluation seeds; not two independent training seeds')
        (OUT/'paired_summary.json').write_text(json.dumps(result,indent=2)+'\n')
        lines+=['','## 冻结选择后的配对测试','', '```json', json.dumps(result,indent=2), '```']
    lines+=['', '旧测试任务已在前次实验分析中出现；本次使用开发集选型，报告为已知基准上的后续改进。',
            '尚未实现全面政策违规率验证。两次评测种子不等于两个独立训练种子。',
            '计划和过程分别见 protocol.md、journal.md；最终候选选择见 selection.json（选型后生成）。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'evaluations':len(tables),'training':training},indent=2))


if __name__=='__main__':
    main()
