"""Offline replay: separate unfinished business state from failure to terminate.

Only historical train/dev traces are inspected; this does not change official
scores, supply model inputs, or call an LLM judge.
"""
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch

from mini_customer_agent.retail.config import configure_tau, load_config
from mini_customer_agent.retail.rewards import WRITE_TOOLS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/retail-grpo-repair'


def main():
    configure_tau(load_config(ROOT/'configs/retail.experiment.json'))
    from loguru import logger
    logger.disable('tau2')
    from tau2.data_model.simulation import SimulationRun
    from tau2.data_model.tasks import RewardType
    from tau2.domains.retail.environment import get_environment, get_tasks
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    tasks = {t.id:t for t in get_tasks()}
    runs=json.loads((ROOT/'experiments/retail-3b/runs.json').read_text())
    result=[]
    with patch('httpx.Client.send', side_effect=AssertionError('Offline replay attempted API access')):
        for stage in ['sft_eval','grpo_eval','grpo']:
            source=ROOT/runs[stage]/'traces.jsonl'
            selected={}
            for line in source.read_text().splitlines():
                row=json.loads(line)
                if row['split'] not in ('dev','train') or row.get('error'):
                    continue
                selected[row['task_id'],row.get('group_id'),row.get('rollout_index')]=row
            for row in selected.values():
                if row['success'] or row['termination'] not in ('max_steps','context_length_exceeded'):
                    continue
                task=tasks[row['task_id']]
                criteria=task.evaluation_criteria
                if not criteria or RewardType.DB not in criteria.reward_basis:
                    continue
                if not any(action.name in WRITE_TOOLS for action in criteria.actions or []):
                    continue  # Unchanged initial DB is not evidence of progress.
                item={'stage':stage,'task_id':row['task_id'],'split':row['split'],
                      'group_id':row.get('group_id'),'rollout_index':row.get('rollout_index'),
                      'termination':row['termination'],'official_reward':row['official_reward']}
                try:
                    if 'partial_messages' in row['simulation']:
                        from pydantic import TypeAdapter
                        from tau2.data_model.message import Message
                        messages=TypeAdapter(list[Message]).validate_python(row['simulation']['partial_messages'])
                    else:
                        messages=SimulationRun.model_validate(row['simulation']).messages
                    reward=EnvironmentEvaluator.calculate_reward(
                        get_environment,task,messages,strict_replay=True)
                    item['db_goal_met']=bool(reward.db_check.db_match)
                except Exception as exc:
                    item['replay_error']=type(exc).__name__
                result.append(item)
    summary={}
    for stage in ['sft_eval','grpo_eval','grpo']:
        subset=[r for r in result if r['stage']==stage]
        summary[stage]={'eligible_budget_failures':len(subset),
                        'db_goal_met_but_unfinished':sum(r.get('db_goal_met',False) for r in subset),
                        'replay_errors':dict(Counter(r['replay_error'] for r in subset if r.get('replay_error')))}
    output={'summary':summary,'rows':result,
            'interpretation':'DB component only on tasks requiring mutation; not full task success or policy compliance. Official scores unchanged.'}
    (OUT/'termination_diagnostics.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
