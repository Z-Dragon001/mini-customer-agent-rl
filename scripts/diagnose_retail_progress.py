"""Offline train-only audit of business-progress signals, not a new experiment."""
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch

from mini_customer_agent.retail.config import configure_tau, load_config
from mini_customer_agent.retail.rewards import READ_TOOLS, WRITE_TOOLS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/retail-grpo-repair'


def main():
    configure_tau(load_config(OUT/'sft.json'))
    from loguru import logger
    logger.disable('tau2')
    from pydantic import TypeAdapter
    from tau2.data_model.message import Message
    from tau2.domains.retail.environment import get_environment,get_tasks
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    from tau2.evaluator.evaluator_action import _check_actions, ActionEvaluator
    tasks={task.id:task for task in get_tasks()}
    runs=json.loads((OUT/'runs.json').read_text())
    result=[];summary={}
    with patch('httpx.Client.send',side_effect=AssertionError('Offline audit attempted API access')):
        for arm in ['outcome','kl','consistent']:
            folder=ROOT/runs[arm]
            groups=[json.loads(line) for line in (folder/'updates.jsonl').read_text().splitlines()]
            rows={}
            for line in (folder/'traces.jsonl').read_text().splitlines():
                row=json.loads(line)
                if row['split']=='train' and not row.get('error') and row['group_id']<len(groups):
                    rows[row['group_id'],row['rollout_index']]=row
            by_group={}
            for (group,index),row in rows.items():
                task=tasks[row['task_id']]
                simulation=row['simulation']
                messages=TypeAdapter(list[Message]).validate_python(simulation.get('messages',simulation.get('partial_messages',[])))
                item={'arm':arm,'group':group,'index':index,'task_id':task.id,'success':row['success']}
                try:
                    reward=EnvironmentEvaluator.calculate_reward(get_environment,task,messages,strict_replay=True)
                    # EnvironmentEvaluator supplies DB checks, not action checks.
                    # Credit only calls with a successful corresponding response.
                    responded={message.id for message in messages
                               if message.role=='tool' and not message.error}
                    calls=[call for call in ActionEvaluator.extract_tool_calls(messages) if call.id in responded]
                    checks=_check_actions(calls,task.evaluation_criteria.actions or [])
                    for label,names in [('reads',READ_TOOLS),('writes',WRITE_TOOLS)]:
                        relevant=[check for check in checks if check.action.name in names]
                        item[label+'_expected']=len(relevant)
                        item[label+'_matched']=sum(check.action_match for check in relevant)
                        item[label+'_coverage']=item[label+'_matched']/len(relevant) if relevant else 0.0
                    item['db_goal_met']=bool(item['writes_expected'] and reward.db_check and reward.db_check.db_match)
                    item['progress']=.15*item['reads_coverage']+.25*item['writes_coverage']+.10*item['db_goal_met']
                    by_group.setdefault(group,[]).append(item)
                except Exception as exc:
                    item['replay_error']=type(exc).__name__
                result.append(item)
            failed=[items for items in by_group.values() if len(items)==4 and not any(item['success'] for item in items)]
            summary[arm]={'completed_groups':len(groups),'all_failure_groups':len(failed),
                          'all_failure_groups_with_progress_variance':sum(len({x['progress'] for x in items})>1 for items in failed),
                          'failed_trajectories_with_correct_write':sum(not x['success'] and x.get('writes_matched',0)>0 for x in result if x['arm']==arm),
                          'replay_errors':dict(Counter(x['replay_error'] for x in result if x['arm']==arm and 'replay_error' in x))}
    (OUT/'progress_diagnostics.json').write_text(json.dumps({'summary':summary,'rows':result,
        'note':'Train only. Hypothetical auxiliary reward; not used in the four registered arms. Does not change official scores.'},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
