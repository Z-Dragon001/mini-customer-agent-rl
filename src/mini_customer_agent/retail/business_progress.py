"""Privileged train-only reward signals; never included in policy observations."""
from .rewards import READ_TOOLS, WRITE_TOOLS


def summarize_progress(task, messages, db_match):
    from tau2.evaluator.evaluator_action import _check_actions, ActionEvaluator
    # Matching an attempted call is insufficient: require successful execution.
    responded = {message.id for message in messages
                 if message.role == 'tool' and not message.error}
    calls = [call for call in ActionEvaluator.extract_tool_calls(messages) if call.id in responded]
    checks = _check_actions(calls, task.evaluation_criteria.actions or [])
    result = {}
    for label, names in [('reads', READ_TOOLS), ('writes', WRITE_TOOLS)]:
        relevant = [check for check in checks if check.action.name in names]
        result[label + '_expected'] = len(relevant)
        result[label + '_matched'] = sum(check.action_match for check in relevant)
        result[label + '_coverage'] = result[label + '_matched'] / len(relevant) if relevant else 0.0
    # An unchanged DB on a read-only task is not partial execution progress.
    result['db_goal_met'] = bool(result['writes_expected'] and db_match)
    result['score'] = (.15 * result['reads_coverage'] + .25 * result['writes_coverage']
                       + .10 * result['db_goal_met'])
    return result


def calculate_business_progress(row):
    if row.get('split') != 'train' or row.get('error'):
        raise ValueError('Business-progress reward accepts valid training episodes only')
    from pydantic import TypeAdapter
    from tau2.data_model.message import Message
    from tau2.domains.retail.environment import get_environment, get_tasks
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    task = next(task for task in get_tasks() if task.id == row['task_id'])
    simulation = row['simulation']
    messages = TypeAdapter(list[Message]).validate_python(
        simulation.get('messages', simulation.get('partial_messages', [])))
    # Offline deterministic replay, including budget-terminated episodes. No
    # language-model judge and no rewriting of the official termination score.
    reward = EnvironmentEvaluator.calculate_reward(get_environment, task, messages, strict_replay=True)
    return summarize_progress(task, messages, bool(reward.db_check and reward.db_check.db_match))
