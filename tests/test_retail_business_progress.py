import importlib.util
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec('tau2'), 'Requires retail environment')
class ProgressTests(unittest.TestCase):
    def test_successful_exact_actions_count_once_and_failed_calls_do_not(self):
        from tau2.data_model.tasks import Action
        from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
        from mini_customer_agent.retail.business_progress import summarize_progress, calculate_business_progress
        action = Action(action_id='1', requestor='assistant', name='modify_pending_order_address',
                        arguments={'order_id': '#1', 'city': 'A'})
        task = SimpleNamespace(evaluation_criteria=SimpleNamespace(actions=[action]))
        messages = []
        for index, (city, failed) in enumerate([('A', True), ('B', False)]):
            call = ToolCall(id=str(index), name=action.name, arguments={'order_id':'#1','city':city})
            messages.extend([AssistantMessage(role='assistant',tool_calls=[call]),
                             ToolMessage(role='tool',id=str(index),requestor='assistant',content='{}',error=failed)])
        self.assertEqual(summarize_progress(task, messages, False)['score'], 0)
        for index in [2,3]:
            call = ToolCall(id=str(index), name=action.name, arguments=action.arguments)
            messages.extend([AssistantMessage(role='assistant',tool_calls=[call]),
                             ToolMessage(role='tool',id=str(index),requestor='assistant',content='{}',error=False)])
        result = summarize_progress(task, messages, True)
        self.assertEqual(result['writes_matched'], 1)
        self.assertAlmostEqual(result['score'], .35)
        empty = SimpleNamespace(evaluation_criteria=SimpleNamespace(actions=[]))
        self.assertEqual(summarize_progress(empty, messages, True)['score'], 0)
        with self.assertRaises(ValueError):
            calculate_business_progress({'split':'test'})


if __name__ == '__main__':
    unittest.main()
