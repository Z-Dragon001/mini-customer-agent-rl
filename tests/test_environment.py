import unittest

from mini_customer_agent.environment import CustomerServiceEnvironment
from mini_customer_agent.task_data import load_tasks


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = {task.task_id: task for task in load_tasks("data/tasks/test.jsonl")}

    def test_change_requires_real_confirmation(self):
        env = CustomerServiceEnvironment(self.tasks["test_change_multiturn"])
        args = {
            "order_id": "O300",
            "province": "广东省",
            "city": "深圳市",
            "detail": "南山区科技园8号",
            "confirmed": True,
        }
        denied = env.call_tool("change_address", args)
        self.assertFalse(denied.ok)
        self.assertEqual(denied.code, "MISSING_INFORMATION")
        env.observe_user_message("是的，我确认修改。")
        allowed = env.call_tool("change_address", args)
        self.assertTrue(allowed.ok)
        self.assertTrue(env.goal_satisfied())

    def test_cannot_access_another_customer_order(self):
        task = self.tasks["test_lookup_pending"]
        task.seed_orders.append(
            {
                "order_id": "OTHER-1",
                "customer_id": "someone-else",
                "status": "pending",
                "total": 1.0,
                "refundable": True,
                "province": "X",
                "city": "Y",
                "detail": "Z",
            }
        )
        env = CustomerServiceEnvironment(task)
        result = env.call_tool("query_order", {"order_id": "OTHER-1"})
        self.assertFalse(result.ok)
        self.assertTrue(result.illegal_operation)


if __name__ == "__main__":
    unittest.main()

