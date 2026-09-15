import unittest

from mini_customer_agent.evaluation import aggregate_metrics, evaluate
from mini_customer_agent.model_clients import ScriptedPolicyClient
from mini_customer_agent.task_data import load_tasks


class HarnessTests(unittest.TestCase):
    def test_scripted_oracle_completes_suite(self):
        traces, metrics = evaluate(
            ScriptedPolicyClient(), load_tasks("data/tasks/test.jsonl")
        )
        self.assertEqual(len(traces), 8)
        self.assertEqual(metrics["success_rate"], 1.0)
        self.assertEqual(metrics["illegal_operation_rate"], 0.0)
        self.assertTrue(all(trace["messages"] for trace in traces))

    def test_empty_aggregate(self):
        self.assertEqual(aggregate_metrics([]), {"episodes": 0})


if __name__ == "__main__":
    unittest.main()

