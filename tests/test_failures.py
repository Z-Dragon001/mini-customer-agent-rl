import unittest

from mini_customer_agent.failures import classify_failure
from mini_customer_agent.schemas import FailureType
from mini_customer_agent.task_data import load_tasks


class FailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = load_tasks("data/tasks/test.jsonl")[0]

    def test_wrong_tool(self):
        primary, labels = classify_failure(
            self.task,
            [
                {
                    "kind": "tool",
                    "tool_name": "request_refund",
                    "arguments": {},
                    "result": {"ok": False, "code": "PARAMETER_ERROR"},
                }
            ],
            False,
        )
        self.assertEqual(primary, FailureType.WRONG_TOOL)
        self.assertIn(FailureType.BAD_PARAMETERS.value, labels)

    def test_premature_termination(self):
        primary, _ = classify_failure(self.task, [], False)
        self.assertEqual(primary, FailureType.PREMATURE_TERMINATION)


if __name__ == "__main__":
    unittest.main()

