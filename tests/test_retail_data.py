import json
from pathlib import Path
import tempfile
import unittest

from mini_customer_agent.retail.dataset import build_sft, validate_messages, validate_splits
from mini_customer_agent.retail.data import REVISION
from mini_customer_agent.retail.rollout import training_reward


class RetailDataTests(unittest.TestCase):
    def setUp(self):
        self.splits = {"train": ["a"], "dev": ["b"], "test": ["c"], "tasks_sha256": "hash"}
        self.messages = [
            {"role": "user", "content": "Look up my order"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call1", "type": "function", "function": {"name": "query", "arguments": '{"id":"1"}'}}]},
            {"role": "tool", "tool_call_id": "call1", "content": '{"status":"pending"}'},
            {"role": "assistant", "content": "Your order is pending."},
        ]
        self.trace = {"split": "train", "task_id": "a", "source": "teacher", "success": True,
                      "tau_revision": REVISION, "tasks_sha256": "hash", "model": "teacher",
                      "tool_errors": 0, "messages": self.messages, "tools": []}

    def test_reject_split_overlap(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_splits({"train": ["a"], "dev": ["a"], "test": ["c"]}, {"a", "c"})

    def test_tool_responses_must_match_calls(self):
        validate_messages(self.messages)
        self.messages[2]["tool_call_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Orphan"):
            validate_messages(self.messages)

    def test_test_data_never_becomes_training_data(self):
        self.trace["split"] = "test"
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/"trace.jsonl", Path(tmp)/"sft.jsonl"
            source.write_text(json.dumps(self.trace), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-training"):
                build_sft(source, target, self.splits)
            self.assertFalse(target.exists())

    def test_sft_filters_errors_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/"trace.jsonl", Path(tmp)/"sft.jsonl"
            source.write_text("\n".join(json.dumps(row) for row in [self.trace, self.trace, {**self.trace, "tool_errors": 1}]), encoding="utf-8")
            counts = build_sft(source, target, self.splits)
            self.assertEqual(counts, {"accepted": 1, "failed": 1, "invalid": 0, "duplicate": 1})
            sample = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(sample["messages"], self.messages)
            self.assertEqual(sample["metadata"]["review_status"], "auto_filtered_not_human_reviewed")

    def test_reward_does_not_hide_infrastructure_errors(self):
        self.assertEqual(training_reward({"official_reward": 1, "tool_errors": 2}), 1)
        self.assertAlmostEqual(training_reward({"official_reward": 1, "tool_errors": 2}, .1), .8)
        with self.assertRaises(ValueError):
            training_reward({"error": "TimeoutError"})


if __name__ == "__main__":
    unittest.main()
