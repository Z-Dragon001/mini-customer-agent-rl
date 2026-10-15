import json
import unittest
from mini_customer_agent.retail.judging import normalize_judge_json


class JudgeFormatTests(unittest.TestCase):
    def test_single_and_wrapped_judgments_preserve_false(self):
        row = {"expectedOutcome": "Exact criterion.", "reasoning": "Not met.", "metExpectation": False}
        for value in (row, [row], {"results": [row]}):
            self.assertEqual(normalize_judge_json(json.dumps(value)), {"results": [row]})

    def test_missing_results_and_non_boolean_values_are_rejected(self):
        for value in ({}, {"results": [{"expectedOutcome": "X", "reasoning": "Y", "metExpectation": "false"}]}):
            with self.assertRaises(ValueError):
                normalize_judge_json(json.dumps(value))
