import unittest

from mini_customer_agent.retail.rewards import process_signals, reward_breakdown, group_is_informative


def call(name, arguments, i, response='{}'):
    return [{"role": "assistant", "tool_calls": [{"id": str(i), "function": {
        "name": name, "arguments": arguments}}]},
        {"role": "tool", "tool_call_id": str(i), "content": response}]


class RewardTests(unittest.TestCase):
    def test_state_changes_invalidate_reads_and_errors_do_not_seed_cache(self):
        messages = call('get_order_details', '{"order_id":"1"}', 0)
        messages += call('get_order_details', '{ "order_id": "1" }', 1)
        messages += call('modify_pending_order_address', '{"order_id":"1"}', 2)
        messages += call('get_order_details', '{"order_id":"1"}', 3)
        messages += call('get_order_details', '{"order_id":"2"}', 4, '{"error":"missing"}')
        messages += call('get_order_details', '{"order_id":"2"}', 5)
        messages += call('modify_pending_order_address', '{"order_id":"1","city":"A"}', 6)
        messages += call('modify_pending_order_address', '{"order_id":"1","city":"B"}', 7)
        messages += call('modify_pending_order_address', '{"order_id":"1","city":"A"}', 8)
        self.assertEqual(process_signals({"messages": messages})['repeated_calls'], 1)

    def test_success_dominates_failure_and_official_score_unchanged(self):
        row = {"official_reward": 1, "tool_errors": 100, "termination": "max_steps"}
        score = reward_breakdown(row, {"mode": "bounded_process"})
        self.assertAlmostEqual(score['total'], .7)
        self.assertEqual(row['official_reward'], 1)
        self.assertEqual(reward_breakdown(row)['total'], 1)
        with self.assertRaises(ValueError):
            reward_breakdown({**row, 'error': 'APIError'})
        with self.assertRaises(ValueError):
            reward_breakdown(row, {"mode": "bounded_process", "max_penalty": 1})

    def test_shaping_cannot_train_only_on_short_failed_episodes(self):
        spec = {'mode': 'bounded_process'}
        self.assertFalse(group_is_informative([0, -.1], [False, False], spec))
        self.assertTrue(group_is_informative([.98, -.1], [True, False], spec))
        self.assertTrue(group_is_informative([.98, 1], [True, True], spec))
        self.assertFalse(group_is_informative([1, 1], [True, True], spec))

    def test_business_progress_preserves_success_and_needs_real_group_difference(self):
        spec = {'mode': 'business_progress'}
        worst_success = {'official_reward': 1, 'split': 'train', 'tool_errors': 100,
                         'business_progress': {'score': 0}}
        best_failure = {'official_reward': 0, 'split': 'train', 'business_progress': {'score': .5}}
        self.assertGreater(reward_breakdown(worst_success, spec)['total'],
                           reward_breakdown(best_failure, spec)['total'])
        self.assertTrue(group_is_informative([.1, .3], [False, False], spec, [.1, .3]))
        self.assertFalse(group_is_informative([-.1, 0], [False, False], spec, [0, 0]))
        with self.assertRaises(ValueError):
            reward_breakdown({**best_failure, 'split': 'dev'}, spec)
        with self.assertRaises(ValueError):
            reward_breakdown(best_failure, {**spec, 'max_penalty': .5})


if __name__ == '__main__':
    unittest.main()
