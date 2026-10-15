import importlib.util
import unittest
from unittest.mock import patch

from mini_customer_agent.retail.grounding import identifiers, make_grounded_user


class GroundingTests(unittest.TestCase):
    def test_known_identifiers_are_normalized(self):
        self.assertEqual(identifiers("A@EXAMPLE.COM #123456, ORD-123 and zip 60601; price 20.50"),
                         {"a@example.com", "123456", "ord-123", "60601"})

    @unittest.skipUnless(importlib.util.find_spec("tau2"), "retail dependency required")
    def test_rejected_candidate_does_not_pollute_history(self):
        from mini_customer_agent.retail.config import configure_tau, load_config
        from mini_customer_agent.retail.data import ROOT
        configure_tau(load_config(ROOT / "configs/retail.example.json"))
        from tau2.data_model.message import AssistantMessage, UserMessage
        user = make_grounded_user({"model": "offline"}, "Email: lucas@example.com", {})
        state = user.get_init_state()
        incoming = AssistantMessage(role="assistant", content="What is your order number?")
        candidates = iter(["ORD-9821", "I do not know it; my email is lucas@example.com."])

        def generate(_self, message, candidate_state):
            candidate_state.messages.append(message)
            return UserMessage(role="user", content=next(candidates))

        with patch("tau2.user.user_simulator.UserSimulator._generate_next_message", generate):
            response, state = user.generate_next_message(incoming, state)
        self.assertEqual(len(state.messages), 2)
        self.assertNotIn("ORD-9821", str(state.messages))
        self.assertIn("lucas@example.com", response.content)
        self.assertEqual(len(user.grounding_events), 1)
