"""Numerical equivalence, including backward, of memory-bounded RL projection."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "GPU runtime provides torch")
class LogprobTests(unittest.TestCase):
    def test_values_and_gradients_match_dense_projection(self):
        import torch
        from mini_customer_agent.retail.logprobs import checkpointed_logprobs
        torch.manual_seed(42)
        states = torch.randn(2, 17, 8, dtype=torch.float64, requires_grad=True)
        head = torch.randn(8, 23, dtype=torch.float64, requires_grad=True)
        targets = torch.randint(23, (2, 17))
        weights = torch.randn(2, 17, dtype=torch.float64)
        logits = states @ head
        full = logits - logits.logsumexp(-1, keepdim=True)
        expected_lp = full.gather(-1, targets[..., None]).squeeze(-1)
        expected_entropy = -(full.exp() * full).sum(-1)
        expected_grad = torch.autograd.grad((expected_lp * weights + .03 * expected_entropy).sum(), (states, head))
        lp, entropy = checkpointed_logprobs(head, states, targets, chunk_size=4)
        actual_grad = torch.autograd.grad((lp * weights + .03 * entropy).sum(), (states, head))
        for actual, expected in zip((lp, entropy, *actual_grad), (expected_lp, expected_entropy, *expected_grad)):
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        with torch.no_grad():
            inferred, _ = checkpointed_logprobs(head, states, targets, chunk_size=4)
        torch.testing.assert_close(inferred, expected_lp)


if __name__ == "__main__":
    unittest.main()
