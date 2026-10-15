import importlib.util
from pathlib import Path
import random
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires GPU runtime (test runs on CPU)')
class OptimizerTests(unittest.TestCase):
    def test_resumed_adam_update_and_rng_equal_uninterrupted(self):
        import numpy as np
        import torch
        from mini_customer_agent.retail.optimizer_state import save_optimizer_state, restore_optimizer_state
        x = torch.nn.Parameter(torch.tensor([1., -1.], dtype=torch.float64))
        opt = torch.optim.AdamW([x], lr=.01)
        def update(p, optimizer):
            (p.square().sum()).backward()
            optimizer.step()
            optimizer.zero_grad()
        update(x, opt)
        before = x.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'optimizer.pt'
            save_optimizer_state(path, opt, step=2, mode='rl')
            expected_rng = (random.random(), np.random.rand(), torch.rand(1))
            update(x, opt)
            y = torch.nn.Parameter(before)
            restored = torch.optim.AdamW([y], lr=.8)
            self.assertEqual(restore_optimizer_state(path, restored, expected_step=2), 'rl')
            observed_rng = (random.random(), np.random.rand(), torch.rand(1))
            self.assertEqual(expected_rng[:2], observed_rng[:2])
            torch.testing.assert_close(expected_rng[2], observed_rng[2], rtol=0, atol=0)
            update(y, restored)
            torch.testing.assert_close(x, y, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                restore_optimizer_state(path, restored, expected_step=3)


if __name__ == '__main__':
    unittest.main()
