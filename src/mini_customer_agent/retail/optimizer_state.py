"""Atomic local optimizer/RNG checkpoints for ART's otherwise adapter-only saves."""
from pathlib import Path
import random


def save_optimizer_state(path: Path, optimizer, *, step: int, mode: str) -> None:
    import numpy as np
    import torch
    payload = {"step": step, "mode": mode, "optimizer": optimizer.state_dict(),
               "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
               "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_optimizer_state(path: Path, optimizer, *, expected_step: int) -> str:
    import numpy as np
    import torch
    # Only project-owned local checkpoint files; never load untrusted pickles.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['step'] != expected_step:
        raise ValueError('Optimizer and model checkpoint steps disagree')
    optimizer.load_state_dict(payload['optimizer'])
    random.setstate(payload['python_rng'])
    np.random.set_state(payload['numpy_rng'])
    torch.set_rng_state(payload['torch_rng'])
    if payload['cuda_rng']:
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return payload['mode']
