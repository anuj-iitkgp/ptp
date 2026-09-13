"""
Seed and determinism utilities.
Ensures identical initial latent generation across runs.

Author: Anuj Yadav (IIT Kharagpur)
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set seeds across Python random, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_generator(seed: int, device: torch.device) -> torch.Generator:
    """
    Create a torch.Generator on the target device (or CPU for consistent CPU seed generators).
    Note: On Apple Silicon MPS, torch.Generator('mps') is supported in modern PyTorch,
    but CPU generator can also be used to generate initial latents deterministically and then transfer.
    """
    # For maximum cross-platform latent consistency, PyTorch generators on CPU produce
    # deterministic random normal tensors that can then be moved to the target device.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen
