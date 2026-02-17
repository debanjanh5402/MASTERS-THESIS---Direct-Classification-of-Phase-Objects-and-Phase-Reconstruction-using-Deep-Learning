import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set seed for reproducibility on:
    - macOS (MPS)
    - CPU
    """

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # MPS uses same RNG as CPU in PyTorch
    # (no separate manual_seed like CUDA)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ["PYTHONHASHSEED"] = str(seed)
