import random
import numpy as np
import torch


def setup_seed(seed=42):
    """Sets the random seed for reproducibility across NumPy, PyTorch, and Python's random module."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
