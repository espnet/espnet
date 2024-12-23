import random

import numpy as np
import torch


def set_all_random_seed(seed: int):
    """
        Sets the random seed for all relevant libraries to ensure reproducibility.

    This function sets the random seed for the built-in random module, NumPy,
    and PyTorch. By fixing the random seed, you can obtain the same results
    across multiple runs of your code, which is essential for debugging and
    testing.

    Args:
        seed (int): The random seed value to be set for reproducibility.

    Examples:
        >>> set_all_random_seed(42)
        This will set the random seed for random, numpy, and torch to 42.

    Note:
        It is advisable to call this function at the beginning of your script
        to ensure that all random number generation is controlled.
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)
