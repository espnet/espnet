import logging
import os
import random

import numpy as np
import torch


def set_all_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # NOTE(kamo): torch.random.manual_seed() also seeds the RNG of every
    # available accelerator (CUDA, MPS, and XPU), so seeding them
    # one by one is not necessary here.
    torch.random.manual_seed(seed)


def set_deterministic(warn_only: bool = False):
    """Restrict PyTorch to the deterministic implementation of each operation.

    Seeding the RNGs is not enough to make a run bit-wise reproducible because
    several CUDA kernels accumulate with atomics, whose ordering varies between
    launches. This turns those kernels off; see
    https://pytorch.org/docs/stable/notes/randomness.html

    Note that not every operation has a deterministic implementation.
    ``torch.nn.CTCLoss`` is a notable one: its CUDA backward is
    non-deterministic, and the deterministic cuDNN implementation is used only
    when the inputs satisfy the conditions listed in the PyTorch document.
    Pass ``warn_only=True`` to downgrade the resulting RuntimeError into a
    warning if you want the remaining operations to stay deterministic.

    Args:
        warn_only: Warn instead of raising an error when an operation has no
            deterministic implementation.
    """
    # NOTE(kamo): cuBLAS reads CUBLAS_WORKSPACE_CONFIG when it creates its
    # handle, i.e. at the first GEMM call, so this must be set before any
    # computation happens.
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        logging.info(
            "CUBLAS_WORKSPACE_CONFIG is already set to "
            f"'{os.environ['CUBLAS_WORKSPACE_CONFIG']}' and it is kept as is. "
            "':4096:8' or ':16:8' is required for deterministic cuBLAS operations."
        )
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
