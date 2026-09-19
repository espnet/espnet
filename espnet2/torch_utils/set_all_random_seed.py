import logging
import os
import random

import numpy as np
import torch

# The only two values of CUBLAS_WORKSPACE_CONFIG that let cuBLAS operations,
# e.g. torch.mm, run under torch.use_deterministic_algorithms(True). Any other
# value, including "not set", makes them raise a RuntimeError.
DETERMINISTIC_CUBLAS_CONFIGS = (":4096:8", ":16:8")


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

    CUBLAS_WORKSPACE_CONFIG is set to ":4096:8" unless it already holds one of
    DETERMINISTIC_CUBLAS_CONFIGS, in which case the existing value is kept.

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
    config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if config not in DETERMINISTIC_CUBLAS_CONFIGS:
        if config is not None:
            logging.warning(
                f"CUBLAS_WORKSPACE_CONFIG='{config}' is not one of "
                f"{DETERMINISTIC_CUBLAS_CONFIGS}, so cuBLAS operations would "
                "raise a RuntimeError in deterministic mode. Overwriting it "
                f"with '{DETERMINISTIC_CUBLAS_CONFIGS[0]}'"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_CONFIGS[0]
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
