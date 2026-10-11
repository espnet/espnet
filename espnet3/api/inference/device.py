"""Where a model is built: ``"auto"`` resolved to a device torch has."""

from __future__ import annotations

import logging
import re

import torch

logger = logging.getLogger(__name__)

_CUDA = re.compile(r"cuda(:\d+)?")


def resolve_device(requested: str = "auto") -> str:
    """Return the device a model is built on, for ``"auto"`` or a named one.

    ``"auto"`` is CUDA when torch sees a CUDA device, else the CPU, so a
    caller need not know whether this machine has a GPU. MPS is not chosen
    on its own: it is faster than the CPU for some models' encoders and
    slower for a CTC prefix search, and until that is measured per model
    it is used only when asked for by name.

    A named device is checked against what torch has, so a typo or a
    device this machine lacks fails here, before a model is downloaded and
    built, rather than in the first forward pass.

    Args:
        requested: ``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda"`` or
            ``"cuda:<n>"``.

    Returns:
        A device torch accepts: never ``"auto"``.

    Raises:
        ValueError: If ``requested`` names no device, or one this machine
            does not have.

    Examples:
        >>> resolve_device()            # a machine without a GPU
        'cpu'
        >>> resolve_device("cuda:0")    # on one with a GPU
        'cuda:0'
        >>> resolve_device("gpu")
        Traceback (most recent call last):
        ValueError: unknown device 'gpu': use auto, cpu, mps, cuda or cuda:<n>
    """
    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("device auto: %s", device)
        return device
    if requested == "cpu":
        return requested
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(
                "mps asked for, but torch has no MPS device here; pass "
                "device='cpu', or 'auto'"
            )
        return requested
    if _CUDA.fullmatch(requested):
        if not torch.cuda.is_available():
            raise ValueError(
                f"{requested} asked for, but torch sees no CUDA device; pass "
                "device='cpu', or 'auto'"
            )
        index = int(requested[5:]) if ":" in requested else 0
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"{requested} asked for, but torch sees "
                f"{torch.cuda.device_count()} CUDA device(s)"
            )
        return requested
    raise ValueError(
        f"unknown device {requested!r}: use auto, cpu, mps, cuda or cuda:<n>"
    )
