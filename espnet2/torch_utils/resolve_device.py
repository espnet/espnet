"""Where a model is built: ``"auto"`` resolved to a device torch has."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Accelerators that publish a device count on their own torch namespace, so an
# index asked for with them can be checked here: ``torch.cuda``, ``torch.xpu``,
# ``torch.npu`` (torch_npu), ``torch.mlu`` and the like. ``cpu`` and ``mps``
# are handled by name, below, and have no index to check.
_INDEXED = ("cuda", "xpu", "npu", "mlu", "musa", "hcu")

_USE = "auto, cpu, mps, or an accelerator name with an index (cuda:0, npu:3)"


def _accelerator_device_type() -> str | None:
    """Which accelerator torch sees, or ``None`` on a CPU-only host.

    ``torch.accelerator`` (torch >= 2.5) is the device-agnostic query, so an
    NPU, XPU or MLU host is recognised without this module naming a vendor.
    A build too old to have it reports CUDA, which is what ``auto`` meant
    before there was more than one accelerator to choose between.
    """
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None:
        return "cuda" if torch.cuda.is_available() else None
    if not accelerator.is_available():
        return None
    device_type = getattr(accelerator.current_accelerator(), "type", None)
    # MPS is used only when asked for by name; see resolve_device(). Reporting
    # it here would make ``auto`` pick it on an Apple-silicon Mac.
    return None if device_type == "mps" else device_type


def _device_count(device_type: str) -> int:
    """How many ``device_type`` devices this machine has; 0 without a namespace."""
    counter = getattr(getattr(torch, device_type, None), "device_count", None)
    return counter() if callable(counter) else 0


def resolve_device(requested: str = "auto") -> str:
    """Return the device a model is built on, for ``"auto"`` or a named one.

    ``"auto"`` is the accelerator torch sees -- CUDA, NPU, XPU, whichever
    ``torch.accelerator`` reports -- else the CPU, so a caller need not know
    what this machine has. MPS is not chosen on its own: it is faster than the
    CPU for some models' encoders and slower for a CTC prefix search, and
    until that is measured per model it is used only when asked for by name.

    A named device is checked against what torch has, so a typo or a device
    this machine lacks fails here, before a model is downloaded and built,
    rather than in the first forward pass.

    Args:
        requested: ``"auto"``, ``"cpu"``, ``"mps"``, or an accelerator name
            with or without an index, e.g. ``"cuda"``, ``"cuda:1"``,
            ``"npu"``, ``"npu:3"``.

    Returns:
        A device torch accepts: never ``"auto"``.

    Raises:
        ValueError: If ``requested`` names no device, or one this machine
            does not have.

    Examples:
        >>> resolve_device()            # a machine without an accelerator
        'cpu'
        >>> resolve_device("cuda:0")    # on one with a GPU
        'cuda:0'
        >>> resolve_device("npu:0")     # on one with an Ascend NPU
        'npu:0'
        >>> resolve_device("gpu")
        Traceback (most recent call last):
        ValueError: unknown device 'gpu': use auto, cpu, mps, or an ...
    """
    if requested == "auto":
        device = _accelerator_device_type() or "cpu"
        logger.info("device auto: %s", device)
        return device
    if requested == "cpu":
        return requested
    if requested == "mps":
        # torch.backends.mps exists since torch 1.12; a build without it has
        # no MPS device either
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ValueError(
                "mps asked for, but torch has no MPS device here; pass "
                "device='cpu', or 'auto'"
            )
        return requested

    device_type, _, index_text = requested.partition(":")
    if device_type not in _INDEXED or (index_text and not index_text.isdigit()):
        raise ValueError(f"unknown device {requested!r}: use {_USE}")
    count = _device_count(device_type)
    if count == 0:
        # CUDA keeps its capitals in the message it has always produced.
        shown = "CUDA" if device_type == "cuda" else device_type
        raise ValueError(
            f"{requested} asked for, but torch sees no {shown} device; "
            "pass device='cpu', or 'auto'"
        )
    if index_text and int(index_text) >= count:
        raise ValueError(
            f"{requested} asked for, but torch sees {count} {device_type} device(s)"
        )
    return requested
