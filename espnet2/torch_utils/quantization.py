"""Dynamic quantization with a usable quantized engine."""

import logging

import torch


def ensure_quantized_engine() -> str:
    """Select a quantized engine when PyTorch has left it unset.

    The macOS wheels of PyTorch ship with qnnpack built in but leave
    ``torch.backends.quantized.engine`` at ``"none"``, so ``quantize_dynamic``
    produces a model whose first ``quantized::linear_prepack`` fails with
    "Didn't find engine for operation ... NoQEngine". When the engine is unset
    and at least one engine is available, the first one is selected.

    Returns:
        The engine in use after the call.
    """
    if torch.backends.quantized.engine == "none":
        engines = [e for e in torch.backends.quantized.supported_engines if e != "none"]
        if engines:
            logging.info(f"Using the quantized engine {engines[0]}")
            torch.backends.quantized.engine = engines[0]
    return torch.backends.quantized.engine


def quantize_dynamic(*args, **kwargs):
    """``torch.quantization.quantize_dynamic`` after :func:`ensure_quantized_engine`."""
    ensure_quantized_engine()
    return torch.quantization.quantize_dynamic(*args, **kwargs)
