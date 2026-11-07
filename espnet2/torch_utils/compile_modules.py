"""Compile chosen submodules of a model with torch.compile."""

import logging
from typing import Dict, Optional, Sequence

import torch


def compile_modules(
    model: torch.nn.Module,
    names: Sequence[str],
    conf: Optional[Dict] = None,
):
    """Compile the named submodules of ``model`` in place with torch.compile.

    Each name is a dotted path below ``model``, e.g. "encoder" or
    "frontend.logmel". This function calls ``nn.Module.compile()`` on each one,
    which compiles the module in place: the parameter names and the state dict
    do not change, so a checkpoint saved from the compiled model loads into an
    uncompiled one. Compiling the modules that do the work, and not the whole
    model, leaves the loss and the statistics in eager mode.

    Args:
        model: The model.
        names: Dotted paths of the submodules to compile.
        conf: Keyword arguments of torch.compile, e.g. {"mode": "max-autotune"}.

    Raises:
        ValueError: A name is not a submodule of ``model``.
    """
    for name in names:
        try:
            module = model.get_submodule(name)
        except AttributeError as e:
            raise ValueError(
                f"{name!r} is not a submodule of {type(model).__name__}"
            ) from e
        module.compile(**(conf or {}))
        logging.info(f"Compiled {name} ({type(module).__name__}) with {conf or {}}")
