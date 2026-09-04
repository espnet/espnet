# The helpers in this file are adapted from microsoft/LoRA
# (https://github.com/microsoft/LoRA):
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License (MIT).
"""Helpers for LoRA-style PEFT layers in ESPnet."""

from typing import Dict, Set

import torch
import torch.nn as nn

from espnet2.layers.lora.layers import LoRALayer


def adapter_param_names(model: nn.Module) -> Set[str]:
    """Return the names of every trainable adapter parameter in ``model``.

    Vanilla LoRA names its parameters ``lora_A`` / ``lora_B``, but the other
    backends do not: SVFT trains ``m_entries``/``gate``, SSVD trains
    ``s``/``gate``/``K_vec``. Matching on the ``"lora_"`` substring alone would
    therefore miss them entirely. Anything that lives inside a
    :class:`LoRALayer` and is not the frozen ``weight``/``bias`` counts.
    """
    names = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, LoRALayer):
            continue
        prefix = f"{module_name}." if module_name else ""
        for param_name, _ in module.named_parameters(recurse=False):
            if param_name in ("weight", "bias"):
                continue
            names.add(prefix + param_name)
    # Keep the historical substring rule as a fallback so adapters that are not
    # LoRALayer subclasses still behave as before.
    names.update(n for n, _ in model.named_parameters() if "lora_" in n)
    return names


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze every parameter that is not an adapter parameter.

    Args:
        model: The model to mutate in place.
        bias: Which bias parameters to keep trainable. One of:
            ``"none"`` (default) keeps no bias trainable;
            ``"all"`` keeps every bias in the model trainable;
            ``"lora_only"`` keeps only the biases of LoRA-adapted modules.
    """
    adapter_names = adapter_param_names(model)
    for n, p in model.named_parameters():
        if n not in adapter_names:
            p.requires_grad = False
    if bias == "none":
        return
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
        return
    if bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
        return
    raise NotImplementedError(f"Unknown bias mode: {bias}")


def lora_state_dict(model: nn.Module, bias: str = "none") -> Dict[str, torch.Tensor]:
    """Return a state dict containing only the adapter (and optionally bias) keys.

    Covers every backend in :mod:`espnet2.layers.lora`, not just the ones whose
    parameters happen to be named ``lora_*`` -- see :func:`adapter_param_names`.
    """
    my_state_dict = model.state_dict()
    adapter_keys = adapter_param_names(model)
    if bias not in ("none", "all", "lora_only"):
        raise NotImplementedError(f"Unknown bias mode: {bias}")
    to_return = {k: my_state_dict[k] for k in my_state_dict if k in adapter_keys}
    if bias == "all":
        to_return.update(
            {k: my_state_dict[k] for k in my_state_dict if k.endswith("bias")}
        )
    elif bias == "lora_only":
        for k in adapter_keys:
            bias_name = k.rsplit(".", 1)[0] + ".bias" if "." in k else "bias"
            if bias_name in my_state_dict:
                to_return[bias_name] = my_state_dict[bias_name]
    return to_return
