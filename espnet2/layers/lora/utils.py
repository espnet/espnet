"""Helpers for LoRA-style PEFT layers in ESPnet."""

from typing import Dict

import torch
import torch.nn as nn

from espnet2.layers.lora.layers import LoRALayer


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze every parameter that is not a LoRA parameter.

    Args:
        model: The model to mutate in place.
        bias: Which bias parameters to keep trainable. One of:
            ``"none"`` (default) keeps no bias trainable;
            ``"all"`` keeps every bias in the model trainable;
            ``"lora_only"`` keeps only the biases of LoRA-adapted modules.
    """
    for n, p in model.named_parameters():
        if "lora_" not in n:
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
            if (
                isinstance(m, LoRALayer)
                and hasattr(m, "bias")
                and m.bias is not None
            ):
                m.bias.requires_grad = True
        return
    raise NotImplementedError(f"Unknown bias mode: {bias}")


def lora_state_dict(
    model: nn.Module, bias: str = "none"
) -> Dict[str, torch.Tensor]:
    """Return a state dict containing only the adapter (and optionally bias) keys."""
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    if bias == "all":
        return {
            k: my_state_dict[k]
            for k in my_state_dict
            if "lora_" in k or "bias" in k
        }
    if bias == "lora_only":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    raise NotImplementedError(f"Unknown bias mode: {bias}")
