# Copyright 2026 Carnegie Mellon University
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""Low-rank adapters for the SpeechLM decoder's linear projections."""

import math
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F


class _LoRALinear(nn.Module):
    """Keep native weight names so a Bagpiper checkpoint initializes the base."""

    def __init__(self, linear, rank, alpha, dropout):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.lora_A = nn.Parameter(self.weight.new_empty((rank, self.in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.register_buffer(
            "lora_scale",
            torch.tensor(alpha / rank, dtype=torch.float32, device=self.weight.device),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        residual = F.linear(self.dropout(inputs), self.lora_A)
        return F.linear(inputs, self.weight, self.bias) + self.lora_scale * F.linear(
            residual, self.lora_B
        )


def apply_lora(model, *, target_modules, rank, alpha, dropout=0.0):
    """Freeze SpeechLM and adapt matching decoder projections in place.

    Args:
        model: Native model with transformer blocks under ``model.layers``.
        target_modules: Exact projection leaf names, e.g. ``["q_proj", "v_proj"]``.
        rank: Positive low-rank dimension.
        alpha: Positive, finite numerator of the ``alpha / rank`` scale.
        dropout: Probability of dropping an input to the low-rank branch.

    Raises:
        ValueError: The configuration is invalid or a target does not select
            a linear projection inside the transformer blocks.
    """
    if type(rank) is not int or rank < 1:
        raise ValueError("LoRA rank must be a positive integer")
    if type(alpha) not in (float, int) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA alpha must be positive and finite")
    if type(dropout) not in (float, int) or not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")
    if (
        not isinstance(target_modules, (list, tuple))
        or not target_modules
        or not all(isinstance(name, str) and name for name in target_modules)
    ):
        raise ValueError("LoRA target_modules must be a nonempty list of names")

    layers = model.model.layers
    targets = {
        name: module
        for name, module in layers.named_modules()
        if name.rsplit(".", 1)[-1] in target_modules
    }
    missing = set(target_modules) - {name.rsplit(".", 1)[-1] for name in targets}
    if missing:
        raise ValueError(f"LoRA targets missing from decoder layers: {sorted(missing)}")
    # A plain exported weight must have one unambiguous merged value.
    uses = Counter(id(p) for _, p in model.named_parameters(remove_duplicate=False))
    for name, module in targets.items():
        if type(module) is not nn.Linear:
            raise ValueError(f"LoRA target {name} must be an ordinary nn.Linear")
        if uses[id(module.weight)] > 1:
            raise ValueError(f"LoRA target {name} has a shared weight")

    model.requires_grad_(False)
    for name, module in targets.items():
        layers.set_submodule(name, _LoRALinear(module, rank, alpha, dropout))


def missing_lora_keys(model_keys, checkpoint_keys):
    """Accept a complete adapted model or a complete base without any adapters."""
    expected = set(model_keys)
    actual = set(checkpoint_keys)
    missing = expected - actual
    extra = actual - expected
    adapter_keys = {
        key for key in expected if key.endswith((".lora_A", ".lora_B", ".lora_scale"))
    }
    if extra or (missing and missing != adapter_keys):
        raise ValueError(
            "Incompatible LoRA initializer: expected the complete base model with "
            "either all or none of its adapters. Missing base keys: "
            f"{sorted(missing - adapter_keys)[:5]}, missing adapter tensors: "
            f"{len(missing & adapter_keys)}/{len(adapter_keys)}, "
            f"unexpected keys: {sorted(extra)[:5]}"
        )
    return missing
