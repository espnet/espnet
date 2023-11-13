"""Definition of the low-rank adaptation (LoRA) for large models.

References:
    1. LoRA: Low-Rank Adaptation of Large Language Models
       (https://arxiv.org/pdf/2106.09685.pdf)
    2. https://github.com/microsoft/LoRA.git
    3. https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py

"""
from typing import List

import torch
from typeguard import check_argument_types

try:
    import loralib as lora
except Exception:
    lora = None


def create_lora_adapter(
    model: torch.nn.Module,
    rank: int = 8,
    alpha: int = 8,
    dropout_rate: float = 0.0,
    target_modules: List[str] = ["query"],
    bias_type: str = "none",
):
    """Create LoRA adapter for the base model.

    See: https://arxiv.org/pdf/2106.09685.pdf

    Args:
        model (torch.nn.Module): Base model to be adapted.
        rank (int): Rank of LoRA matrices. Defaults to 8.
        alpha (int): Constant number for LoRA scaling. Defaults to 8.
        dropout_rate (float): Dropout probability for LoRA layers. Defaults to 0.0.
        target_modules (List[str]): List of module(s) to apply LoRA adaptation.
            e.g. ["query", "key", "value"] for all layers,
            while ["encoder.encoders.blocks.0.attn.key"] for a specific layer.
        bias_type (str): Bias training type for LoRA adaptaion, can be
            one of ["none", "all", "lora_only"].
            "none" means not training any bias vectors;
            "all" means training all bias vectors, include LayerNorm biases;
            "lora_only" means only training bias vectors in LoRA adapted modules.

    Returns:
        torch.nn.Module: The LoRA adapted model.
    """
    assert check_argument_types()

    if lora is None:
        raise RuntimeError(
            "Requiring loralib. Install loralib following: "
            "https://github.com/microsoft/LoRA"
        )

    is_traget_module_exists = False
    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if not check_target_module_exists(key, target_modules):
            continue

        is_traget_module_exists = True

        parent_module, target_name, target_module = get_submodules(model, key)
        if not isinstance(target_module, lora.LoRALayer):
            new_module = create_new_module(target_module, rank, alpha, dropout_rate)
            replace_module(parent_module, target_name, target_module, new_module)
        else:
            continue

    if not is_traget_module_exists:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model."
        )

    lora.mark_only_lora_as_trainable(model, bias_type)


def check_target_module_exists(key: str, target_modules: List[str]):
    """Check if the target_modules matchs the given key."""

    return any([key.endswith(target_key) for target_key in target_modules])


def get_submodules(model: torch.nn.Module, key: str):
    """Return the submodules of the given key."""

    parent_module = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target_module = model.get_submodule(key)
    return parent_module, target_name, target_module


def create_new_module(
    target_module: torch.nn.Module, rank: int, alpha: int, dropout_rate: float
):
    """Create a new module for the given target module."""

    bias = hasattr(target_module, "bias") and target_module.bias is not None

    if isinstance(target_module, torch.nn.Embedding):
        new_module = lora.Embedding(
            target_module.num_embeddings,
            target_module.embedding_dim,
            r=rank,
            lora_alpha=alpha,
        )
    elif isinstance(target_module, torch.nn.Linear):
        new_module = lora.Linear(
            target_module.in_features,
            target_module.out_features,
            bias=bias,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout_rate,
        )
    else:
        raise ValueError(
            f"Target module {target_module} is not supported. "
            f"Currently, only `torch.nn.Embedding`, `torch.nn.Conv2d` "
            f"`torch.nn.Linear` and are supported."
        )

    return new_module


def replace_module(
    parent_module: torch.nn.Module,
    child_name: str,
    old_module: torch.nn.Module,
    new_module: torch.nn.Module,
):
    """Replace the target module with the new module."""

    device = old_module.weight.device
    setattr(parent_module, child_name, new_module)

    # copy weight and bias from the target module
    new_module.weight = old_module.weight
    if hasattr(old_module, "bias") and old_module.bias is not None:
        new_module.bias = old_module.bias

    # move the new_module to the same device as the old_module
    new_module.to(device)
