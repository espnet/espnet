from typing import List

import torch
from typeguard import typechecked


@typechecked
def replace_module(
    parent_module: torch.nn.Module,
    child_name: str,
    old_module: torch.nn.Module,
    new_module: torch.nn.Module,
):
    """Replace the target module with the new module."""
    # TODO(gituser) add hook and whether requires_grad to them
    device = old_module.weight.device
    setattr(parent_module, child_name, new_module)

    # copy weight and bias from the target module
    new_module.weight = old_module.weight
    if hasattr(old_module, "bias") and old_module.bias is not None:
        new_module.bias = old_module.bias

    # move the new_module to the same device as the old_module
    new_module.to(device)


@typechecked
def check_target_module_exists(key: str, target_modules: List[str]):
    """Check if the target_modules matchs the given key."""
    return any([key.endswith(target_key) for target_key in target_modules])


@typechecked
def get_submodules(model: torch.nn.Module, key: str):
    """Return the submodules of the given key."""
    parent_module = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target_module = model.get_submodule(key)
    return parent_module, target_name, target_module
