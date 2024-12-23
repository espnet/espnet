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
    """
        Replace the target module within a parent module with a new module.

    This function identifies the specified child module within the parent module and
    replaces it with a new module. It also copies the weights and biases from the
    old module to the new module and ensures that the new module is moved to the
    same device as the old module.

    Args:
        parent_module (torch.nn.Module): The parent module containing the child
            module to be replaced.
        child_name (str): The name of the child module to be replaced.
        old_module (torch.nn.Module): The module to be replaced.
        new_module (torch.nn.Module): The new module that will replace the old
            module.

    Raises:
        AttributeError: If the specified child_name does not exist in the
            parent_module.

    Note:
        - This function currently does not handle the addition of hooks or the
          requires_grad attribute for the new module.

    Examples:
        >>> parent = torch.nn.Sequential(
        ...     torch.nn.Linear(10, 5),
        ...     torch.nn.ReLU()
        ... )
        >>> old_module = parent[0]
        >>> new_module = torch.nn.Linear(10, 5)
        >>> replace_module(parent, '0', old_module, new_module)
        >>> assert isinstance(parent[0], torch.nn.Linear)  # New module is in place
    """
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
    """
        Check if the target_modules matches the given key.

    This function checks whether any of the target module names in the
    `target_modules` list match the specified `key`. It returns True if
    there is a match; otherwise, it returns False.

    Args:
        key (str): The key to check against the target modules.
        target_modules (List[str]): A list of target module names to match.

    Returns:
        bool: True if the key matches any of the target modules; False otherwise.

    Examples:
        >>> check_target_module_exists("conv1", ["conv1", "conv2"])
        True
        >>> check_target_module_exists("fc", ["conv1", "conv2"])
        False
    """
    return any([key.endswith(target_key) for target_key in target_modules])


@typechecked
def get_submodules(model: torch.nn.Module, key: str):
    """
        Retrieve the submodules of a specified key from the given model.

    This function navigates through the hierarchical structure of a PyTorch
    module to locate and return the parent module, the name of the target
    submodule, and the target submodule itself based on the provided key.

    Args:
        model (torch.nn.Module): The parent model from which to retrieve the
            submodules.
        key (str): The key representing the path to the target submodule in the
            model, using dot notation.

    Returns:
        tuple: A tuple containing:
            - parent_module (torch.nn.Module): The parent module containing the
              target submodule.
            - target_name (str): The name of the target submodule.
            - target_module (torch.nn.Module): The target submodule itself.

    Examples:
        >>> import torch
        >>> class MyModel(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.layer1 = torch.nn.Linear(10, 5)
        ...         self.layer2 = torch.nn.Linear(5, 2)
        ...
        ...     def get_submodule(self, key):
        ...         return getattr(self, key)

        >>> model = MyModel()
        >>> parent, name, module = get_submodules(model, "layer2")
        >>> print(name)  # Output: layer2
        >>> print(module)  # Output: Linear(in_features=5, out_features=2, bias=True)
    """
    parent_module = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target_module = model.get_submodule(key)
    return parent_module, target_name, target_module
