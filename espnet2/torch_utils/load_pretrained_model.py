from pathlib import Path
from typing import Any
from typing import Union

import torch
import torch.nn
import torch.optim


def load_pretrained_model(
    pretrain_path: Union[str, Path],
    model: torch.nn.Module,
    pretrain_key: str = None,
    map_location: str = "cpu",
    ignore_not_existing_keys: bool = True,
):
    """Load a model state and set it to the model.

    Examples:
        >>> load_pretrained_model("somewhere/model.pth", model)
        >>> load_pretrained_model("somewhere/encoder.pth", model, "encoder")
    """
    if pretrain_key is None:
        obj = model
    else:

        def get_attr(obj: Any, key: str):
            """Get an nested attribute.

            >>> class A(torch.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = torch.nn.Linear(10, 10)
            >>> a = A()
            >>> assert A.linear.weight is get_attr(A, 'linear.weight')

            """
            if key.strip() == "":
                return obj
            for k in key.split("."):
                obj = getattr(obj, k)
            return obj

        obj = get_attr(model, pretrain_key)

    state_dict = obj.state_dict()
    pretrained_dict = torch.load(pretrain_path, map_location=map_location)
    if ignore_not_existing_keys:
        # Ignores the parameters not existing in the train-model
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in state_dict}
    state_dict.update(pretrained_dict)
    obj.load_state_dict(state_dict)
