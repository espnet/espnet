from pathlib import Path
from typing import Any
from typing import Union

import torch


def load_pretrained_model(
        model: torch.nn.Module,
        pretrain_path: Union[str, Path],
        pretrain_key: str = None,
        map_location: str = "cpu",
):
    """Load pre-trained model

    Examples:
        >>> load_pretrained_model(model, "somewhere/model.pth")
        >>> load_pretrained_model(model, "somewhere/enc.pth", "enc")
    """
    if pretrain_key is None:
        obj = model
    else:

        def get_attr(obj: Any, key: str):
            """

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
    # Ignores the parameters not existing in the train-model
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if k in state_dict
    }
    state_dict.update(pretrained_dict)
    obj.load_state_dict(state_dict)