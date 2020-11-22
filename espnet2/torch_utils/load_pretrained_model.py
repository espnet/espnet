from typing import Any

import torch
import torch.nn
import torch.optim


def load_pretrained_model(
    init_param: str,
    model: torch.nn.Module,
    map_location: str = "cpu",
):
    """Load a model state and set it to the model.

    Args:
        init_param: <file_path>:<src_key>:<dst_key>:<exclude_Keys>

    Examples:
        >>> load_pretrained_model("somewhere/model.pth", model)
        >>> load_pretrained_model("somewhere/model.pth:decoder:decoder", model)
        >>> load_pretrained_model("somewhere/model.pth:decoder:decoder:", model)
        >>> load_pretrained_model(
        ...     "somewhere/model.pth:decoder:decoder:decoder.embed", model
        ... )
        >>> load_pretrained_model("somewhere/decoder.pth::decoder", model)
    """
    sps = init_param.split(":", 4)
    if len(sps) == 4:
        path, src_key, dst_key, excludes = sps
    elif len(sps) == 3:
        path, src_key, dst_key = sps
        excludes = None
    elif len(sps) == 2:
        path, src_key = sps
        dst_key, excludes = None, None
    else:
        (path,) = sps
        src_key, dst_key, excludes = None, None, None
    if src_key == "":
        src_key = None
    if dst_key == "":
        dst_key = None

    if dst_key is None:
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

        obj = get_attr(model, dst_key)

    src_state = torch.load(path, map_location=map_location)
    if excludes is not None:
        for e in excludes.split(","):
            src_state = {k: v for k, v in src_state.items() if not k.startswith(e)}

    if src_key is not None:
        src_state = {
            k[len(src_key) + 1 :]: v
            for k, v in src_state.items()
            if k.startswith(src_key)
        }

    dst_state = obj.state_dict()
    dst_state.update(src_state)
    obj.load_state_dict(dst_state)
