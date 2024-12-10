import logging
from typing import Any, Dict, Union

import torch
import torch.nn
import torch.optim


def filter_state_dict(
    dst_state: Dict[str, Union[float, torch.Tensor]],
    src_state: Dict[str, Union[float, torch.Tensor]],
):
    """
        Filter name, size mismatch instances between two state dictionaries.

    This function compares two state dictionaries, `dst_state` and `src_state`,
    and filters out any entries from `src_state` that either do not exist in
    `dst_state` or have a size mismatch. The filtered state dictionary containing
    only the matching entries is returned.

    Args:
        dst_state (Dict[str, Union[float, torch.Tensor]]): Reference state
            dictionary for filtering.
        src_state (Dict[str, Union[float, torch.Tensor]]): Target state
            dictionary for filtering.

    Returns:
        Dict[str, Union[float, torch.Tensor]]: A new dictionary containing
        only the entries from `src_state` that match the keys and sizes
        of `dst_state`.

    Raises:
        None

    Examples:
        >>> dst = {'layer1.weight': torch.zeros(2, 2), 'layer2.weight': torch.zeros(3, 3)}
        >>> src = {'layer1.weight': torch.ones(2, 2), 'layer2.weight': torch.ones(2, 2)}
        >>> filtered = filter_state_dict(dst, src)
        >>> filtered
        {'layer1.weight': tensor([[1., 1.],
                                   [1., 1.]])}

    Note:
        This function is primarily used when loading pretrained models to ensure
        compatibility between the source and destination state dictionaries.
    """
    match_state = {}
    for key, value in src_state.items():
        if key in dst_state and (dst_state[key].size() == src_state[key].size()):
            match_state[key] = value
        else:
            if key not in dst_state:
                logging.warning(
                    f"Filter out {key} from pretrained dict"
                    + " because of name not found in target dict"
                )
            else:
                logging.warning(
                    f"Filter out {key} from pretrained dict"
                    + " because of size mismatch"
                    + f"({dst_state[key].size()}-{src_state[key].size()})"
                )
    return match_state


def load_pretrained_model(
    init_param: str,
    model: torch.nn.Module,
    ignore_init_mismatch: bool,
    map_location: str = "cpu",
):
    """
        Load a model state and set it to the model.

    This function loads a pretrained model's state dictionary from a specified file
    and applies it to the provided model. It supports various configurations for
    selecting source and destination keys, as well as excluding certain keys.

    Args:
        init_param: A string formatted as <file_path>:<src_key>:<dst_key>:<exclude_Keys>.
            - file_path: Path to the pretrained model file.
            - src_key: The key in the source state dictionary to load.
            - dst_key: The key in the model to which the state should be loaded.
            - exclude_Keys: A comma-separated list of keys to exclude from loading.
        model: The model instance (torch.nn.Module) to which the state will be applied.
        ignore_init_mismatch: A boolean indicating whether to ignore mismatches
            during initialization.
        map_location: The location to map the model parameters (default is "cpu").

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
    if ignore_init_mismatch:
        src_state = filter_state_dict(dst_state, src_state)
    dst_state.update(src_state)
    obj.load_state_dict(dst_state)
