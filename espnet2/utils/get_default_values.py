import inspect
from pathlib import Path

import torch
import numpy as np


class Invalid:
    """Marker object for not serializable-object"""


def get_defaut_values(func):
    """

    Examples:
        >>> def func(a, b=3):  pass
        >>> get_defaut_values(func, 'output.yaml')

    """
    def yaml_serializable(value):
        # isinstance(x, tuple) includes namedtuple, so type is used here
        if type(value) is tuple:
            return yaml_serializable(list(value))
        elif isinstance(value, dict):
            assert all(isinstance(k, str) for k in value), \
                f'dict keys must be str: {list(value)}'
            retval = {}
            for k, v in value.items():
                if not isinstance(k, str):
                    return Invalid
                v2 = yaml_serializable(v)
                # Register only valid object
                if v2 not in (Invalid, inspect.Parameter.empty):
                    retval[k] = v2
            return retval
        elif isinstance(value, list):
            retval = []
            for v in value:
                v2 = yaml_serializable(v)
                # If any elements in the list are invalid,
                # the list also becomes invalid
                if v2 is Invalid:
                    return Invalid
                else:
                    retval.append(v2)
            return retval
        elif isinstance(value, torch.Tensor):
            return yaml_serializable(value.cpu().numpy())
        elif isinstance(value, np.ndarray):
            assert value.ndim == 1, value.shape
            return yaml_serializable(value.tolist())
        elif isinstance(value, Path):
            return str(value)
        elif value in (inspect.Parameter.empty, None):
            return value
        elif isinstance(value, (float, int, complex, bool, str, bytes)):
            return value
        else:
            return Invalid

    # params: An ordered mapping of inspect.Parameter
    params = inspect.signature(func).parameters
    data = {p.name: p.default for p in params.values()}
    # Remove not yaml-serializable object
    data = yaml_serializable(data)
    return data
