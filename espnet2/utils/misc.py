import inspect
from pathlib import Path
from typing import Dict, Any

from pytypes import typechecked
import numpy as np


@typechecked
def get_arguments_from_func(func) -> Dict[str, Any]:
    """

    Examples:
        >>> def func(a, b=3):  pass
        >>> get_arguments_from_func(func, 'output.yaml')

    """
    def yaml_serializable(value):
        print(value)
        if value is inspect.Parameter.empty or value is None:
            return None
        # Maybe named_tuple?
        elif isinstance(value, tuple) and type(value) is not tuple:
            return yaml_serializable(vars(value))
        elif isinstance(value, dict):
            assert all(isinstance(k, str) for k in value), \
                f'dict keys must be str: {list(value)}'
            return {k: yaml_serializable(v) for k, v in value.items()}
        elif isinstance(value, tuple):
            return yaml_serializable(list(value))
        elif isinstance(value, list):
            return [yaml_serializable(v) for v in value]
        elif isinstance(value, np.ndarray):
            assert value.ndim == 1, value.shape
            return yaml_serializable(value.tolist())
        elif isinstance(value, Path):
            return str(value)
        elif isinstance(value, (float, int, complex, bool, str, bytes)):
            return value
        else:
            return None

    # params: An ordered mapping of inspect.Parameter
    params = inspect.signature(func).parameters
    data = {p.name: yaml_serializable(p.default) for p in params.values()}
    return data
