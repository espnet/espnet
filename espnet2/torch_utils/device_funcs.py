import dataclasses
import gc
import warnings

import numpy as np
import torch


def to_device(data, device=None, dtype=None, non_blocking=False, copy=False):
    """Change the device of object recursively"""
    if isinstance(data, dict):
        return {
            k: to_device(v, device, dtype, non_blocking, copy) for k, v in data.items()
        }
    elif dataclasses.is_dataclass(data) and not isinstance(data, type):
        return type(data)(
            *[
                to_device(v, device, dtype, non_blocking, copy)
                for v in dataclasses.astuple(data)
            ]
        )
    # maybe namedtuple. I don't know the correct way to judge namedtuple.
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(
            *[to_device(o, device, dtype, non_blocking, copy) for o in data]
        )
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device, dtype, non_blocking, copy) for v in data)
    elif isinstance(data, np.ndarray):
        return to_device(torch.from_numpy(data), device, dtype, non_blocking, copy)
    elif isinstance(data, torch.Tensor):
        if dtype is not None:
            dtype = str(dtype).removeprefix("torch.")
            cur_dtype = str(data.dtype).removeprefix("torch.")

            if not (
                ("int" in dtype and "int" in cur_dtype)
                or ("float" in dtype and "float" in cur_dtype)
            ):
                dtype = None  # avoid conversion between int and float.
            else:
                dtype = getattr(torch, dtype)

        return data.to(device, dtype, non_blocking, copy)
    else:
        return data


def force_gatherable(data, device):
    """Change object to gatherable in torch.nn.DataParallel recursively

    The difference from to_device() is changing to torch.Tensor if float or int
    value is found.

    The restriction to the returned value in DataParallel:
        The object must be
        - torch.cuda.Tensor
        - 1 or more dimension. 0-dimension-tensor sends warning.
        or a list, tuple, dict.

    """
    if isinstance(data, dict):
        return {k: force_gatherable(v, device) for k, v in data.items()}
    # DataParallel can't handle NamedTuple well
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(*[force_gatherable(o, device) for o in data])
    elif isinstance(data, (list, tuple, set)):
        return type(data)(force_gatherable(v, device) for v in data)
    elif isinstance(data, np.ndarray):
        return force_gatherable(torch.from_numpy(data), device)
    elif isinstance(data, torch.Tensor):
        if data.dim() == 0:
            # To 1-dim array
            data = data[None]
        return data.to(device)
    elif isinstance(data, float):
        return torch.tensor([data], dtype=torch.float, device=device)
    elif isinstance(data, int):
        return torch.tensor([data], dtype=torch.long, device=device)
    elif data is None:
        return None
    else:
        warnings.warn(f"{type(data)} may not be gatherable by DataParallel")
        return data


def is_out_of_memory_error(exc: BaseException) -> bool:
    """Return whether ``exc``, or an exception it was raised from, is an OOM.

    CUDA raises :class:`torch.OutOfMemoryError`; MPS raises a plain
    :class:`RuntimeError` whose message starts with "MPS backend out of
    memory". Both are covered, as is an OOM that a caller re-raised as another
    exception with ``raise ... from exc``.
    """
    oom_cls = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
    seen = 0
    while exc is not None and seen < 8:
        if isinstance(exc, oom_cls):
            return True
        if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
            return True
        exc = exc.__cause__
        seen += 1
    return False


def release_accelerator_memory() -> None:
    """Return cached accelerator memory to the device, after an OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        torch.mps.empty_cache()
