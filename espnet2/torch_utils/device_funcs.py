import dataclasses
import warnings

import numpy as np
import torch


def to_device(data, device=None, dtype=None, non_blocking=False, copy=False):
    """
    Change the device of object recursively.

    This function recursively moves a given data structure (which can include
    dictionaries, lists, tuples, NumPy arrays, and PyTorch tensors) to the
    specified device (e.g., CPU or GPU). It can also optionally change the
    data type of the tensors.

    Args:
        data: The input data, which can be a tensor, numpy array, list,
            tuple, dictionary, or a dataclass.
        device: The target device to move the data to. This can be a
            string (e.g., 'cpu', 'cuda:0') or a torch.device object.
        dtype: The desired data type to convert the tensors to (e.g.,
            torch.float, torch.int). If None, the dtype will remain unchanged.
        non_blocking: If True and the source is in pinned memory,
            the copy will be non-blocking. Defaults to False.
        copy: If True, a new tensor will be created, and the original
            tensor will not be modified. Defaults to False.

    Returns:
        The input data, moved to the specified device with the desired
        dtype, if applicable.

    Examples:
        >>> import torch
        >>> data = torch.tensor([1, 2, 3])
        >>> to_device(data, device='cuda:0')
        tensor([1, 2, 3], device='cuda:0')

        >>> data = {'a': torch.tensor([1]), 'b': [torch.tensor([2]),
        ...     torch.tensor([3])]}
        >>> to_device(data, device='cuda:0')
        {'a': tensor([1], device='cuda:0'),
         'b': [tensor([2], device='cuda:0'), tensor([3], device='cuda:0')]}

    Note:
        If `dtype` is specified, conversion between int and float types
        is avoided to prevent unexpected behavior.

    Raises:
        ValueError: If an unsupported data type is provided.

    Todo:
        - Add support for additional data structures if necessary.
    """
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
    """
        Change object to gatherable in torch.nn.DataParallel recursively.

    This function modifies the input data structure to ensure that it is
    suitable for use with `torch.nn.DataParallel`. It will convert numerical
    values (integers and floats) into `torch.Tensor` objects and will ensure
    that tensors are moved to the specified device. The resulting structure
    must conform to the requirements of `DataParallel`, which include being
    a `torch.cuda.Tensor` and having at least one dimension.

    Args:
        data: The input data, which can be a tensor, list, tuple, set,
            dictionary, or a numpy array.
        device: The target device (e.g., 'cuda:0' or 'cpu') to which
            the tensors should be moved.

    Returns:
        A structure of the same type as `data`, with all applicable elements
        converted to tensors and moved to the specified device.

    Raises:
        UserWarning: If an element is of a type that may not be gatherable
        by `DataParallel`.

    Examples:
        >>> import torch
        >>> data = [1.0, 2.0, 3.0]
        >>> device = 'cuda:0'  # Example device
        >>> gatherable_data = force_gatherable(data, device)
        >>> print(gatherable_data)
        tensor([1., 2., 3.], device='cuda:0')

        >>> data_dict = {'a': 1, 'b': 2.0}
        >>> gatherable_data_dict = force_gatherable(data_dict, device)
        >>> print(gatherable_data_dict)
        {'a': tensor([1], device='cuda:0'), 'b': tensor([2.], device='cuda:0')}
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
