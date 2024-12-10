"""Torch utility module."""

import torch

if torch.distributed.is_available():
    from torch.distributed import ReduceOp


def recursive_sum(obj, weight: torch.Tensor, distributed: bool = False):
    """
        Recursively computes the weighted sum of elements in a nested structure.

    This function can handle various types of nested structures, including
    tuples, lists, and dictionaries, as well as PyTorch tensors. It applies
    a weight tensor to the elements and can perform operations in a
    distributed setting.

    Attributes:
        obj (Union[tuple, list, dict, torch.Tensor, None]): The input object
            which can be a nested structure containing tensors.
        weight (torch.Tensor): A 1D tensor of weights to apply to the elements
            in `obj`.
        distributed (bool): If True, perform distributed summation.

    Args:
        obj: The input object (tuple, list, dict, or tensor).
        weight: A 1D tensor of weights.
        distributed: A boolean indicating whether to perform distributed
            operations.

    Returns:
        Union[tuple, list, dict, torch.Tensor, None]: The weighted sum of the
        elements in the input object.

    Raises:
        ValueError: If the input object is of an unsupported type or if the
            dimensions of the tensors do not match.

    Examples:
        >>> import torch
        >>> weights = torch.tensor([0.1, 0.2, 0.3])
        >>> tensors = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
        >>> recursive_sum(tensors, weights)
        tensor(1.4)

        >>> weights = torch.tensor([1.0, 1.0])
        >>> data = {'a': torch.tensor([1.0]), 'b': torch.tensor([2.0])}
        >>> recursive_sum(data, weights)
        {'a': tensor(1.0), 'b': tensor(2.0)}
    """
    assert weight.dim() == 1, weight.size()
    if isinstance(obj, (tuple, list)):
        return type(obj)(recursive_sum(v, weight, distributed) for v in obj)
    elif isinstance(obj, dict):
        return {k: recursive_sum(v, weight, distributed) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        assert obj.size() == weight.size(), (obj.size(), weight.size())
        obj = (obj * weight.type(obj.dtype)).sum()
        if distributed:
            lst = [
                torch.empty_like(obj) for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(lst, obj)
            if all([torch.isnan(o) for o in lst]):
                obj = torch.sum(torch.stack(lst))
            else:
                # NOTE(wangyou): not using torch.nansum here to compensate for the
                # reduced samples.
                # This is important so that the condition-specific loss values reported
                # in Reporter will be consistent with the general loss value.
                obj = torch.nanmean(torch.stack(lst)) * len(lst)
        return obj
    elif obj is None:
        return None
    else:
        raise ValueError(type(obj))


def recursive_divide(a, b: torch.Tensor):
    """
        Divides elements of a given object by a specified tensor recursively.

    This function supports various types of input structures including
    tuples, lists, dictionaries, and PyTorch tensors. The function will
    recursively divide each element of the input `a` by the tensor `b`.
    It raises a ValueError if the shapes of `a` and `b` do not match.

    Attributes:
        None

    Args:
        a (Union[torch.Tensor, list, tuple, dict, None]): The object to be divided.
        b (torch.Tensor): The tensor by which to divide each element of `a`.

    Returns:
        Union[torch.Tensor, list, tuple, dict, None]: A new object with each
        element divided by `b`. The structure of the returned object matches
        that of the input `a`.

    Raises:
        ValueError: If the types of `a` are unsupported or if the sizes of
        `a` and `b` do not match.

    Examples:
        >>> import torch
        >>> a = torch.tensor([4.0, 8.0, 16.0])
        >>> b = torch.tensor([2.0, 4.0, 8.0])
        >>> result = recursive_divide(a, b)
        >>> print(result)
        tensor([2.0, 2.0, 2.0])

        >>> a_list = [torch.tensor([4.0]), torch.tensor([8.0])]
        >>> b_tensor = torch.tensor([2.0])
        >>> result_list = recursive_divide(a_list, b_tensor)
        >>> print(result_list)
        [tensor([2.0]), tensor([4.0])]

        >>> a_dict = {'x': torch.tensor([4.0]), 'y': torch.tensor([8.0])}
        >>> result_dict = recursive_divide(a_dict, b_tensor)
        >>> print(result_dict)
        {'x': tensor([2.0]), 'y': tensor([4.0])}

        >>> result_none = recursive_divide(None, b_tensor)
        >>> print(result_none)
        None
    """
    if isinstance(a, (tuple, list)):
        return type(a)(recursive_divide(v, b) for v in a)
    elif isinstance(a, dict):
        return {k: recursive_divide(v, b) for k, v in a.items()}
    elif isinstance(a, torch.Tensor):
        assert a.size() == b.size(), (a.size(), b.size())
        return a / b.type(a.dtype)
    elif a is None:
        return None
    else:
        raise ValueError(type(a))


def recursive_average(obj, weight: torch.Tensor, distributed: bool = False):
    """
        Calculates the weighted average of a nested structure of tensors.

    This function recursively computes the sum of the tensors in the given
    object, normalizes it by the sum of the weights, and optionally handles
    distributed settings. It is designed to work with nested structures such as
    lists, tuples, and dictionaries containing PyTorch tensors.

    Attributes:
        obj: A nested structure (list, tuple, dict) or tensor from which to compute
            the average.
        weight: A 1D tensor containing weights corresponding to the elements in
            `obj`.
        distributed: A boolean indicating whether to perform the operation in a
            distributed manner.

    Args:
        obj (Union[torch.Tensor, List, Tuple, Dict]): The input object for which
            the weighted average is calculated.
        weight (torch.Tensor): A 1D tensor of weights that matches the size of
            the tensors in `obj`.
        distributed (bool, optional): Flag to indicate if the operation should be
            performed in a distributed setting. Defaults to False.

    Returns:
        Tuple[Union[torch.Tensor, List, Dict], torch.Tensor]: A tuple containing
        the weighted average of the input object and the total weight.

    Raises:
        ValueError: If the input object `obj` is not of a valid type (tensor,
            list, tuple, or dict) or if the dimensions of the tensors do not match.

    Examples:
        >>> import torch
        >>> weights = torch.tensor([0.2, 0.3, 0.5])
        >>> tensors = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        >>> average, total_weight = recursive_average(tensors, weights)
        >>> print(average)  # Output: tensor(2.0)
        >>> print(total_weight)  # Output: tensor(1.0)

    Note:
        This function is primarily intended for use in machine learning contexts,
        where weighted averages of loss values or predictions are common.
    """
    obj = recursive_sum(obj, weight, distributed)
    weight = weight.sum()
    if distributed:
        torch.distributed.all_reduce(weight, op=ReduceOp.SUM)
    # Normalize weight to be sum-to-1
    obj = recursive_divide(obj, weight)
    return obj, weight
