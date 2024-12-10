# Copyright 2024 Jiatong Shi
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted from https://github.com/facebookresearch/encodec

# Original license as follows:
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/encodec/tree/main


"""Torch distributed utilities."""
from typing import Dict, Iterable, List

import torch


def rank():
    """
    Retrieve the rank of the current process in the distributed setting.

    This function checks if the PyTorch distributed backend is initialized.
    If it is, it returns the rank (ID) of the current process within the
    distributed group. If distributed processing is not initialized, it
    returns 0, indicating the default rank for single-process execution.

    Returns:
        int: The rank of the current process. Returns 0 if not in a
        distributed environment.

    Examples:
        >>> import torch
        >>> torch.distributed.init_process_group(backend='nccl')
        >>> rank()
        0  # This will return the rank of the current process.

    Note:
        This function is intended to be used within a context where
        distributed processing is configured using PyTorch's
        `torch.distributed` module.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def world_size():
    """
    Get the total number of processes in the current distributed group.

    This function checks if the PyTorch distributed environment is initialized.
    If it is, it returns the total number of processes (world size) that are part
    of the distributed group. If not, it returns 1, indicating that there is
    only a single process.

    Returns:
        int: The total number of processes in the distributed group. Returns 1
        if the distributed environment is not initialized.

    Examples:
        >>> import torch
        >>> if torch.distributed.is_initialized():
        ...     print(world_size())
        ... else:
        ...     print(world_size())  # Output: 1

    Note:
        This function is typically used in conjunction with other distributed
        utilities to facilitate multi-process training.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def is_distributed():
    """
        Check if the current process is part of a distributed training setup.

    A process is considered to be in a distributed setup if the world size
    is greater than 1, which indicates that multiple processes are
    communicating with each other.

    Returns:
        bool: True if the world size is greater than 1, indicating that
        distributed training is in use; otherwise, False.

    Examples:
        >>> is_distributed()
        True  # If running in a distributed environment
        >>> is_distributed()
        False  # If running in a non-distributed environment
    """
    return world_size() > 1


def all_reduce(tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM):
    """
    Perform a collective operation to reduce the tensor across all processes.

    This function uses PyTorch's distributed backend to perform an all-reduce
    operation on the provided tensor. The operation can be specified using the
    `op` argument, which defaults to summing the values across all processes.

    Args:
        tensor (torch.Tensor): The tensor to be reduced across all processes.
        op (torch.distributed.ReduceOp, optional): The reduction operation to
            perform. Default is `torch.distributed.ReduceOp.SUM`.

    Returns:
        torch.Tensor: The reduced tensor, updated in-place.

    Raises:
        RuntimeError: If called when not in a distributed environment.

    Examples:
        >>> import torch
        >>> from your_module import all_reduce
        >>> torch.distributed.init_process_group(backend='nccl')
        >>> tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        >>> all_reduce(tensor)
        >>> print(tensor)  # Outputs the sum of the tensor across all processes.

    Note:
        This function should only be called when the distributed process group
        is initialized. If the environment is not set up for distributed
        processing, a RuntimeError will be raised.
    """
    if is_distributed():
        return torch.distributed.all_reduce(tensor, op)


def _is_complex_or_float(tensor):
    return torch.is_floating_point(tensor) or torch.is_complex(tensor)


def _check_number_of_params(params: List[torch.Tensor]):
    # utility function to check that the number of params in all workers is the same,
    # and thus avoid a deadlock with distributed all reduce.
    if not is_distributed() or not params:
        return
    # print('params[0].device ', params[0].device)
    tensor = torch.tensor([len(params)], device=params[0].device, dtype=torch.long)
    all_reduce(tensor)
    if tensor.item() != len(params) * world_size():
        # If not all the workers have the same number, for at least one of them,
        # this inequality will be verified.
        raise RuntimeError(
            f"Mismatch in number of params: ours is {len(params)}, "
            "at least one worker has a different one."
        )


def broadcast_tensors(tensors: Iterable[torch.Tensor], src: int = 0):
    """
    Broadcast the tensors from the given parameters to all workers.

    This function ensures that all workers in a distributed setting have the same
    model parameters to start with. It checks if the distributed environment is
    initialized and filters the tensors to only include complex or floating-point
    types. It also verifies that all workers have the same number of parameters
    to prevent deadlocks during the broadcast operation.

    Args:
        tensors (Iterable[torch.Tensor]): An iterable of tensors to be broadcasted.
        src (int, optional): The source rank from which to broadcast the tensors.
                             Defaults to 0.

    Returns:
        None: This function does not return a value.

    Raises:
        RuntimeError: If there is a mismatch in the number of parameters across
                       workers.

    Examples:
        >>> if is_distributed():
        >>>     tensor1 = torch.tensor([1.0], requires_grad=True)
        >>>     tensor2 = torch.tensor([2.0], requires_grad=True)
        >>>     broadcast_tensors([tensor1, tensor2], src=0)
        >>>     # All workers will now have tensor1 and tensor2 with the same values.

    Note:
        This function is intended for use in a PyTorch distributed environment
        and requires the `torch.distributed` package to be initialized.
    """
    if not is_distributed():
        return
    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    _check_number_of_params(tensors)
    handles = []
    for tensor in tensors:
        # src = int(rank()) # added code
        handle = torch.distributed.broadcast(tensor.data, src=src, async_op=True)
        handles.append(handle)
    for handle in handles:
        handle.wait()


def sync_buffer(buffers, average=True):
    """
        Sync gradient for buffers across distributed workers.

    This function synchronizes the gradients of the provided buffers among all
    workers in a distributed training environment. If the `average` parameter is
    set to `True`, the gradients are averaged across all workers; otherwise,
    the buffers are broadcasted from the source worker.

    Attributes:
        buffers (Iterable[torch.Tensor]): A collection of tensors whose gradients
            need to be synchronized.
        average (bool): A flag indicating whether to average the gradients
            (True) or broadcast them (False). Defaults to True.

    Args:
        buffers (Iterable[torch.Tensor]): A collection of tensors to synchronize.
        average (bool): If True, average the gradients; if False, broadcast them.

    Returns:
        None: The function operates in place on the provided buffers.

    Raises:
        RuntimeError: If the operation is attempted in a non-distributed
            environment.

    Examples:
        >>> import torch
        >>> from some_module import sync_buffer
        >>> buffer1 = torch.tensor([1.0, 2.0], requires_grad=True)
        >>> buffer2 = torch.tensor([3.0, 4.0], requires_grad=True)
        >>> sync_buffer([buffer1, buffer2], average=True)
        >>> print(buffer1)  # Buffers are averaged across workers
        >>> sync_buffer([buffer1, buffer2], average=False)
        >>> print(buffer1)  # Buffers are broadcasted from the source worker

    Note:
        This function requires that PyTorch's distributed package is properly
        initialized and that the number of buffers is consistent across all
        workers to avoid deadlocks.

    Todo:
        - Add support for non-float tensors if needed.
    """
    if not is_distributed():
        return
    handles = []
    for buffer in buffers:
        if torch.is_floating_point(buffer.data):
            if average:
                handle = torch.distributed.all_reduce(
                    buffer.data, op=torch.distributed.ReduceOp.SUM, async_op=True
                )
            else:
                handle = torch.distributed.broadcast(buffer.data, src=0, async_op=True)
            handles.append((buffer, handle))
    for buffer, handle in handles:
        handle.wait()
        if average:
            buffer.data /= world_size


def sync_grad(params):
    """
    Synchronize gradients across all distributed processes.

    This function serves as a simpler alternative to
    DistributedDataParallel, providing a straightforward method for
    synchronizing gradients without relying on complex mechanisms.
    It is especially useful for simple models where it can perform
    as efficiently as DistributedDataParallel. Call this function on
    your model parameters after invoking the backward pass to ensure
    that gradients are synchronized across all workers.

    Args:
        params (Iterable[torch.Tensor]): An iterable of model parameters
        (e.g., from `model.parameters()`) whose gradients will be synchronized.

    Returns:
        None

    Raises:
        RuntimeError: If the distributed environment is initialized but
        there is a mismatch in the number of parameters across workers.

    Examples:
        >>> model = MyModel()
        >>> loss = compute_loss(model(input), target)
        >>> loss.backward()  # Compute gradients
        >>> sync_grad(model.parameters())  # Synchronize gradients

    Note:
        This function assumes that the distributed environment has been
        properly initialized using PyTorch's distributed package.
    """
    if not is_distributed():
        return
    handles = []
    for p in params:
        if p.grad is not None:
            handle = torch.distributed.all_reduce(
                p.grad.data, op=torch.distributed.ReduceOp.SUM, async_op=True
            )
            handles.append((p, handle))
    for p, handle in handles:
        handle.wait()
        p.grad.data /= world_size()


def average_metrics(metrics: Dict[str, float], count=1.0):
    """
        Average a dictionary of metrics across all workers, using the optional `count`
    as an unnormalized weight.

    This function is designed to be used in a distributed training context, where
    multiple workers compute metrics that need to be averaged. It takes into account
    the `count` parameter to provide weighted averaging.

    Args:
        metrics (Dict[str, float]): A dictionary containing metric names as keys
            and their corresponding values as floats.
        count (float, optional): An optional weight for the metrics. Defaults to
            1.0.

    Returns:
        Dict[str, float]: A new dictionary with the same keys as `metrics`, but
            with values averaged across all workers.

    Examples:
        >>> metrics = {'accuracy': 0.8, 'loss': 0.2}
        >>> averaged_metrics = average_metrics(metrics, count=2.0)
        >>> print(averaged_metrics)
        {'accuracy': 0.8, 'loss': 0.2}

    Note:
        This function assumes that the PyTorch distributed environment has been
        initialized. If it is not, the original `metrics` dictionary will be
        returned unchanged.
    """
    if not is_distributed():
        return metrics
    keys, values = zip(*metrics.items())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor(list(values) + [1], device=device, dtype=torch.float32)
    tensor *= count
    all_reduce(tensor)
    averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
    return dict(zip(keys, averaged))
