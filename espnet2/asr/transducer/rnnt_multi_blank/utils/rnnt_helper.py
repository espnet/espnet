# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2018-2019, Mingkun Huang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from typing import Optional, Tuple

import torch
from numba import cuda

from espnet2.asr.transducer.rnnt_multi_blank.utils import global_constants

threshold = global_constants.THRESHOLD


@cuda.jit(device=True, inline=True)
def log_sum_exp(a: float, b: float):
    """
    Calculate the log of the sum of exponentials of two input values.

    This function efficiently computes the logarithm of the sum of the
    exponentials of two floating-point numbers, `a` and `b`, while
    handling cases of negative infinity as defined in the
    `global_constants` module.

    This implementation is designed for use in CUDA kernels, thus it
    uses the `@cuda.jit` decorator for Just-In-Time compilation. The
    function is also inlined for performance optimization.

    Args:
        a (float): The first input value.
        b (float): The second input value.

    Returns:
        float: The logarithm of the sum of exponentials of `a` and `b`.

    Examples:
        >>> result = log_sum_exp(1.0, 2.0)
        >>> print(result)  # Output will be approximately 2.3133

        >>> result = log_sum_exp(global_constants.FP32_NEG_INF, 3.0)
        >>> print(result)  # Output will be 3.0, as -inf is ignored.

        >>> result = log_sum_exp(4.0, global_constants.FP32_NEG_INF)
        >>> print(result)  # Output will be 4.0, as -inf is ignored.

    Note:
        The function assumes that inputs are valid floating-point numbers
        and uses constants defined in `global_constants` to handle edge
        cases effectively.
    """
    if a == global_constants.FP32_NEG_INF:
        return b

    if b == global_constants.FP32_NEG_INF:
        return a

    if a > b:
        return math.log1p(math.exp(b - a)) + a
    else:
        return math.log1p(math.exp(a - b)) + b


@cuda.jit(device=True, inline=True)
def div_up(x: int, y: int):
    """
    Computes the ceiling division of two integers.

    This function takes two integers, `x` and `y`, and returns the result of
    dividing `x` by `y`, rounding up to the nearest whole number. This is useful
    in scenarios where you want to ensure that the result of a division operation
    does not fall short of the intended number of groups or buckets, especially
    in applications like batching or memory allocation.

    Args:
        x (int): The numerator, an integer value that represents the total
                number to be divided.
        y (int): The denominator, a positive integer value by which to divide
                `x`. It must not be zero.

    Returns:
        int: The ceiling result of the division of `x` by `y`.

    Raises:
        ZeroDivisionError: If `y` is zero, as division by zero is undefined.

    Examples:
        >>> div_up(5, 2)
        3
        >>> div_up(10, 3)
        4
        >>> div_up(7, 1)
        7
        >>> div_up(0, 5)
        0

    Note:
        This function is designed to be used in CUDA kernels and is decorated
        with @cuda.jit for that purpose. Ensure that this function is called
        within a proper CUDA context.
    """
    return (x + y - 1) // y


@cuda.jit(device=True)
def maximum(x, y):
    """
    Computes the element-wise maximum of two input values.

    This function takes two input values and returns the greater of the two.
    It is a device function intended to be used within CUDA kernels, enabling
    efficient computation on GPU hardware.

    Args:
        x (float): The first input value.
        y (float): The second input value.

    Returns:
        float: The maximum value between `x` and `y`.

    Examples:
        >>> from espnet2.asr.transducer.rnnt_multi_blank.utils.rnnt_helper import maximum
        >>> result = maximum(3.0, 5.0)
        >>> print(result)
        5.0

        >>> result = maximum(-1.0, 0.0)
        >>> print(result)
        0.0

    Note:
        This function is decorated with `@cuda.jit(device=True)` to indicate
        that it can be called from other CUDA device functions.
    """
    if x < y:
        return y
    else:
        return x


@cuda.jit(device=True)
def add(x, y):
    """
    Compute the sum of two values.

    This function performs an addition operation on two input values,
    returning the result. It is designed to be used in a CUDA kernel.

    Args:
        x (float): The first value to add.
        y (float): The second value to add.

    Returns:
        float: The sum of the two input values.

    Examples:
        >>> result = add(3.5, 2.0)
        >>> print(result)
        5.5

        >>> result = add(-1.5, 1.5)
        >>> print(result)
        0.0

    Note:
        This function is optimized for use within a CUDA kernel context and
        should not be used outside of that environment.
    """
    return x + y


@cuda.jit(device=True)
def identity(x):
    """
    Identity function for use in CUDA kernels.

    This function returns the input value as-is. It is typically used in scenarios
    where an operation requires a function but no transformation of the input is
    needed. This is especially useful in neural network architectures or during
    computation graphs where identity mappings are required.

    Args:
        x (float): The input value to be returned unchanged.

    Returns:
        float: The input value `x`.

    Examples:
        >>> result = identity(5.0)
        >>> print(result)
        5.0

        >>> result = identity(-3.2)
        >>> print(result)
        -3.2

    Note:
        This function is compiled to run on CUDA devices using Numba, making it
        suitable for GPU-based computations.
    """
    return x


@cuda.jit(device=True)
def negate(x):
    """
    Negate the input value.

    This function takes a single input value and returns its negation.
    It is designed to be used in a CUDA kernel to perform negation on
    values in parallel.

    Args:
        x (float): The input value to be negated.

    Returns:
        float: The negated value of the input.

    Examples:
        >>> result = negate(5.0)
        >>> print(result)
        -5.0

        >>> result = negate(-3.2)
        >>> print(result)
        3.2
    """
    return -x


@cuda.jit(device=True)
def exponential(x):
    """
    Compute the exponential of a given value.

    This function calculates the exponential of the input value `x`
    using the mathematical constant e. It is designed to be used in a
    CUDA kernel, allowing for efficient computation on the GPU.

    Args:
        x (float): The input value for which the exponential is to be
            computed.

    Returns:
        float: The exponential of the input value `x`, calculated as
            e^x.

    Examples:
        >>> result = exponential(1.0)
        >>> print(result)  # Output: 2.718281828459045

        >>> result = exponential(0.0)
        >>> print(result)  # Output: 1.0

        >>> result = exponential(-1.0)
        >>> print(result)  # Output: 0.36787944117144233

    Note:
        This function should only be called within a CUDA kernel context.
    """
    return math.exp(x)


@cuda.jit(device=True)
def log_plus(p1: float, p2: float):
    """
    Compute the log of the sum of exponentials of two values.

    This function calculates the logarithm of the sum of exponentials of
    two input probabilities, `p1` and `p2`, using a numerically stable
    approach. This is particularly useful in applications such as
    log-probability calculations in machine learning, where directly
    computing the sum of exponentials can lead to overflow issues.

    Args:
        p1 (float): The first probability value, which may be negative infinity.
        p2 (float): The second probability value, which may be negative infinity.

    Returns:
        float: The log of the sum of exponentials of `p1` and `p2`.

    Examples:
        >>> log_plus(0.5, 0.5)
        0.6931471805599453  # log(2) = log(0.5 + 0.5)

        >>> log_plus(-float('inf'), 1.0)
        1.0  # log(0 + exp(1.0)) = 1.0

        >>> log_plus(-1.0, -2.0)
        -0.3132616875182228  # log(exp(-1.0) + exp(-2.0))

    Note:
        If either `p1` or `p2` is negative infinity, the other value will be
        returned directly.
    """
    if p1 == global_constants.FP32_NEG_INF:
        return p2

    if p2 == global_constants.FP32_NEG_INF:
        return p1

    result = math.log1p(math.exp(-math.fabs(p1 - p2))) + maximum(p1, p2)
    return result


@cuda.jit(device=True, inline=True)
def copy_data_1d(source: torch.Tensor, dest: torch.Tensor, idx: int):
    """
    Copies a single element from a source tensor to a destination tensor at the
    specified index.

    This function is intended for use in CUDA kernels to facilitate data
    manipulation between tensors. It assumes that both `source` and `dest` are
    1-dimensional tensors and that the provided index is within the bounds of
    these tensors.

    Args:
        source (torch.Tensor): The source tensor from which to copy data. It must
            be a 1-dimensional tensor.
        dest (torch.Tensor): The destination tensor where data will be copied.
            It must also be a 1-dimensional tensor.
        idx (int): The index at which to copy the data from the source tensor
            to the destination tensor.

    Examples:
        >>> import torch
        >>> source_tensor = torch.tensor([1.0, 2.0, 3.0])
        >>> dest_tensor = torch.zeros(3)
        >>> copy_data_1d(source_tensor, dest_tensor, 1)
        >>> print(dest_tensor)  # Output: tensor([0.0, 2.0, 0.0])

    Note:
        This function should be called within a CUDA kernel and is not intended
        for direct invocation in Python code.
    """
    dest[idx] = source[idx]


@cuda.jit()
def compute_costs_data(
    source: torch.Tensor, dest: torch.Tensor, fastemit_lambda: float
):
    """
    Compute the costs for the RNN Transducer model.

    This CUDA kernel computes the costs for the RNN Transducer model by
    copying data from the source tensor to the destination tensor,
    modifying the destination values based on a given `fastemit_lambda`.
    The computed costs are the negative values of the source tensor
    scaled by the factor (1 + `fastemit_lambda`).

    Args:
        source (torch.Tensor): The input tensor containing source values.
        dest (torch.Tensor): The output tensor where computed costs will be stored.
        fastemit_lambda (float): A scaling factor that influences the cost computation.

    Returns:
        None: This function modifies the `dest` tensor in place.

    Examples:
        >>> source_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        >>> dest_tensor = torch.empty_like(source_tensor)
        >>> fastemit_lambda = 0.5
        >>> compute_costs_data(source_tensor, dest_tensor, fastemit_lambda)
        >>> print(dest_tensor)  # Should output modified values based on source

    Note:
        This function is designed to run on a CUDA device. Ensure that
        the input tensors are on the GPU.
    """
    block = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    idx = block * cuda.blockDim.x + tid
    length = source.shape[0]

    if idx < length:
        copy_data_1d(source, dest, idx)
        dest[idx] *= -1.0
        dest[idx] *= 1.0 + fastemit_lambda


def get_workspace_size(
    maxT: int, maxU: int, minibatch: int, gpu: bool
) -> Tuple[Optional[int], global_constants.RNNTStatus]:
    """
    Calculate the required workspace size for the RNNT (Recurrent Neural Network
    Transducer) model based on input parameters.

    This function computes the amount of memory needed for the RNNT model to
    perform forward and backward passes during training or inference, considering
    both CPU and GPU execution environments. It takes into account the maximum
    sequence lengths, the number of tokens, and the size of the minibatch.

    Attributes:
        maxT (int): The maximum length of the input sequences.
        maxU (int): The maximum number of tokens (including blanks).
        minibatch (int): The number of sequences processed in parallel.
        gpu (bool): A flag indicating whether to calculate for GPU or CPU.

    Args:
        maxT (int): Maximum time steps in the input sequences.
        maxU (int): Maximum number of unique labels (including blanks).
        minibatch (int): Number of sequences processed in a single pass.
        gpu (bool): Flag indicating if the calculations should be done for GPU.

    Returns:
        Tuple[Optional[int], global_constants.RNNTStatus]: A tuple containing the
        computed workspace size in bytes (or None if invalid) and the status of
        the operation.

    Raises:
        ValueError: If any of the input parameters (minibatch, maxT, maxU) are
        less than or equal to zero.

    Examples:
        >>> size, status = get_workspace_size(100, 50, 32, True)
        >>> print(size)  # Expected output: Size in bytes required for workspace
        >>> print(status)  # Expected output: RNNT_STATUS_SUCCESS

    Note:
        The calculated workspace size is essential for memory management in
        deep learning applications to avoid runtime errors related to memory
        allocation.
    """
    if minibatch <= 0 or maxT <= 0 or maxU <= 0:
        return (None, global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE)

    # per minibatch memory
    per_minibatch_size = 0

    # alphas & betas
    per_minibatch_size += maxT * maxU * 2

    if not gpu:
        # // blank & label log probability cache
        per_minibatch_size += maxT * maxU * 2
    else:
        # // softmax denominator
        per_minibatch_size += maxT * maxU
        # // forward - backward loglikelihood
        per_minibatch_size += 2

    size = per_minibatch_size * minibatch
    return (size, global_constants.RNNTStatus.RNNT_STATUS_SUCCESS)


def flatten_tensor(x: torch.Tensor):
    """
    Flatten a multi-dimensional tensor into a one-dimensional tensor.

    This function takes a tensor as input and reshapes it into a one-dimensional
    tensor while also returning its original shape. This is useful in scenarios
    where you need to process data in a flattened format, such as during certain
    operations in neural networks.

    Args:
        x (torch.Tensor): The input tensor to be flattened. It can be of any
                        shape.

    Returns:
        Tuple[torch.Tensor, Tuple[int, ...]]: A tuple containing the flattened
        tensor and its original shape. The first element is the flattened tensor,
        and the second element is a tuple representing the original dimensions.

    Examples:
        >>> import torch
        >>> tensor = torch.tensor([[1, 2], [3, 4]])
        >>> flattened_tensor, original_shape = flatten_tensor(tensor)
        >>> print(flattened_tensor)
        tensor([1, 2, 3, 4])
        >>> print(original_shape)
        (2, 2)

    Note:
        The input tensor should be a PyTorch tensor. The function modifies
        the view of the tensor but does not alter the underlying data.
    """
    original_shape = x.shape
    x = x.view([-1])
    return x, original_shape
