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

import enum
import math

import torch
from numba import cuda

from espnet2.asr.transducer.rnnt_multi_blank.utils import global_constants, rnnt_helper

warp_size = global_constants.warp_size()
dtype = global_constants.dtype()

CTA_REDUCE_SIZE = 128


class I_Op(enum.Enum):
    """
    Represents an operation that is performed on the input tensor.

    This enumeration defines two types of operations that can be applied to 
    the input tensor during processing. The operations are categorized into 
    exponential and identity transformations.

    Attributes:
        EXPONENTIAL (int): Represents the exponential operation.
        IDENTITY (int): Represents the identity operation.

    Examples:
        >>> operation = I_Op.EXPONENTIAL
        >>> print(operation)
        I_Op.EXPONENTIAL
        >>> operation = I_Op.IDENTITY
        >>> print(operation)
        I_Op.IDENTITY
    """

    EXPONENTIAL = 0
    IDENTITY = 1


class R_Op(enum.Enum):
    """
    Represents a reduction operation performed on the input tensor.

    This enumeration defines the types of reduction operations that can be 
    performed during the CUDA kernel execution. The available operations are 
    as follows:

    Attributes:
        ADD: Represents the addition operation for reduction.
        MAXIMUM: Represents the maximum operation for reduction.

    Examples:
        To use the R_Op enum for reduction in a CUDA kernel:

        ```python
        reduction_operation = R_Op.ADD
        if reduction_operation == R_Op.ADD:
            # Perform addition reduction
        elif reduction_operation == R_Op.MAXIMUM:
            # Perform maximum reduction
        ```

    Note:
        This enum is typically used in conjunction with the CTAReduce and 
        other reduction functions to specify the desired reduction behavior.
    """

    ADD = 0
    MAXIMUM = 1


@cuda.jit(device=True)
def CTAReduce(tid: int, x, storage, count: int, R_opid: int):
    """
    CTAReduce performs a CUDA Warp reduction on a given input tensor.

    This function implements a device kernel for reducing input values using a
    specified reduction operation. The data is recursively read from the right 
    segment and reduced onto the left half, continuing until the warp size is 
    larger than a given offset. Beyond this offset, warp reduction is performed 
    using `shfl_down_sync`, effectively halving the reduction space and 
    combining the results in each iteration.

    Note:
        Efficient warp occurs at input shapes of 2 ^ K.

    References:
        - Warp Primitives
        [https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/]

    Args:
        tid: int
            CUDA thread index.
        x: float
            Activation value to be reduced.
        storage: array
            Shared memory of size CTA_REDUCE_SIZE used for parallel reduction.
        count: int
            Equivalent to num_rows, which corresponds to alphabet_size (V + 1).
        R_opid: int
            Operator ID for reduction. See R_Op for more information.

    Returns:
        float
            The reduced value after applying the specified reduction operation.

    Examples:
        >>> storage = cuda.shared.array(shape=(CTA_REDUCE_SIZE,), dtype=float)
        >>> reduced_value = CTAReduce(tid=0, x=5.0, storage=storage, count=16, R_opid=0)

    Raises:
        None
    """

    storage[tid] = x

    cuda.syncthreads()

    # Fold the data in half with each pass
    offset = CTA_REDUCE_SIZE // 2
    while offset >= warp_size:
        if (tid + offset) < count and tid < offset:
            # Read from the right half and store to the left half.
            if R_opid == 0:
                x = rnnt_helper.add(x, storage[offset + tid])
            else:
                x = rnnt_helper.maximum(x, storage[offset + tid])

            storage[tid] = x

        cuda.syncthreads()
        offset = offset // 2

    offset = warp_size // 2
    while offset > 0:
        # warp reduction and sync
        shuff = cuda.shfl_down_sync(0xFFFFFFFF, x, offset)

        if (tid + offset < count) and (tid < offset):
            if R_opid == 0:
                x = rnnt_helper.add(x, shuff)
            else:
                x = rnnt_helper.maximum(x, shuff)

        offset = offset // 2

    return x


@cuda.jit()
def _reduce_rows(I_opid: int, R_opid: int, acts, output, num_rows: int):
    """CUDA Warp reduction kernel which reduces via the R_Op.Maximum

    Reduces the input data such that I_Op = Identity and R_op = Maximum.
    The result is stored in the blockIdx, and is stored as an identity op.

    Note:
        Efficient warp occurs at input shapes of 2 ^ K.

    References:
        - Warp Primitives
          [https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/]

    Args:
        I_opid: Operator ID for input. See I_Op for more information. For this kernel,
            the Identity op is chosen in general, and therefore the input
            is reduced in place without scaling.
        R_opid: Operator ID for reduction. See R_Op for more information.
            For this kernel, generally Maximum op is chosen.
            It reduces the kernel via max.
        acts: Flatened activation matrix of shape [B * T * U * (V+1)].
        output: Flatened output matrix of shape [B * T * U * (V+1)].
            Data will be overwritten.
        num_rows: Vocabulary size (including blank token) - V+1.
    """

    tid = cuda.threadIdx.x
    idx = tid
    col = cuda.blockIdx.x

    # allocate shared thread memory
    storage = cuda.shared.array(shape=(CTA_REDUCE_SIZE,), dtype=acts.dtype)

    max = output[col]

    # // Each block works on a column
    if idx < num_rows:
        curr = acts[col * num_rows + idx] - max
        if I_opid == 0:
            curr = rnnt_helper.exponential(curr)
        else:
            curr = rnnt_helper.identity(curr)

    idx += CTA_REDUCE_SIZE

    while idx < num_rows:
        activation_ = acts[col * num_rows + idx] - max

        if I_opid == 0 and R_opid == 0:
            curr = rnnt_helper.add(curr, rnnt_helper.exponential(activation_))
        elif I_opid == 0 and R_opid == 1:
            curr = rnnt_helper.maximum(curr, rnnt_helper.exponential(activation_))
        elif I_opid == 1 and R_opid == 0:
            curr = rnnt_helper.add(curr, rnnt_helper.identity(activation_))
        else:
            curr = rnnt_helper.maximum(curr, rnnt_helper.identity(activation_))

        idx += CTA_REDUCE_SIZE

    # // Sum thread-totals over the CTA.
    curr = CTAReduce(tid, curr, storage, num_rows, R_opid)

    # // Store result in out (inplace, I_op: identity)
    if tid == 0:
        output[col] = curr


@cuda.jit()
def _reduce_minus(I_opid: int, R_opid: int, acts, output, num_rows: int):
    """CUDA Warp reduction kernel which reduces via the R_Op.Add

    Reduces the input data such that I_Op = Exponential and R_op = Add.
    The result is stored in the blockIdx, and is stored as an exp op.

    Note:
        Efficient warp occurs at input shapes of 2 ^ K.

    References:
        - Warp Primitives
          [https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/]

    Args:
        I_opid: Operator ID for input. See I_Op for more information. For this kernel,
            the Exponential op is chosen in general, and therefore the input
            is reduced in place with scaling.
        R_opid: Operator ID for reduction. See R_Op for more information. For this
            kernel, generally Add op is chosen. It reduces the kernel via summation.
        acts: Flatened activation matrix of shape [B * T * U * (V+1)].
        output: Flatened output matrix of shape [B * T * U * (V+1)].
            Data will be overwritten.
        num_rows: Vocabulary size (including blank token) - V+1.
    """

    tid = cuda.threadIdx.x
    idx = tid
    col = cuda.blockIdx.x

    # allocate shared thread memory
    storage = cuda.shared.array(shape=(CTA_REDUCE_SIZE,), dtype=acts.dtype)

    max = output[col]

    # // Each block works on a column
    if idx < num_rows:
        curr = acts[col * num_rows + idx] - max
        if I_opid == 0:
            curr = rnnt_helper.exponential(curr)
        else:
            curr = rnnt_helper.identity(curr)

    idx += CTA_REDUCE_SIZE

    while idx < num_rows:
        activation_ = acts[col * num_rows + idx] - max

        if I_opid == 0 and R_opid == 0:
            curr = rnnt_helper.add(curr, rnnt_helper.exponential(activation_))
        elif I_opid == 0 and R_opid == 1:
            curr = rnnt_helper.maximum(curr, rnnt_helper.exponential(activation_))
        elif I_opid == 1 and R_opid == 0:
            curr = rnnt_helper.add(curr, rnnt_helper.identity(activation_))
        else:
            curr = rnnt_helper.maximum(curr, rnnt_helper.identity(activation_))

        idx += CTA_REDUCE_SIZE

    # // Sum thread-totals over the CTA.
    curr = CTAReduce(tid, curr, storage, num_rows, R_opid)

    # // Store result in out (inplace, I_op: exponential)
    if tid == 0:
        output[col] = -max - math.log(curr)


def ReduceHelper(
    I_opid: int,
    R_opid: int,
    acts: torch.Tensor,
    output: torch.Tensor,
    num_rows: int,
    num_cols: int,
    minus: bool,
    stream,
):
    """
    ReduceHelper is a CUDA Warp reduction kernel helper that performs reductions on 
    input activation matrices according to specified input and reduction operator IDs. 
    The results are written to the `output` tensor based on the selected operations.

    This function can execute either a maximum or an additive reduction based on the 
    specified parameters, while efficiently handling the input shapes that are powers 
    of two.

    Note:
        Efficient warp occurs at input shapes of 2 ^ K.

    References:
        - Warp Primitives
        [https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/]

    Args:
        I_opid (int): Operator ID for input, defined in I_Op enumeration.
        R_opid (int): Operator ID for reduction, defined in R_Op enumeration.
        acts (torch.Tensor): Flattened activation matrix of shape 
            [B * T * U * (V+1)].
        output (torch.Tensor): Flattened output matrix of shape 
            [B * T * U * (V+1)]. Data will be overwritten.
        num_rows (int): Vocabulary size (including blank token) - V+1.
            Represents the number of threads per block.
        num_cols (int): Flattened shape of activation matrix, without 
            vocabulary dimension (B * T * U). Represents number of blocks per grid.
        minus (bool): Flag indicating whether to perform subtraction. If set to 
            True, calls the _reduce_minus kernel; otherwise, calls the 
            _reduce_rows kernel.
        stream: CUDA Stream to manage asynchronous execution.

    Returns:
        bool: Returns True upon successful execution of the reduction operation.

    Examples:
        >>> acts = torch.randn((2, 3, 4, 5))  # Example activation tensor
        >>> output = torch.zeros((2, 3, 4, 5))  # Output tensor
        >>> num_rows = 6  # Example number of rows
        >>> num_cols = 4  # Example number of columns
        >>> minus = False  # Example flag
        >>> stream = None  # Assuming synchronous execution
        >>> ReduceHelper(I_opid=0, R_opid=0, acts=acts, output=output,
        ...               num_rows=num_rows, num_cols=num_cols,
        ...               minus=minus, stream=stream)
    """

    if minus:
        grid_size = num_cols
        # call kernel
        _reduce_minus[grid_size, CTA_REDUCE_SIZE, stream, 0](
            I_opid, R_opid, acts, output, num_rows
        )

    else:
        grid_size = num_cols
        # call kernel
        _reduce_rows[grid_size, CTA_REDUCE_SIZE, stream, 0](
            I_opid, R_opid, acts, output, num_rows
        )

    return True


def reduce_exp(acts: torch.Tensor, denom, rows: int, cols: int, minus: bool, stream):
    """
    Helper method to call the Warp Reduction Kernel to perform `exp` reduction.

    This function facilitates the reduction of an input activation matrix by applying
    an exponential operation followed by an addition operation, utilizing CUDA
    for efficient parallel processing.

    Note:
        Efficient warp occurs at input shapes of 2 ^ K.

    References:
        - Warp Primitives
          [https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/]

    Args:
        acts: Flatened activation matrix of shape [B * T * U * (V+1)].
        denom: Flatened output matrix of shape [B * T * U * (V+1)].
            Data will be overwritten with the reduction results.
        rows: Vocabulary size (including blank token) - V+1.
            Represents the number of threads per block.
        cols: Flattened shape of activation matrix, without vocabulary dimension
            (B * T * U). Represents number of blocks per grid.
        minus: Bool flag indicating whether to add or subtract as reduction.
            If minus is set to True, it calls the _reduce_minus kernel; 
            otherwise, it calls the _reduce_rows kernel.
        stream: CUDA Stream for managing execution of kernels.

    Returns:
        bool: Returns True upon successful execution of the reduction.
    
    Examples:
        >>> acts = torch.randn((B, T, U, V+1)).flatten()
        >>> denom = torch.zeros((B, T, U, V+1)).flatten()
        >>> reduce_exp(acts, denom, V+1, B*T*U, False, stream)
    """

    return ReduceHelper(
        I_opid=I_Op.EXPONENTIAL.value,
        R_opid=R_Op.ADD.value,
        acts=acts,
        output=denom,
        num_rows=rows,
        num_cols=cols,
        minus=minus,
        stream=stream,
    )


def reduce_max(acts: torch.Tensor, denom, rows: int, cols: int, minus: bool, stream):
    """
    Helper method to call the Warp Reduction Kernel to perform `max` reduction.

    This function facilitates the execution of a warp reduction operation
    that computes the maximum value across the specified dimensions of
    the flattened activation matrix `acts`. The result is stored in
    the `denom` output tensor.

    Efficient warp reduction is particularly effective when the input
    shapes are powers of two (2^K).

    References:
        - Warp Primitives
          [https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/]

    Args:
        acts: A flattened activation matrix of shape [B * T * U * (V+1)].
              This tensor contains the values to be reduced.
        denom: A flattened output matrix of shape [B * T * U * (V+1)].
               This tensor will be overwritten with the results of the
               reduction operation.
        rows: An integer representing the vocabulary size (including
              blank token) - V+1. This value indicates the number of
              threads per block.
        cols: An integer representing the flattened shape of the
              activation matrix, excluding the vocabulary dimension
              (B * T * U). This value indicates the number of blocks
              per grid.
        minus: A boolean flag indicating whether to add or subtract
               during the reduction. If `minus` is set to True, it
               calls the `_reduce_minus` kernel; otherwise, it calls
               the `_reduce_rows` kernel.
        stream: The CUDA Stream used for executing the kernel.

    Returns:
        A boolean value indicating the success of the reduction operation.

    Examples:
        >>> acts = torch.rand(256, 10, 20, 30)  # Example activation tensor
        >>> denom = torch.zeros(256, 10, 20, 30)  # Output tensor
        >>> rows = 31  # Example vocabulary size
        >>> cols = 2560  # Example number of blocks
        >>> stream = cuda.stream()  # Example CUDA stream
        >>> reduce_max(acts, denom, rows, cols, False, stream)
    """

    return ReduceHelper(
        I_opid=I_Op.IDENTITY.value,
        R_opid=R_Op.MAXIMUM.value,
        acts=acts,
        output=denom,
        num_rows=rows,
        num_cols=cols,
        minus=minus,
        stream=stream,
    )
