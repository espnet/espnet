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

import numpy as np
from numba import float32

# Internal globals
_THREADS_PER_BLOCK = 32
_WARP_SIZE = 32
_DTYPE = float32

# Constants
FP32_INF = np.inf
FP32_NEG_INF = -np.inf
THRESHOLD = 1e-1

"""
Getters
"""


def threads_per_block():
    """
    Retrieve the number of threads per block used in GPU computations.

    This function returns the constant value representing the number of threads 
    that are configured to run in parallel within a single block in a GPU. 
    The default value is set to 32, which is a common choice for optimizing 
    performance in CUDA applications.

    Args:
        None

    Returns:
        int: The number of threads per block (default is 32).

    Examples:
        >>> num_threads = threads_per_block()
        >>> print(num_threads)
        32

    Note:
        This value can be adjusted in the code to optimize for specific 
        hardware or application requirements.

    Todo:
        Consider exposing a mechanism to configure this value at runtime 
        for more flexibility in different environments.
    """
    global _THREADS_PER_BLOCK
    return _THREADS_PER_BLOCK


def warp_size():
    """
    Retrieve the size of a warp in GPU programming.

    A warp is a group of threads that execute instructions in lockstep. 
    This function returns the constant size of a warp, which is typically 
    32 threads for NVIDIA GPUs.

    Returns:
        int: The size of a warp, which is 32.

    Examples:
        >>> size = warp_size()
        >>> print(size)
        32

    Note:
        This function is designed to provide a consistent value for the 
        warp size across different parts of the application.
    """
    global _WARP_SIZE
    return _WARP_SIZE


def dtype():
    """
    Return the data type used in the computation.

    This function retrieves the global data type defined for 
    numerical operations within the package. The current 
    implementation uses `float32` from the Numba library for 
    performance optimization, particularly in GPU computations.

    Returns:
        dtype: The data type (currently set to float32).

    Examples:
        >>> dt = dtype()
        >>> print(dt)
        <class 'numba.float32'>
    
    Note:
        The data type can be adjusted if needed for different 
        precision requirements or hardware capabilities.

    Raises:
        None: This function does not raise any exceptions.

    Todo:
        Consider extending functionality to allow dynamic 
        selection of data types based on user input.
    """
    global _DTYPE
    return _DTYPE


# RNNT STATUS
class RNNTStatus(enum.Enum):
    """
    Enumeration for the status codes used in the RNNT (Recurrent Neural Network 
    Transducer) operations.

    Attributes:
        RNNT_STATUS_SUCCESS (int): Indicates that the operation was successful.
        RNNT_STATUS_INVALID_VALUE (int): Indicates that an invalid value was 
            provided to the operation.

    Examples:
        >>> status = RNNTStatus.RNNT_STATUS_SUCCESS
        >>> if status == RNNTStatus.RNNT_STATUS_SUCCESS:
        ...     print("Operation completed successfully.")
        ...
        Operation completed successfully.

        >>> status = RNNTStatus.RNNT_STATUS_INVALID_VALUE
        >>> if status == RNNTStatus.RNNT_STATUS_INVALID_VALUE:
        ...     print("An invalid value was encountered.")
        ...
        An invalid value was encountered.
    """
    RNNT_STATUS_SUCCESS = 0
    RNNT_STATUS_INVALID_VALUE = 1
