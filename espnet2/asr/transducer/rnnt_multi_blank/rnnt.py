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

import multiprocessing

import torch
from numba import cuda

from espnet2.asr.transducer.rnnt_multi_blank.utils import global_constants, rnnt_helper
from espnet2.asr.transducer.rnnt_multi_blank.utils.cpu_utils import cpu_rnnt
from espnet2.asr.transducer.rnnt_multi_blank.utils.cuda_utils import gpu_rnnt


def rnnt_loss_cpu(
    acts: torch.Tensor,
    labels: torch.Tensor,
    input_lengths: torch.Tensor,
    label_lengths: torch.Tensor,
    costs: torch.Tensor,
    grads: torch.Tensor,
    blank_label: int,
    fastemit_lambda: float,
    clamp: float,
    num_threads: int,
):
    """
    Calculate the RNNT loss using a CPU implementation.

    This function serves as a wrapper to compute the RNNT loss for a given 
    set of activations and labels on the CPU. The implementation is based on 
    the work by HawkAaron in the warp-transducer repository.

    Args:
        acts: Activation tensor of shape [B, T, U, V+1], where B is the 
            batch size, T is the time dimension, U is the target sequence 
            length, and V is the vocabulary size.
        labels: Ground truth labels of shape [B, U], where U is the 
            target sequence length.
        input_lengths: A tensor of shape [B] representing the lengths of 
            the acoustic sequences.
        label_lengths: A tensor of shape [B] representing the lengths of 
            the target sequences.
        costs: A tensor of shape [B] initialized to zero, where the computed 
            costs will be stored.
        grads: A tensor of shape [B, T, U, V+1] initialized to zero, where 
            the computed gradients will be stored.
        blank_label: An integer indicating the index of the blank token 
            in the vocabulary.
        fastemit_lambda: A float scaling factor for FastEmit regularization, 
            which can improve the efficiency of streaming ASR.
        clamp: A float value that, when set to a value >= 0.0, will clamp 
            the gradients to the range [-clamp, clamp].
        num_threads: An integer specifying the number of threads to use 
            for OpenMP parallelization.

    Returns:
        bool: Returns True if the computation was successful.

    Raises:
        RuntimeError: If the working space memory allocation fails or if 
        the RNNT status indicates an error during computation.

    Examples:
        >>> acts = torch.randn(32, 100, 20, 30)  # Example activation tensor
        >>> labels = torch.randint(0, 30, (32, 20))  # Example labels
        >>> input_lengths = torch.randint(1, 100, (32,))  # Example lengths
        >>> label_lengths = torch.randint(1, 20, (32,))  # Example lengths
        >>> costs = torch.zeros(32)  # Initialize costs
        >>> grads = torch.zeros(32, 100, 20, 30)  # Initialize grads
        >>> blank_label = 0
        >>> fastemit_lambda = 0.1
        >>> clamp = 1.0
        >>> num_threads = 4

        >>> rnnt_loss_cpu(acts, labels, input_lengths, label_lengths, 
        ...                costs, grads, blank_label, fastemit_lambda, 
        ...                clamp, num_threads)
    """

    # aliases
    log_probs = acts
    flat_labels = labels

    minibatch_size = log_probs.shape[0]
    maxT = log_probs.shape[1]
    maxU = log_probs.shape[2]
    alphabet_size = log_probs.shape[3]

    if num_threads < 0:
        num_threads = multiprocessing.cpu_count()

    num_threads = max(1, num_threads)  # have to use at least 1 thread

    gpu_size, status = rnnt_helper.get_workspace_size(
        maxT, maxU, minibatch_size, gpu=False
    )
    if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
        raise RuntimeError(
            "Invalid parameter passed when calculating working space memory"
        )

    cpu_workspace = torch.zeros(
        gpu_size, device=log_probs.device, dtype=log_probs.dtype, requires_grad=False
    )

    # VIEW TENSORS AS VECTORS FOR POINTER INDEXING
    log_probs, acts_shape = rnnt_helper.flatten_tensor(log_probs)
    flat_labels, labels_shape = rnnt_helper.flatten_tensor(flat_labels)

    wrapper = cpu_rnnt.CPURNNT(
        minibatch=minibatch_size,
        maxT=maxT,
        maxU=maxU,
        alphabet_size=alphabet_size,
        workspace=cpu_workspace,
        blank=blank_label,
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
        num_threads=num_threads,
        batch_first=True,
    )

    if grads is None:
        status = wrapper.score_forward(
            log_probs=log_probs.data,
            costs=costs,
            flat_labels=flat_labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    else:
        # FLATTEN GRAD TENSOR
        grads, grads_shape = rnnt_helper.flatten_tensor(grads)

        status = wrapper.cost_and_grad(
            log_probs=log_probs.data,
            grads=grads.data,
            costs=costs,
            flat_labels=flat_labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    del cpu_workspace, wrapper
    return True


def rnnt_loss_gpu(
    acts: torch.Tensor,
    labels: torch.Tensor,
    input_lengths: torch.Tensor,
    label_lengths: torch.Tensor,
    costs: torch.Tensor,
    grads: torch.Tensor,
    blank_label: int,
    fastemit_lambda: float,
    clamp: float,
    num_threads: int,
):
    """
    Wrapper method for accessing GPU RNNT loss.

    This function computes the Recurrent Neural Network Transducer (RNNT) loss 
    using GPU acceleration. The CUDA implementation is ported from 
    [HawkAaron/warp-transducer](https://github.com/HawkAaron/warp-transducer).

    Args:
        acts: Activation tensor of shape [B, T, U, V+1], where B is the 
            batch size, T is the time dimension, U is the target length, 
            and V is the number of classes excluding the blank label.
        labels: Ground truth labels of shape [B, U].
        input_lengths: Lengths of the acoustic sequence as a vector of 
            integers of shape [B].
        label_lengths: Lengths of the target sequence as a vector of 
            integers of shape [B].
        costs: Zero vector of length [B] where costs will be set.
        grads: Zero tensor of shape [B, T, U, V+1] where the gradient 
            will be set.
        blank_label: Index of the blank token in the vocabulary.
        fastemit_lambda: Float scaling factor for FastEmit regularization. 
            Refer to the FastEmit paper for more details.
        clamp: Float value. When set to value >= 0.0, it clamps the 
            gradient to [-clamp, clamp].
        num_threads: Number of threads for OpenMP. If negative, it will 
            use the number of available CPU cores.

    Returns:
        bool: Returns True if the operation was successful.

    Raises:
        RuntimeError: If there is an invalid parameter passed when calculating 
            workspace memory or if forward scores cannot be calculated.

    Examples:
        >>> acts = torch.randn(16, 50, 30, 20).cuda()  # Example activations
        >>> labels = torch.randint(0, 30, (16, 10)).cuda()  # Example labels
        >>> input_lengths = torch.randint(1, 51, (16,)).cuda()  # Example lengths
        >>> label_lengths = torch.randint(1, 11, (16,)).cuda()  # Example lengths
        >>> costs = torch.zeros(16).cuda()  # Costs initialization
        >>> grads = torch.zeros(16, 50, 30, 21).cuda()  # Gradients initialization
        >>> rnnt_loss_gpu(acts, labels, input_lengths, label_lengths, costs, 
        ...                grads, blank_label=29, fastemit_lambda=0.5, 
        ...                clamp=0.1, num_threads=4)

    Note:
        This function requires the CUDA-enabled version of PyTorch and 
        the appropriate GPU resources.
    """

    minibatch_size = acts.shape[0]
    maxT = acts.shape[1]
    maxU = acts.shape[2]
    alphabet_size = acts.shape[3]

    if hasattr(cuda, "external_stream"):
        stream = cuda.external_stream(
            torch.cuda.current_stream(acts.device).cuda_stream
        )
    else:
        stream = cuda.default_stream()

    if num_threads < 0:
        num_threads = multiprocessing.cpu_count()

    num_threads = max(1, num_threads)  # have to use at least 1 thread

    gpu_size, status = rnnt_helper.get_workspace_size(
        maxT, maxU, minibatch_size, gpu=True
    )
    if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
        raise RuntimeError(
            "Invalid parameter passed when calculating working space memory"
        )

    # Select GPU index
    cuda.select_device(acts.device.index)
    gpu_workspace = torch.zeros(
        gpu_size, device=acts.device, dtype=acts.dtype, requires_grad=False
    )

    # VIEW TENSORS AS VECTORS FOR POINTER INDEXING
    acts, acts_shape = rnnt_helper.flatten_tensor(acts)

    wrapper = gpu_rnnt.GPURNNT(
        minibatch=minibatch_size,
        maxT=maxT,
        maxU=maxU,
        alphabet_size=alphabet_size,
        workspace=gpu_workspace,
        blank=blank_label,
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
        num_threads=num_threads,
        stream=stream,
    )

    if grads is None:
        status = wrapper.score_forward(
            acts=acts.data,
            costs=costs.data,
            pad_labels=labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    else:
        # FLATTEN GRAD TENSOR
        grads, grads_shape = rnnt_helper.flatten_tensor(grads)

        status = wrapper.cost_and_grad(
            acts=acts.data,
            grads=grads.data,
            costs=costs.data,
            pad_labels=labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    del gpu_workspace, wrapper
    return True


def multiblank_rnnt_loss_gpu(
    acts: torch.Tensor,
    labels: torch.Tensor,
    input_lengths: torch.Tensor,
    label_lengths: torch.Tensor,
    costs: torch.Tensor,
    grads: torch.Tensor,
    blank_label: int,
    big_blank_durations: list,
    fastemit_lambda: float,
    clamp: float,
    num_threads: int,
    sigma: float,
):
    """
    Wrapper method for accessing GPU Multi-blank RNNT loss.

    This function computes the RNNT loss for models that utilize a multi-blank
    approach, as described in the paper: 
    https://arxiv.org/pdf/2211.03541.pdf. It is a CUDA implementation
    ported from the Warp Transducer framework.

    Args:
        acts: Activation tensor of shape [B, T, U, V + num_big_blanks + 1].
        labels: Ground truth labels of shape [B, U].
        input_lengths: Lengths of the acoustic sequence as a vector of ints [B].
        label_lengths: Lengths of the target sequence as a vector of ints [B].
        costs: Zero vector of length [B] in which costs will be set.
        grads: Zero tensor of shape [B, T, U, V + num_big_blanks + 1]
            where the gradient will be set.
        blank_label: Index of the standard blank token in the vocabulary.
        big_blank_durations: A list of supported durations for big blank symbols
            in the model, e.g. [2, 4, 8]. This should not include 1 for the
            standard blank.
        fastemit_lambda: Float scaling factor for FastEmit regularization.
            Refer to the FastEmit paper for more details.
        clamp: Float value. When set to value >= 0.0, will clamp the
            gradient to [-clamp, clamp].
        num_threads: Number of threads for OpenMP.
        sigma: Logit-undernormalization weight used in the multi-blank model.
            Refer to the multi-blank paper for detailed explanations.

    Returns:
        bool: Returns True if the computation was successful.

    Raises:
        RuntimeError: If there is an issue with memory allocation or if the
        RNNT status is not successful during calculations.

    Examples:
        >>> acts = torch.randn(8, 50, 20, 10)  # Example activation tensor
        >>> labels = torch.randint(0, 10, (8, 20))  # Example labels
        >>> input_lengths = torch.randint(1, 50, (8,))
        >>> label_lengths = torch.randint(1, 20, (8,))
        >>> costs = torch.zeros(8)
        >>> grads = torch.zeros(8, 50, 20, 10)
        >>> big_blank_durations = [2, 4, 8]
        >>> result = multiblank_rnnt_loss_gpu(
        ...     acts, labels, input_lengths, label_lengths,
        ...     costs, grads, blank_label=0, 
        ...     big_blank_durations=big_blank_durations,
        ...     fastemit_lambda=0.1, clamp=0.1,
        ...     num_threads=4, sigma=0.5
        ... )
        >>> print(result)  # Should print True if successful
    """

    minibatch_size = acts.shape[0]
    maxT = acts.shape[1]
    maxU = acts.shape[2]
    alphabet_size = acts.shape[3]

    if hasattr(cuda, "external_stream"):
        stream = cuda.external_stream(
            torch.cuda.current_stream(acts.device).cuda_stream
        )
    else:
        stream = cuda.default_stream()

    if num_threads < 0:
        num_threads = multiprocessing.cpu_count()

    num_threads = max(1, num_threads)  # have to use at least 1 thread

    gpu_size, status = rnnt_helper.get_workspace_size(
        maxT, maxU, minibatch_size, gpu=True
    )

    if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
        raise RuntimeError(
            "Invalid parameter passed when calculating working space memory"
        )

    # Select GPU index
    cuda.select_device(acts.device.index)
    gpu_workspace = torch.zeros(
        gpu_size, device=acts.device, dtype=acts.dtype, requires_grad=False
    )

    big_blank_workspace = torch.zeros(
        len(big_blank_durations),
        device=acts.device,
        dtype=torch.long,
        requires_grad=False,
    )

    for i in range(0, len(big_blank_durations)):
        big_blank_workspace[i] = big_blank_durations[i]

    # VIEW TENSORS AS VECTORS FOR POINTER INDEXING
    acts, acts_shape = rnnt_helper.flatten_tensor(acts)

    wrapper = gpu_rnnt.MultiblankGPURNNT(
        minibatch=minibatch_size,
        maxT=maxT,
        maxU=maxU,
        alphabet_size=alphabet_size,
        workspace=gpu_workspace,
        big_blank_workspace=big_blank_workspace,
        num_big_blanks=len(big_blank_durations),
        blank=blank_label,
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
        num_threads=num_threads,
        stream=stream,
        sigma=sigma,
    )

    if grads is None:
        status = wrapper.score_forward(
            acts=acts.data,
            costs=costs.data,
            pad_labels=labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    else:
        # FLATTEN GRAD TENSOR
        grads, grads_shape = rnnt_helper.flatten_tensor(grads)

        status = wrapper.cost_and_grad(
            acts=acts.data,
            grads=grads.data,
            costs=costs.data,
            pad_labels=labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    del gpu_workspace, big_blank_workspace, wrapper
    return True
