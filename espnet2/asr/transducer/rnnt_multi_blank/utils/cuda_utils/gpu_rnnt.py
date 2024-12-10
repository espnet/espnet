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
from typing import Optional, Tuple, Union

import numba
import torch
from numba import cuda

from espnet2.asr.transducer.rnnt_multi_blank.utils import global_constants, rnnt_helper
from espnet2.asr.transducer.rnnt_multi_blank.utils.cuda_utils import (
    gpu_rnnt_kernel,
    reduce,
)


class GPURNNT:
    """
    Helper class to launch the CUDA Kernels to compute the Transducer Loss.

        This class is responsible for computing the RNNT (Recurrent Neural
        Network Transducer) loss and its gradients using CUDA kernels. It
        manages workspace memory and handles input activations to efficiently
        compute both the loss and the gradients.

        Args:
            minibatch: Int representing the batch size.
            maxT: The maximum possible acoustic sequence length. Represents T
                in the logprobs tensor.
            maxU: The maximum possible target sequence length. Represents U
                in the logprobs tensor.
            alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT
                blank).
            workspace: An allocated chunk of memory that will be sliced off
                and reshaped into required blocks used as working memory.
            blank: Index of the RNNT blank token in the vocabulary. Generally
                the first or last token in the vocab.
            fastemit_lambda: Float scaling factor for FastEmit regularization.
                Refer to FastEmit: Low-latency Streaming ASR with
                Sequence-level Emission Regularization.
            clamp: Float value. When set to value >= 0.0, will clamp the
                gradient to [-clamp, clamp].
            num_threads: Number of OMP threads to launch.
            stream: Numba Cuda Stream.

        Examples:
            >>> workspace = torch.zeros((minibatch, maxT * maxU * (alphabet_size + 1)))
            >>> gpu_rnnt = GPURNNT(minibatch=32, maxT=100, maxU=50,
            ...                    alphabet_size=29, workspace=workspace,
            ...                    blank=0, fastemit_lambda=0.5, clamp=0.1,
            ...                    num_threads=4, stream=cuda.stream())
    """

    def __init__(
        self,
        minibatch: int,
        maxT: int,
        maxU: int,
        alphabet_size: int,
        workspace,
        blank: int,
        fastemit_lambda: float,
        clamp: float,
        num_threads: int,
        stream,
    ):
        """Helper class to launch the CUDA Kernels to compute the Transducer Loss.

        Args:
            minibatch: Int representing the batch size.
            maxT: The maximum possible acoustic sequence length.
                Represents T in the logprobs tensor.
            maxU: The maximum possible target sequence length.
                Represents U in the logprobs tensor.
            alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
            workspace: An allocated chunk of memory that will be sliced off and
                reshaped into required blocks used as working memory.
            blank: Index of the RNNT blank token in the vocabulary.
                Generally the first or last token in the vocab.
            fastemit_lambda: Float scaling factor for FastEmit regularization. Refer to
                FastEmit: Low-latency Streaming ASR with Sequence-level
                Emission Regularization.
            clamp: Float value. When set to value >= 0.0, will clamp
                the gradient to [-clamp, clamp].
            num_threads: Number of OMP threads to launch.
            stream: Numba Cuda Stream.
        """

        self.minibatch_ = minibatch
        self.maxT_ = maxT
        self.maxU_ = maxU
        self.alphabet_size_ = alphabet_size
        self.gpu_workspace = cuda.as_cuda_array(
            workspace
        )  # a flat vector of floatX numbers that represents allocated memory slices
        self.blank_ = blank
        self.fastemit_lambda_ = fastemit_lambda
        self.clamp_ = abs(clamp)
        self.num_threads_ = num_threads
        self.stream_ = stream  # type: cuda.cudadrv.driver.Stream

        if num_threads > 0:
            numba.set_num_threads(min(multiprocessing.cpu_count(), num_threads))
            self.num_threads_ = numba.get_num_threads()
        else:
            self.num_threads_ = numba.get_num_threads()

    def log_softmax(self, acts: torch.Tensor, denom: torch.Tensor):
        """
        Computes the log softmax denominator of the input activation tensor
        and stores the result in the provided denom tensor.

        This method calculates the log softmax of the input activation tensor
        `acts`, which is expected to be a tensor of shape [B, T, U, V+1]. The
        results are stored in the `denom` tensor, which should be initialized
        as a zero tensor of the same shape as `acts`.

        Args:
            acts: Activation tensor of shape [B, T, U, V+1]. The input must be
                represented as a flat tensor of shape [B * T * U * (V+1)] to
                allow pointer indexing.
            denom: A zero tensor of the same shape as acts that will be updated
                in place with the computed log softmax values.

        Updates:
            This method performs in-place updates to the `denom` tensor.

        Examples:
            >>> acts = torch.randn(2, 3, 4, 5)  # Random activation tensor
            >>> denom = torch.zeros_like(acts)  # Initialize denom tensor
            >>> log_softmax(acts, denom)  # Compute log softmax

        Note:
            This method uses CUDA kernels to perform the computations efficiently
            on the GPU.

        Raises:
            ValueError: If `acts` or `denom` are not the correct shapes or types.
        """

        # // trans_acts + pred_acts -> log_softmax denominator
        reduce.reduce_max(
            acts,
            denom,
            rows=self.alphabet_size_,
            cols=self.minibatch_ * self.maxT_ * self.maxU_,
            minus=False,
            stream=self.stream_,
        )

        reduce.reduce_exp(
            acts,
            denom,
            rows=self.alphabet_size_,
            cols=self.minibatch_ * self.maxT_ * self.maxU_,
            minus=True,
            stream=self.stream_,
        )

    def compute_cost_and_score(
        self,
        acts: torch.Tensor,
        grads: Optional[torch.Tensor],
        costs: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> global_constants.RNNTStatus:
        """
        Compute both the loss and the gradients.

        This method computes the negative log likelihood costs and the gradients
        for the RNNT model during training or evaluation. It performs the forward
        pass through the RNNT computation graph, calculating both alphas and
        betas, which are used to derive the gradients and costs.

        Args:
            acts: A flattened tensor of shape [B, T, U, V+1] representing the
                activation matrix, where B is the batch size, T is the maximum
                acoustic sequence length, U is the maximum target sequence length,
                and V is the vocabulary size.
            grads: An optional flattened tensor of the same shape as `acts`,
                initialized to zero, to store gradients. If not provided,
                gradients will not be computed.
            costs: A zero vector of length B that will be updated in place
                with the log probability costs.
            labels: A flattened matrix of labels of shape [B, U], representing
                the target sequences for each batch.
            label_lengths: A vector of length B that contains the original
                lengths of the target sequences.
            input_lengths: A vector of length B that contains the original
                lengths of the acoustic sequences.

        Updates:
            This method will launch kernels that will update the following variables
            in place:
            - grads: Gradients of the activation matrix with respect to the costs vector.
            - costs: Negative log likelihood of the forward variable.

        Returns:
            An enum that represents either a successful RNNT operation or failure.

        Examples:
            >>> acts = torch.rand((2, 10, 5, 20))  # Random activation matrix
            >>> grads = torch.zeros((2, 10, 5, 20))  # Initialize gradients
            >>> costs = torch.zeros(2)  # Initialize costs
            >>> labels = torch.tensor([[1, 2, 3, 0, 0], [2, 3, 0, 0, 0]])  # Example labels
            >>> label_lengths = torch.tensor([3, 2])  # Lengths of each label
            >>> input_lengths = torch.tensor([10, 10])  # Lengths of each input
            >>> status = compute_cost_and_score(acts, grads, costs, labels,
                                                label_lengths, input_lengths)
            >>> print(status)  # Check the status of the operation

        Note:
            Ensure that the input tensors are properly shaped and initialized
            before calling this method.

        Raises:
            ValueError: If the shapes of the input tensors do not match the
            expected dimensions.
        """

        training = grads is not None

        if training:
            grads *= 0.0  # zero grads

        used_offset, (
            denom,
            alphas,
            betas,
            llForward,
            llBackward,
        ) = self._prepare_workspace()

        # START EXECUTION
        self.log_softmax(acts, denom)

        # Compute alphas
        gpu_rnnt_kernel.compute_alphas_kernel[
            self.minibatch_, self.maxU_, self.stream_, 0
        ](
            acts,
            denom,
            alphas,
            llForward,
            input_lengths,
            label_lengths,
            labels,
            self.minibatch_,
            self.maxT_,
            self.maxU_,
            self.alphabet_size_,
            self.blank_,
        )

        if training:
            # Compute betas
            gpu_rnnt_kernel.compute_betas_kernel[
                self.minibatch_, self.maxU_, self.stream_, 0
            ](
                acts,
                denom,
                betas,
                llBackward,
                input_lengths,
                label_lengths,
                labels,
                self.minibatch_,
                self.maxT_,
                self.maxU_,
                self.alphabet_size_,
                self.blank_,
            )

            # Compute gradient
            grad_blocks_per_grid = self.minibatch_ * self.maxT_ * self.maxU_
            grad_threads_per_block = gpu_rnnt_kernel.GPU_RNNT_THREAD_SIZE
            gpu_rnnt_kernel.compute_grad_kernel[
                grad_blocks_per_grid, grad_threads_per_block, self.stream_, 0
            ](
                grads,
                acts,
                denom,
                alphas,
                betas,
                llForward,
                input_lengths,
                label_lengths,
                labels,
                self.minibatch_,
                self.maxT_,
                self.maxU_,
                self.alphabet_size_,
                self.blank_,
                self.fastemit_lambda_,
                self.clamp_,
            )

        # // cost copy, negate (for log likelihood) and update with additional
        # regularizers This needs to be done via CUDA, because we used temporary
        # memory llForward passed to alpha, which was updated with log likelihoods.
        # But copying this data into a pytorch pointer is more difficult
        # (numba api is one way)
        # Therefore launch a pointwise CUDA kernel to update the costs inplace
        # from data of llForward then negate to compute the loglikelihood.
        threadsperblock = min(costs.shape[0], 32)
        blockspergrid = (costs.shape[0] + (threadsperblock - 1)) // threadsperblock
        rnnt_helper.compute_costs_data[blockspergrid, threadsperblock, self.stream_, 0](
            llForward, costs, self.fastemit_lambda_
        )
        self.stream_.synchronize()

        return global_constants.RNNTStatus.RNNT_STATUS_SUCCESS

    def cost_and_grad(
        self,
        acts: torch.Tensor,
        grads: torch.Tensor,
        costs: torch.Tensor,
        pad_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        """
        Computes the cost and gradients for the given activation tensor.

        This function evaluates the negative log likelihood and computes the
        gradients of the RNNT model with respect to the input activations.
        It is designed to handle cases where the gradients need to be computed
        during training, as well as to compute the forward score when gradients
        are not required.

        Args:
            acts (torch.Tensor): A flattened tensor of shape [B, T, U, V+1]
                representing the activation matrix.
            grads (torch.Tensor): A flattened zero tensor of the same shape as
                `acts`, which will be updated in place to hold the gradients.
            costs (torch.Tensor): A zero vector of length B that will be updated
                in place with the log probability costs.
            pad_labels (torch.Tensor): A flattened matrix of labels of shape [B, U].
            label_lengths (torch.Tensor): A vector of length B containing the
                original lengths of the target sequences.
            input_lengths (torch.Tensor): A vector of length B containing the
                original lengths of the acoustic sequences.

        Returns:
            global_constants.RNNTStatus: An enum that indicates the status of
            the RNNT operation, which can either represent success or failure.

        Raises:
            global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE: If any of the
            input tensors are None.

        Examples:
            >>> acts = torch.randn(2, 10, 5, 20)  # Random activation tensor
            >>> grads = torch.zeros_like(acts)  # Zero gradients tensor
            >>> costs = torch.zeros(2)  # Zero costs vector
            >>> pad_labels = torch.randint(0, 20, (2, 5))  # Random labels
            >>> label_lengths = torch.tensor([5, 4])  # Example lengths
            >>> input_lengths = torch.tensor([10, 10])  # Example lengths
            >>> status = gpurnnt.cost_and_grad(acts, grads, costs, pad_labels,
            ...                                 label_lengths, input_lengths)
            >>> print(status)  # Should print the status of the operation

        Note:
            This method is part of the GPURNNT class, which provides various
            functionalities for working with RNNT models.
        """
        if (
            acts is None
            or grads is None
            or costs is None
            or pad_labels is None
            or label_lengths is None
            or input_lengths is None
        ):
            return global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE

        return self.compute_cost_and_score(
            acts, grads, costs, pad_labels, label_lengths, input_lengths
        )

    def score_forward(
        self,
        acts: torch.Tensor,
        costs: torch.Tensor,
        pad_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        """
        Computes the forward score and updates the costs tensor.

        This method calculates the loss based on the given activation tensor
        and updates the costs tensor with the negative log likelihood. It does
        not compute gradients since the `grads` parameter is set to None.

        Args:
            acts: A flattened tensor of shape [B, T, U, V+1] representing the
                activation matrix.
            costs: A zero vector of length B which will be updated in-place
                with the log probability costs.
            pad_labels: A flattened matrix of labels of shape [B, U].
            label_lengths: A vector of length B that contains the original
                lengths of the target sequences.
            input_lengths: A vector of length B that contains the original
                lengths of the acoustic sequences.

        Returns:
            An enum that either represents a successful RNNT operation or failure.

        Raises:
            ValueError: If any of the input tensors are None.

        Examples:
            >>> acts = torch.rand(2, 5, 6, 10)  # Example activation tensor
            >>> costs = torch.zeros(2)  # Initialize costs tensor
            >>> pad_labels = torch.tensor([[1, 2, 3], [1, 2, 3]])
            >>> label_lengths = torch.tensor([3, 3])
            >>> input_lengths = torch.tensor([5, 5])
            >>> result = model.score_forward(acts, costs, pad_labels,
            ...                               label_lengths, input_lengths)
            >>> print(costs)  # Updated costs after calling score_forward
        """
        if (
            acts is None
            or costs is None
            or pad_labels is None
            or label_lengths is None
            or input_lengths is None
        ):
            return global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE

        return self.compute_cost_and_score(
            acts, None, costs, pad_labels, label_lengths, input_lengths
        )

    def _prepare_workspace(self) -> Tuple[int, Tuple[torch.Tensor, ...]]:
        """Helper method that uses the workspace and constructs slices of it

        that can be used.

        Returns:
            An int, representing the offset of the used workspace (practically, the
            slice of the workspace consumed) A tuple of tensors representing
            the shared workspace.
        """

        used_offset = 0

        # // denom
        denom = self.gpu_workspace[
            used_offset : used_offset + self.maxT_ * self.maxU_ * self.minibatch_
        ]
        used_offset += self.maxT_ * self.maxU_ * self.minibatch_

        # // alphas & betas
        alphas = self.gpu_workspace[
            used_offset : used_offset + self.maxT_ * self.maxU_ * self.minibatch_
        ]
        used_offset += self.maxT_ * self.maxU_ * self.minibatch_
        betas = self.gpu_workspace[
            used_offset : used_offset + self.maxT_ * self.maxU_ * self.minibatch_
        ]
        used_offset += self.maxT_ * self.maxU_ * self.minibatch_

        # // logllh
        llForward = self.gpu_workspace[used_offset : used_offset + self.minibatch_]
        used_offset += self.minibatch_
        llBackward = self.gpu_workspace[used_offset : used_offset + self.minibatch_]
        used_offset += self.minibatch_

        return used_offset, (denom, alphas, betas, llForward, llBackward)


class MultiblankGPURNNT(GPURNNT):
    """
    Helper class to launch the CUDA Kernels to compute Multi-blank Transducer Loss.

    This class extends the GPURNNT class to accommodate multi-blank RNNTs. It utilizes
    CUDA kernels to efficiently compute both the loss and gradients required for
    training multi-blank transducers as described in the paper
    (https://arxiv.org/pdf/2211.03541).

    Attributes:
        sigma (float): Hyper-parameter related to the logit-normalization method in
            training multi-blank transducers.
        num_big_blanks (int): Number of big blank symbols the model has, excluding
            the standard blank symbol.
        big_blank_workspace (torch.Tensor): Allocated memory for multi-blank related
            computations.

    Args:
        sigma: Hyper-parameter related to the logit-normalization method.
        num_big_blanks: Number of big blank symbols the model has.
        minibatch: Int representing the batch size.
        maxT: The maximum possible acoustic sequence length (T in logprobs).
        maxU: The maximum possible target sequence length (U in logprobs).
        alphabet_size: The vocabulary dimension (V + 1 + num_big_blanks).
        workspace: Memory chunk for working memory.
        big_blank_workspace: Memory chunk specifically for multi-blank computations.
        blank: Index of the RNNT blank token in the vocabulary.
        fastemit_lambda: Float scaling factor for FastEmit regularization.
        clamp: Float value to clamp the gradient to [-clamp, clamp].
        num_threads: Number of OMP threads to launch.
        stream: Numba CUDA Stream.

    Examples:
        ```python
        multiblank_rnnt = MultiblankGPURNNT(
            sigma=0.5,
            num_big_blanks=3,
            minibatch=32,
            maxT=100,
            maxU=50,
            alphabet_size=30,
            workspace=torch.zeros(1024).cuda(),
            big_blank_workspace=torch.zeros(512).cuda(),
            blank=0,
            fastemit_lambda=0.1,
            clamp=5.0,
            num_threads=4,
            stream=cuda.stream()
        )
        ```

    Note:
        The `compute_cost_and_score` method computes both the loss and gradients.
        Ensure that all input tensors are properly initialized before calling methods.
    """

    def __init__(
        self,
        sigma: float,
        num_big_blanks: int,
        minibatch: int,
        maxT: int,
        maxU: int,
        alphabet_size: int,
        workspace,
        big_blank_workspace,
        blank: int,
        fastemit_lambda: float,
        clamp: float,
        num_threads: int,
        stream,
    ):
        """Helper class to launch the CUDA Kernels to compute Multi-blank

        Transducer Loss(https://arxiv.org/pdf/2211.03541).

        Args:
            sigma: Hyper-parameter related to the logit-normalization method
                in training multi-blank transducers.
            num_big_blanks: Number of big blank symbols the model has. This should
                not include the standard blank symbol.
            minibatch: Int representing the batch size.
            maxT: The maximum possible acoustic sequence length.
                Represents T in the logprobs tensor.
            maxU: The maximum possible target sequence length.
                Represents U in the logprobs tensor.
            alphabet_size: The vocabulary dimension V + 1 + num-big-blanks
            workspace: An allocated chunk of memory that will be sliced off and
                reshaped into required blocks used as working memory.
            big_blank_workspace: An allocated chunk of memory that will be sliced
                off and reshaped into required blocks used as working memory
                specifically for the multi-blank related computations.
            blank: Index of the RNNT blank token in the vocabulary.
                Generally the first or last token in the vocab.
            fastemit_lambda: Float scaling factor for FastEmit regularization. Refer to
                FastEmit: Low-latency Streaming ASR with
                Sequence-level Emission Regularization.
            clamp: Float value. When set to value >= 0.0, will clamp the
                gradient to [-clamp, clamp].
            num_threads: Number of OMP threads to launch.
            stream: Numba Cuda Stream.
        """

        super().__init__(
            minibatch,
            maxT,
            maxU,
            alphabet_size,
            workspace,
            blank,
            fastemit_lambda,
            clamp,
            num_threads,
            stream,
        )
        self.big_blank_workspace = cuda.as_cuda_array(
            big_blank_workspace
        )  # a flat vector of integer numbers that represents allocated memory slices

        self.num_big_blanks = num_big_blanks
        self.sigma = sigma

    def compute_cost_and_score(
        self,
        acts: torch.Tensor,
        grads: Optional[torch.Tensor],
        costs: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> global_constants.RNNTStatus:
        """
        Compute both the loss and the gradients.

        This method calculates the negative log likelihood loss and, if
        gradients are required, computes the gradients of the activation
        matrix with respect to the costs vector. It utilizes CUDA kernels
        for efficient computation.

        Args:
            acts: A flattened tensor of shape [B, T, U, V+1] representing
                the activation matrix.
            grads: A flattened zero tensor of the same shape as acts,
                which will be updated with the computed gradients.
            costs: A zero vector of length B that will be updated in-place
                with the log probability costs.
            labels: A flattened matrix of labels of shape [B, U].
            label_lengths: A vector of length B that contains the original
                lengths of the target sequence.
            input_lengths: A vector of length B that contains the original
                lengths of the acoustic sequence.

        Updates:
            This method will launch CUDA kernels that update the following
            variables in-place:
            - grads: Gradients of the activation matrix with respect to the
              costs vector.
            - costs: Negative log likelihood of the forward variable.

        Returns:
            An enum representing the status of the RNNT operation, which can
            indicate success or failure.

        Examples:
            # Example usage:
            acts = torch.rand(B, T, U, V + 1)  # Random activation matrix
            grads = torch.zeros_like(acts)      # Initialize gradients
            costs = torch.zeros(B)               # Initialize costs
            labels = torch.randint(0, V, (B, U)) # Random labels
            label_lengths = torch.randint(1, U, (B,))  # Random label lengths
            input_lengths = torch.randint(1, T, (B,))   # Random input lengths

            status = compute_cost_and_score(acts, grads, costs, labels,
                                             label_lengths, input_lengths)

        Note:
            Ensure that the input tensors are properly flattened and
            have the correct shapes as expected by the function.
        """

        training = grads is not None

        if training:
            grads *= 0.0  # zero grads

        _, (
            denom,
            alphas,
            betas,
            llForward,
            llBackward,
            bigblank_durations,
        ) = self._prepare_workspace()

        # START EXECUTION
        self.log_softmax(acts, denom)

        # Compute alphas
        gpu_rnnt_kernel.compute_multiblank_alphas_kernel[
            self.minibatch_, self.maxU_, self.stream_, 0
        ](
            acts,
            denom,
            self.sigma,
            alphas,
            llForward,
            input_lengths,
            label_lengths,
            labels,
            self.minibatch_,
            self.maxT_,
            self.maxU_,
            self.alphabet_size_,
            self.blank_,
            bigblank_durations,
            self.num_big_blanks,
        )

        if training:
            # Compute betas
            gpu_rnnt_kernel.compute_multiblank_betas_kernel[
                self.minibatch_, self.maxU_, self.stream_, 0
            ](
                acts,
                denom,
                self.sigma,
                betas,
                llBackward,
                input_lengths,
                label_lengths,
                labels,
                self.minibatch_,
                self.maxT_,
                self.maxU_,
                self.alphabet_size_,
                self.blank_,
                bigblank_durations,
                self.num_big_blanks,
            )

            # Compute gradient
            grad_blocks_per_grid = self.minibatch_ * self.maxT_ * self.maxU_
            grad_threads_per_block = gpu_rnnt_kernel.GPU_RNNT_THREAD_SIZE
            gpu_rnnt_kernel.compute_multiblank_grad_kernel[
                grad_blocks_per_grid, grad_threads_per_block, self.stream_, 0
            ](
                grads,
                acts,
                denom,
                self.sigma,
                alphas,
                betas,
                llForward,
                input_lengths,
                label_lengths,
                labels,
                self.minibatch_,
                self.maxT_,
                self.maxU_,
                self.alphabet_size_,
                self.blank_,
                bigblank_durations,
                self.num_big_blanks,
                self.fastemit_lambda_,
                self.clamp_,
            )

        # // cost copy, negate (for log likelihood) and update with additional
        # regularizers. This needs to be done via CUDA, because we used temporary
        # memory llForward passed to alpha, which was updated with log likelihoods.
        # But copying this data into a pytorch pointer is more difficult
        # (numba api is one way)
        # Therefore launch a pointwise CUDA kernel to update the costs inplace
        # from data of llForward. Then negate to compute the loglikelihood.
        threadsperblock = min(costs.shape[0], 32)
        blockspergrid = (costs.shape[0] + (threadsperblock - 1)) // threadsperblock
        rnnt_helper.compute_costs_data[blockspergrid, threadsperblock, self.stream_, 0](
            llForward, costs, self.fastemit_lambda_
        )
        self.stream_.synchronize()

        return global_constants.RNNTStatus.RNNT_STATUS_SUCCESS

    def cost_and_grad(
        self,
        acts: torch.Tensor,
        grads: torch.Tensor,
        costs: torch.Tensor,
        pad_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        """
        Computes the cost and gradients of the activation tensor.

        This function checks for the validity of the input tensors and then
        computes the cost and gradients by calling the internal method
        `compute_cost_and_score`. The inputs include the activation tensor,
        gradients tensor, costs tensor, padded labels, label lengths, and
        input lengths.

        Args:
            acts (torch.Tensor): A flattened tensor of shape [B, T, U, V+1]
                representing the activation matrix.
            grads (torch.Tensor): A flattened tensor of the same shape as
                `acts`, which will be updated in place with gradients.
            costs (torch.Tensor): A zero vector of length B that will be
                updated in place with the log probability costs.
            pad_labels (torch.Tensor): A flattened matrix of labels of shape [B, U].
            label_lengths (torch.Tensor): A vector of length B that contains
                the original lengths of the acoustic sequences.
            input_lengths (torch.Tensor): A vector of length B that contains
                the original lengths of the target sequences.

        Returns:
            global_constants.RNNTStatus: An enum indicating either a successful
            RNNT operation or failure due to invalid input.

        Raises:
            global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE: If any
            of the input tensors are None.

        Examples:
            >>> acts = torch.randn(4, 10, 5, 12)  # Example activation tensor
            >>> grads = torch.zeros_like(acts)     # Initialize gradients
            >>> costs = torch.zeros(4)              # Initialize costs
            >>> pad_labels = torch.randint(0, 10, (4, 5))  # Padded labels
            >>> label_lengths = torch.tensor([5, 4, 5, 3])  # Lengths of labels
            >>> input_lengths = torch.tensor([10, 10, 10, 10])  # Input lengths
            >>> result = model.cost_and_grad(acts, grads, costs, pad_labels,
                                              label_lengths, input_lengths)

        Note:
            Ensure that all tensors are on the same device (CPU or GPU) to
            avoid runtime errors.
        """
        if (
            acts is None
            or grads is None
            or costs is None
            or pad_labels is None
            or label_lengths is None
            or input_lengths is None
        ):
            return global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE

        return self.compute_cost_and_score(
            acts, grads, costs, pad_labels, label_lengths, input_lengths
        )

    def score_forward(
        self,
        acts: torch.Tensor,
        costs: torch.Tensor,
        pad_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        """
        Compute the forward score for the RNNT model.

        This function computes the negative log likelihood costs for the given
        activations without calculating gradients. It is useful during inference
        or evaluation where only the score is required.

        Args:
            acts: A tensor of shape [B, T, U, V+1] representing the activation
                matrix from the model, where B is the batch size, T is the
                maximum acoustic sequence length, U is the maximum target
                sequence length, and V is the vocabulary size.
            costs: A tensor of shape [B] that will be updated in-place with
                the log probability costs for each element in the batch.
            pad_labels: A tensor of shape [B, U] containing the padded target
                labels for each element in the batch.
            label_lengths: A tensor of shape [B] containing the actual lengths
                of the target labels for each element in the batch.
            input_lengths: A tensor of shape [B] containing the actual lengths
                of the input sequences for each element in the batch.

        Returns:
            An enum value representing the status of the RNNT operation,
            which can indicate success or failure.

        Raises:
            global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE: If any of
            the input tensors are None.

        Examples:
            >>> acts = torch.rand(2, 10, 5, 20)  # Example activation tensor
            >>> costs = torch.zeros(2)  # Initialize costs tensor
            >>> pad_labels = torch.tensor([[1, 2, 3], [1, 0, 0]])
            >>> label_lengths = torch.tensor([3, 1])
            >>> input_lengths = torch.tensor([10, 10])
            >>> status = model.score_forward(acts, costs, pad_labels,
                                             label_lengths, input_lengths)
            >>> print(costs)  # Updated costs tensor after the call

        Note:
            Ensure that all input tensors are correctly shaped and contain
            valid data before calling this function to avoid errors.
        """
        if (
            acts is None
            or costs is None
            or pad_labels is None
            or label_lengths is None
            or input_lengths is None
        ):
            return global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE

        return self.compute_cost_and_score(
            acts, None, costs, pad_labels, label_lengths, input_lengths
        )

    def _prepare_workspace(self) -> Union[int, Tuple[torch.Tensor]]:
        """Helper method that uses the workspace and constructs slices of it that

        can be used.

        Returns:
            An int, representing the offset of the used workspace (practically,
            the slice of the workspace consumed) A tuple of tensors representing
            the shared workspace.
        """

        used_offset = 0

        # // denom
        denom = self.gpu_workspace[
            used_offset : used_offset + self.maxT_ * self.maxU_ * self.minibatch_
        ]
        used_offset += self.maxT_ * self.maxU_ * self.minibatch_

        # // alphas & betas
        alphas = self.gpu_workspace[
            used_offset : used_offset + self.maxT_ * self.maxU_ * self.minibatch_
        ]
        used_offset += self.maxT_ * self.maxU_ * self.minibatch_
        betas = self.gpu_workspace[
            used_offset : used_offset + self.maxT_ * self.maxU_ * self.minibatch_
        ]
        used_offset += self.maxT_ * self.maxU_ * self.minibatch_

        # // logllh
        llForward = self.gpu_workspace[used_offset : used_offset + self.minibatch_]
        used_offset += self.minibatch_
        llBackward = self.gpu_workspace[used_offset : used_offset + self.minibatch_]
        used_offset += self.minibatch_

        bigblank_durations = self.big_blank_workspace[: self.num_big_blanks]

        return used_offset, (
            denom,
            alphas,
            betas,
            llForward,
            llBackward,
            bigblank_durations,
        )
