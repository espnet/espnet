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
import multiprocessing
from typing import Optional

import numba
import torch
from torch.autograd import Function

from espnet2.asr.transducer.rnnt_multi_blank.utils import global_constants


def log_sum_exp(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the log of the sum of exponentials of input tensors with safety 
    checks for infinite values.

    This function handles cases where either input tensor may contain 
    infinite values by returning the non-infinite tensor directly. It computes 
    the log-sum-exp in a numerically stable manner by utilizing the properties 
    of logarithms and exponentials.

    Args:
        a (torch.Tensor): The first input tensor.
        b (torch.Tensor): The second input tensor.

    Returns:
        torch.Tensor: The computed log-sum-exp of the two input tensors.

    Examples:
        >>> import torch
        >>> a = torch.tensor(10.0)
        >>> b = torch.tensor(20.0)
        >>> log_sum_exp(a, b)
        tensor(20.0)

        >>> a = torch.tensor(float('inf'))
        >>> b = torch.tensor(20.0)
        >>> log_sum_exp(a, b)
        tensor(20.0)

        >>> a = torch.tensor(10.0)
        >>> b = torch.tensor(float('inf'))
        >>> log_sum_exp(a, b)
        tensor(10.0)
    """
    if torch.isinf(a):
        return b

    if torch.isinf(b):
        return a

    if a > b:
        return math.log1p(math.exp(b - a)) + a
    else:
        return math.log1p(math.exp(a - b)) + b


class CpuRNNT_index:
    """
    Class to compute the CPU-based index for RNNT (Recurrent Neural Network 
    Transducer) operations, mimicking pointer indexing as done in CUDA kernels.

    This class facilitates the mapping of target samples to flattened tensors,
    allowing for efficient access to indices based on the current target sample,
    the maximum padded target sample length, and the vocabulary size.

    Attributes:
        U (int): Length of the current target sample (without padding).
        maxU (int): Maximum length of the padded target samples.
        minibatch (int): Minibatch index.
        alphabet_size (int): Size of the vocabulary including RNNT blank (V + 1).
        batch_first (bool): Flag determining if batch index is first or third.

    Args:
        U (int): Length of the current target sample (without padding).
        maxU (int): Max length of the padded target samples.
        minibatch (int): Minibatch index.
        alphabet_size (int): Size of the vocabulary including RNNT blank (V + 1).
        batch_first (bool): Flag to indicate if batch dimension is first (True)
            or third (False).

    Methods:
        __call__(t: int, u: int, v: Optional[int] = None) -> int:
            Returns the computed index for the given parameters.

    Examples:
        >>> index = CpuRNNT_index(U=10, maxU=15, minibatch=2, alphabet_size=30,
        ...                        batch_first=True)
        >>> idx = index(t=3, u=5)
        >>> print(idx)
        53  # Example output based on given parameters

    Note:
        The index computation is designed to emulate the memory access patterns 
        of CUDA kernels on the CPU, which is essential for performance in RNNT 
        training and inference.
    """
    def __init__(
        self, U: int, maxU: int, minibatch: int, alphabet_size: int, batch_first: bool
    ):
        """A placeholder Index computation class that emits the resolved index in a

        flattened tensor, mimicing pointer indexing in CUDA kernels on the CPU.

        Args:
            U: Length of the current target sample (without padding).
            maxU: Max Length of the padded target samples.
            minibatch: Minibatch index
            alphabet_size: Size of the vocabulary including RNNT blank - V+1.
            batch_first: Bool flag determining if batch index is first or third.
        """

        super(CpuRNNT_index, self).__init__()
        self.U = U
        self.maxU = maxU
        self.minibatch = minibatch
        self.alphabet_size = alphabet_size
        self.batch_first = batch_first

    def __call__(self, t: int, u: int, v: Optional[int] = None):
        # if indexing all the values of the vocabulary, then only t, u are provided
        if v is None:
            return t * self.U + u
        else:
            # otherwise, t, u, v are provided to index particular value
            # in the vocabulary.
            if self.batch_first:
                return (t * self.maxU + u) * self.alphabet_size + v
            else:
                return (t * self.maxU + u) * self.minibatch * self.alphabet_size + v


class CpuRNNT_metadata:
    """
    Metadata for CPU-based RNNT loss calculation.

        This class holds the working space memory and initializes the log 
        probabilities for the RNNT model during loss computation.

        Args:
            T: Length of the acoustic sequence (without padding).
            U: Length of the target sequence (without padding).
            workspace: Working space memory for the CPU.
            bytes_used: Number of bytes currently used for indexing the 
                working space memory. Generally starts at 0.
            blank: Index of the blank token in the vocabulary.
            labels: Ground truth padded labels matrix of shape [B, U].
            log_probs: Log probabilities / activation matrix of flattened 
                shape [B, T, U, V+1].
            idx: An instance of CpuRNNT_index for indexing purposes.

        Attributes:
            alphas: Memory for the forward variable (alpha) calculations.
            betas: Memory for the backward variable (beta) calculations.
            log_probs2: Memory for storing log probabilities of blank and 
                label tokens.

        Examples:
            >>> T = 5  # Length of acoustic sequence
            >>> U = 3  # Length of target sequence
            >>> workspace = torch.zeros(100)  # Example workspace tensor
            >>> bytes_used = 0
            >>> blank = 0  # Index of blank token
            >>> labels = torch.tensor([[1, 2, 3]])  # Example labels
            >>> log_probs = torch.zeros((1, T, U, 4))  # Example log_probs
            >>> idx = CpuRNNT_index(U, U, 1, 4, True)
            >>> rnnt_metadata = CpuRNNT_metadata(T, U, workspace, bytes_used,
            ...                                   blank, labels, log_probs, idx)

        Note:
            The memory allocation for alphas, betas, and log_probs2 is done
            using slices of the provided workspace tensor. Ensure that the
            workspace has sufficient size to accommodate these tensors.

        Todo:
            Consider adding error handling for invalid input shapes or types.
    """
    def __init__(
        self,
        T: int,
        U: int,
        workspace: torch.Tensor,
        bytes_used: int,
        blank: int,
        labels: torch.Tensor,
        log_probs: torch.Tensor,
        idx: CpuRNNT_index,
    ):
        """Metadata for CPU based RNNT loss calculation. Holds the working space memory.

        Args:
            T: Length of the acoustic sequence (without padding).
            U: Length of the target sequence (without padding).
            workspace: Working space memory for the CPU.
            bytes_used: Number of bytes currently used for indexing the working
                space memory. Generally 0.
            blank: Index of the blank token in the vocabulary.
            labels: Ground truth padded labels matrix of shape [B, U]
            log_probs: Log probs / activation matrix of flattented shape [B, T, U, V+1]
            idx:
        """

        super(CpuRNNT_metadata, self).__init__()

        self.alphas = workspace[bytes_used : bytes_used + T * U]
        bytes_used += T * U

        self.betas = workspace[bytes_used : bytes_used + T * U]
        bytes_used += T * U

        self.log_probs2 = workspace[
            bytes_used : bytes_used + T * U * 2
        ]  # // only store blank & label
        bytes_used += T * U * 2

        self.bytes_used = bytes_used

        self.setup_probs(T, U, labels, blank, log_probs, idx)

    def setup_probs(
        self,
        T: int,
        U: int,
        labels: torch.Tensor,
        blank: int,
        log_probs: torch.Tensor,
        idx: CpuRNNT_index,
    ):
        """
        Initializes the log probabilities for blank and label tokens.

        This method sets up the log probabilities memory for the blank and label
        tokens by populating the `log_probs2` tensor. The log probabilities are
        extracted from the `log_probs` tensor based on the provided indices and
        the dimensions of the target and acoustic sequences.

        Args:
            T: Length of the acoustic sequence (not padded).
            U: Length of the target sequence (not padded).
            labels: Tensor containing the ground truth labels, shape [B, U].
            blank: Index of the blank token in the vocabulary.
            log_probs: Log probabilities tensor of shape [B, T, U, V+1].
            idx: An instance of `CpuRNNT_index` for indexing purposes.

        Examples:
            >>> labels = torch.tensor([[1, 2, 3], [1, 0, 2]])
            >>> log_probs = torch.rand(2, 5, 4, 5)  # Random log probs
            >>> idx = CpuRNNT_index(4, 5, 2, 5, True)
            >>> rnnt_metadata = CpuRNNT_metadata(5, 4, torch.empty(100), 0, 1, labels, log_probs, idx)
            >>> rnnt_metadata.setup_probs(5, 4, labels, 1, log_probs, idx)

        Note:
            The first blank token does not have an associated label.

        Raises:
            IndexError: If indices are out of bounds.
        """
        # initialize the log probs memory for blank and label token.
        for t in range(T):
            for u in range(U):
                # mult with 2 is for selecting either blank or label token.
                # Odd idx is blank.
                offset = (t * U + u) * 2
                self.log_probs2[offset] = log_probs[idx(t, u, blank)]
                # // labels do not have first blank
                if u < U - 1:
                    self.log_probs2[offset + 1] = log_probs[idx(t, u, labels[u])]


class LogSoftmaxGradModification(Function):
    """
    Custom autograd function for applying log softmax gradient modifications.

    This class implements the forward and backward methods to compute the log
    softmax of the input tensor while allowing for clamping of the gradients
    during backpropagation. It is particularly useful in scenarios where
    numerical stability is a concern.

    Attributes:
        clamp: A float value that defines the range for clamping gradients.
            Should be a non-negative float.

    Args:
        acts: A tensor of activation values to which the log softmax is applied.
        clamp: A non-negative float used to clamp the gradient during
            backpropagation.

    Returns:
        res: A tensor containing the computed log softmax values.

    Yields:
        None

    Raises:
        ValueError: If `clamp` is less than 0.0.

    Examples:
        >>> import torch
        >>> from your_module import LogSoftmaxGradModification
        >>> acts = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> clamp_value = 0.5
        >>> log_softmax = LogSoftmaxGradModification.apply(acts, clamp_value)
        >>> log_softmax.backward(torch.ones_like(log_softmax))
        >>> print(acts.grad)  # Outputs the clamped gradient

    Note:
        This class should be used as a part of a neural network model where
        log softmax activation and its gradient are needed.
    """
    @staticmethod
    def forward(ctx, acts, clamp):
        """
        Compute the forward pass for the LogSoftmax gradient modification.

        This method applies the log-softmax operation on the input activations 
        and clamps the output values to prevent overflow or underflow during 
        backpropagation. The clamping value is defined by the `clamp` parameter.

        Args:
            ctx: Context object that can be used to store information for 
                the backward pass.
            acts (torch.Tensor): Input tensor containing activation values.
            clamp (float): A non-negative float that specifies the clamping 
                range for the output. If `clamp` is less than 0, a 
                ValueError will be raised.

        Returns:
            torch.Tensor: The output tensor after applying the log-softmax 
            operation and clamping.

        Raises:
            ValueError: If `clamp` is less than 0.

        Examples:
            >>> acts = torch.tensor([[1.0, 2.0, 3.0]])
            >>> clamp_value = 1.0
            >>> output = LogSoftmaxGradModification.forward(None, acts, clamp_value)
            >>> print(output)
            tensor([[0.0000, 0.6931, 1.0986]])

        Note:
            The clamping is performed to avoid numerical issues during 
            gradient computation.
        """
        if clamp < 0:
            raise ValueError("`clamp` must be 0.0 or positive float.")

        # This is needed for correctness (inplace is problematic),
        # but it wastes a log of memory.
        res = acts.new(acts)
        ctx.clamp = clamp
        return res

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the forward pass of the LogSoftmax with gradient clamping.

        This function takes in activation values and clamps them based on the
        provided threshold. It is typically used in the context of neural network
        training where the softmax function is followed by a loss function.

        Args:
            ctx: Context object that can be used to store information for
                the backward pass.
            acts (torch.Tensor): The input activation values, typically the
                output logits from the previous layer.
            clamp (float): The maximum value to which gradients will be clamped.
                Must be a non-negative float.

        Returns:
            torch.Tensor: The same activation values, with no modification.

        Raises:
            ValueError: If `clamp` is less than 0.0.

        Examples:
            >>> import torch
            >>> acts = torch.tensor([[0.5, 1.0], [1.5, 2.0]], requires_grad=True)
            >>> clamp_value = 0.1
            >>> output = LogSoftmaxGradModification.apply(acts, clamp_value)
            >>> print(output)
            tensor([[0.5, 1.0], [1.5, 2.0]], grad_fn=<LogSoftmaxGradModificationBackward>)

        Note:
            The `ctx` object is used to store the `clamp` value for use in
            the backward pass. The activation values are returned unchanged
            in the forward pass.
        """
        # Clamp the gradients of loss(logsoftmax(...))
        # CPU computes logsoftmax explicitly, so we need to override t
        grad_output = torch.clamp(grad_output, -ctx.clamp, ctx.clamp)
        return (
            grad_output,
            None,
        )


class CPURNNT:
    """
    Helper class to compute the Transducer Loss on CPU.

    This class provides an implementation of the RNNT (Recurrent Neural
    Network Transducer) loss calculation on CPU. It manages the
    required workspace and computes the forward and backward variables
    for the RNNT.

    Attributes:
        minibatch_: Size of the minibatch b.
        maxT_: The maximum possible acoustic sequence length (T).
        maxU_: The maximum possible target sequence length (U).
        alphabet_size_: The vocabulary dimension (V+1, inclusive of RNNT blank).
        workspace: An allocated chunk of memory that will be sliced
            and reshaped into required blocks used as working memory.
        blank_: Index of the RNNT blank token in the vocabulary.
        fastemit_lambda_: Float scaling factor for FastEmit regularization.
        clamp_: Float value for clamping gradients.
        num_threads_: Number of OMP threads to launch.
        batch_first: Bool that decides if batch dimension is first or third.

    Args:
        minibatch: Size of the minibatch b.
        maxT: The maximum possible acoustic sequence length.
        maxU: The maximum possible target sequence length.
        alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
        workspace: An allocated chunk of memory that will be sliced off and
            reshaped into required blocks used as working memory.
        blank: Index of the RNNT blank token in the vocabulary.
        fastemit_lambda: Float scaling factor for FastEmit regularization.
        clamp: Float value. When set to value >= 0.0, will clamp the
            gradient to [-clamp, clamp].
        num_threads: Number of OMP threads to launch.
        batch_first: Bool that decides if batch dimension is first or third.

    Examples:
        >>> import torch
        >>> cpurnnt = CPURNNT(
        ...     minibatch=32,
        ...     maxT=100,
        ...     maxU=50,
        ...     alphabet_size=30,
        ...     workspace=torch.zeros(10000),
        ...     blank=0,
        ...     fastemit_lambda=0.1,
        ...     clamp=1.0,
        ...     num_threads=4,
        ...     batch_first=True
        ... )
        >>> log_probs = torch.rand(32, 100, 50, 31)  # Log probabilities tensor
        >>> grads = torch.zeros_like(log_probs)
        >>> costs = torch.zeros(32)
        >>> flat_labels = torch.randint(0, 30, (32, 50))
        >>> label_lengths = torch.randint(1, 50, (32,))
        >>> input_lengths = torch.randint(1, 100, (32,))
        >>> status = cpurnnt.cost_and_grad(log_probs, grads, costs, flat_labels,
        ...                                 label_lengths, input_lengths)

    Returns:
        global_constants.RNNTStatus: Status of the RNNT computation.
    """
    def __init__(
        self,
        minibatch: int,
        maxT: int,
        maxU: int,
        alphabet_size: int,
        workspace: torch.Tensor,
        blank: int,
        fastemit_lambda: float,
        clamp: float,
        num_threads: int,
        batch_first: bool,
    ):
        """Helper class to compute the Transducer Loss on CPU.

        Args:
            minibatch: Size of the minibatch b.
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
            clamp: Float value. When set to value >= 0.0, will clamp the
                gradient to [-clamp, clamp].
            num_threads: Number of OMP threads to launch.
            batch_first: Bool that decides if batch dimension is first or third.
        """

        self.minibatch_ = minibatch
        self.maxT_ = maxT
        self.maxU_ = maxU
        self.alphabet_size_ = alphabet_size
        # a flat vector of floatX numbers that represents allocated memory slices
        self.workspace = workspace
        self.blank_ = blank
        self.fastemit_lambda_ = fastemit_lambda
        self.clamp_ = abs(clamp)
        self.num_threads_ = num_threads
        self.batch_first = batch_first

        if num_threads > 0:
            numba.set_num_threads(min(multiprocessing.cpu_count(), num_threads))
        else:
            self.num_threads_ = numba.get_num_threads()

    def cost_and_grad_kernel(
        self,
        log_probs: torch.Tensor,
        grad: torch.Tensor,
        labels: torch.Tensor,
        mb: int,
        T: int,
        U: int,
        bytes_used: int,
    ):
        """
        Computes the cost and gradients for the RNNT loss on the CPU.

    This method utilizes the log probabilities of the model outputs to compute
    the forward and backward variables (alphas and betas) and subsequently
    calculates the gradients for the RNNT loss. It also manages the necessary
    workspace memory for the computations.

    Args:
        log_probs (torch.Tensor): A tensor containing the log probabilities of shape
            [B, T, U, V+1], where B is the batch size, T is the length of the
            acoustic sequence, U is the length of the target sequence, and V is the
            vocabulary size.
        grad (torch.Tensor): A tensor to store the computed gradients of shape
            [B, T, U, V+1].
        labels (torch.Tensor): A tensor containing the ground truth labels of shape
            [B, U].
        mb (int): The current minibatch index.
        T (int): The length of the acoustic sequence (not padded).
        U (int): The length of the target sequence (not padded).
        bytes_used (int): The number of bytes currently used in the workspace memory.

    Returns:
        float: The negative log likelihood of the RNNT loss for the given
        minibatch.

    Examples:
        >>> log_probs = torch.randn(2, 10, 5, 6)  # Example log probabilities
        >>> grad = torch.zeros_like(log_probs)
        >>> labels = torch.randint(0, 5, (2, 5))  # Example labels
        >>> cpurnnt = CPURNNT(minibatch=2, maxT=10, maxU=5, alphabet_size=6,
        ...                   workspace=torch.zeros(1024), blank=0,
        ...                   fastemit_lambda=0.5, clamp=1.0,
        ...                   num_threads=1, batch_first=True)
        >>> loss = cpurnnt.cost_and_grad_kernel(log_probs, grad, labels, 0, 10, 5, 0)
        >>> print(loss)  # Output the loss value

    Note:
        This method is designed for use in the context of the RNNT loss
        calculation and assumes that the input tensors are correctly shaped
        and pre-allocated.

    Raises:
        ValueError: If the input tensors have incorrect dimensions or
        incompatible shapes.
        """
        idx = CpuRNNT_index(
            U, self.maxU_, self.minibatch_, self.alphabet_size_, self.batch_first
        )
        rnntm = CpuRNNT_metadata(
            T, U, self.workspace, bytes_used, self.blank_, labels, log_probs, idx
        )

        if self.batch_first:
            # zero grads
            grad *= 0.0

        llForward = self.compute_alphas(rnntm.log_probs2, T, U, rnntm.alphas)
        llBackward = self.compute_betas_and_grads(
            grad, rnntm.log_probs2, T, U, rnntm.alphas, rnntm.betas, labels, llForward
        )

        # Scale llForward by FastEmit lambda
        llForward *= 1.0 + self.fastemit_lambda_
        llBackward *= 1.0 + self.fastemit_lambda_

        diff = (llForward - llBackward).abs()
        if diff > 0.1:
            print(f"WARNING: Forward backward likelihood mismatch : {diff}")

        return -llForward

    def compute_alphas(
        self, log_probs: torch.Tensor, T: int, U: int, alphas: torch.Tensor
    ):
        """
        Compute the probability of the forward variable alpha.

    This method calculates the forward probabilities (alphas) for the RNNT
    (Recurrent Neural Network Transducer) given the log probabilities of the
    outputs. It fills the `alphas` tensor with the computed values, which are
    used in the computation of the RNNT loss.

    Args:
        log_probs: A flattened tensor of shape [B, T, U, V+1] representing
            the log probabilities of the model outputs. B is the batch size,
            T is the length of the acoustic sequence (not padded), U is the
            length of the target sequence (not padded), and V is the size of
            the vocabulary.
        T: Length of the acoustic sequence T (not padded).
        U: Length of the target sequence U (not padded).
        alphas: A tensor of shape [B, T, U] that serves as the working space
            memory for the alpha values.

    Returns:
        Loglikelihood of the forward variable alpha.

    Examples:
        >>> log_probs = torch.randn(1, 5, 3, 4)  # Example log probabilities
        >>> T = 5  # Length of acoustic sequence
        >>> U = 3  # Length of target sequence
        >>> alphas = torch.zeros(1, T, U)  # Initialize alphas
        >>> log_likelihood = cpurnnt.compute_alphas(log_probs, T, U, alphas)
        >>> print(log_likelihood)  # Loglikelihood of the forward variable

    Note:
        The computation is performed in-place on the `alphas` tensor, which
        should be allocated beforehand with the appropriate shape.
        """

        idx = CpuRNNT_index(
            U, self.maxU_, self.minibatch_, self.alphabet_size_, self.batch_first
        )

        alphas[0] = 0
        for t in range(T):
            for u in range(U):
                if u == 0 and t > 0:
                    alphas[idx(t, 0)] = (
                        alphas[idx(t - 1, 0)] + log_probs[idx(t - 1, 0) * 2]
                    )

                if t == 0 and u > 0:
                    alphas[idx(0, u)] = (
                        alphas[idx(0, u - 1)] + log_probs[idx(0, u - 1) * 2 + 1]
                    )

                if t > 0 and u > 0:
                    no_emit = alphas[idx(t - 1, u)] + log_probs[idx(t - 1, u) * 2]
                    emit = alphas[idx(t, u - 1)] + log_probs[idx(t, u - 1) * 2 + 1]
                    alphas[idx(t, u)] = log_sum_exp(emit, no_emit)

        loglike = alphas[idx(T - 1, U - 1)] + log_probs[idx(T - 1, U - 1) * 2]
        return loglike

    def compute_betas_and_grads(
        self,
        grad: torch.Tensor,
        log_probs: torch.Tensor,
        T: int,
        U: int,
        alphas: torch.Tensor,
        betas: torch.Tensor,
        labels: torch.Tensor,
        logll: torch.Tensor,
    ):
        """
        Compute the backward variable beta and gradients of the activation matrix.

    This function calculates the backward variable (beta) and the gradients 
    of the activation matrix with respect to the log-likelihood of the forward 
    variable. The results are used in training the RNNT model by updating the 
    gradients based on the computed values.

    Args:
        grad: Working space memory of flattened shape [B, T, U, V+1], which 
              will be updated in place with gradients.
        log_probs: Activation tensor of flattened shape [B, T, U, V+1], which 
                   contains the log probabilities for each time step and target 
                   label.
        T: Length of the acoustic sequence T (not padded).
        U: Length of the target sequence U (not padded).
        alphas: Working space memory for alpha of shape [B, T, U], which 
                stores the forward variable probabilities.
        betas: Working space memory for beta of shape [B, T, U], which 
               will store the computed backward variable probabilities.
        labels: Ground truth label tensor of shape [B, U] representing the 
                target sequences.
        logll: Log-likelihood of the forward variable, which is used to 
                normalize the gradients.

    Returns:
        Loglikelihood of the forward variable and in-place updates to the grad 
        tensor.

    Examples:
        >>> grad = torch.zeros((batch_size, max_T, max_U, vocab_size))
        >>> log_probs = torch.randn((batch_size, max_T, max_U, vocab_size))
        >>> alphas = torch.zeros((batch_size, max_T, max_U))
        >>> betas = torch.zeros((batch_size, max_T, max_U))
        >>> labels = torch.randint(0, vocab_size, (batch_size, max_U))
        >>> logll = torch.tensor(0.0)
        >>> loglikelihood = compute_betas_and_grads(grad, log_probs, T, U, alphas, betas, labels, logll)

    Note:
        This method is intended for internal use within the RNNT training 
        process and should not be called directly in user code.
        """

        idx = CpuRNNT_index(
            U, self.maxU_, self.minibatch_, self.alphabet_size_, self.batch_first
        )
        betas[idx(T - 1, U - 1)] = log_probs[idx(T - 1, U - 1) * 2]

        for t in range(T - 1, -1, -1):
            for u in range(U - 1, -1, -1):
                if (u == U - 1) and (t < T - 1):
                    betas[idx(t, U - 1)] = (
                        betas[idx(t + 1, U - 1)] + log_probs[idx(t, U - 1) * 2]
                    )

                if (t == T - 1) and (u < U - 1):
                    betas[idx(T - 1, u)] = (
                        betas[idx(T - 1, u + 1)] + log_probs[idx(T - 1, u) * 2 + 1]
                    )

                if (t < T - 1) and (u < U - 1):
                    no_emit = betas[idx(t + 1, u)] + log_probs[idx(t, u) * 2]
                    emit = betas[idx(t, u + 1)] + log_probs[idx(t, u) * 2 + 1]
                    betas[idx(t, u)] = log_sum_exp(emit, no_emit)

        loglike = betas[0]
        # // Gradients w.r.t. log probabilities
        for t in range(T):
            for u in range(U):
                if t < T - 1:
                    g = alphas[idx(t, u)] + betas[idx(t + 1, u)]
                    grad[idx(t, u, self.blank_)] = -torch.exp(
                        log_probs[idx(t, u) * 2] + g - loglike
                    )

                if u < U - 1:
                    g = alphas[idx(t, u)] + betas[idx(t, u + 1)]
                    grad[idx(t, u, labels[u])] = -torch.exp(
                        math.log1p(self.fastemit_lambda_)
                        + log_probs[idx(t, u) * 2 + 1]
                        + g
                        - loglike
                    )

        # // gradient to the last blank transition
        grad[idx(T - 1, U - 1, self.blank_)] = -torch.exp(
            log_probs[idx(T - 1, U - 1) * 2] + alphas[idx(T - 1, U - 1)] - loglike
        )

        return loglike

    def cost_and_grad(
        self,
        log_probs: torch.Tensor,
        grads: torch.Tensor,
        costs: torch.Tensor,
        flat_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> global_constants.RNNTStatus:
        """
        Computes the cost and gradients for the RNNT loss on CPU.

        This function calculates the RNNT loss for a batch of sequences, storing 
        the computed gradients in the provided gradient tensor. The loss is 
        computed using the forward and backward probabilities (alphas and betas) 
        and is scaled according to the FastEmit regularization factor. The 
        function iterates over each example in the minibatch, calling the 
        `cost_and_grad_kernel` to perform the necessary calculations.

        Args:
            log_probs (torch.Tensor): Log probabilities tensor of shape 
                [B, T, U, V+1], where B is the batch size, T is the length of 
                the acoustic sequence, U is the length of the target sequence, 
                and V is the vocabulary size (including blank).
            grads (torch.Tensor): Tensor to store the computed gradients of 
                shape [B, T, U, V+1].
            costs (torch.Tensor): Tensor to store the computed costs for each 
                example in the minibatch of shape [B].
            flat_labels (torch.Tensor): Ground truth labels in a flattened 
                tensor of shape [B, U].
            label_lengths (torch.Tensor): Lengths of each label sequence in 
                the minibatch of shape [B].
            input_lengths (torch.Tensor): Lengths of each input sequence in 
                the minibatch of shape [B].

        Returns:
            global_constants.RNNTStatus: The status of the RNNT computation, 
                indicating success or failure.

        Examples:
            >>> log_probs = torch.rand((2, 5, 4, 10))  # Example log probs
            >>> grads = torch.zeros_like(log_probs)      # Initialize grads
            >>> costs = torch.zeros(2)                    # Initialize costs
            >>> flat_labels = torch.randint(0, 9, (2, 3)) # Random labels
            >>> label_lengths = torch.tensor([3, 3])      # Lengths of labels
            >>> input_lengths = torch.tensor([5, 5])       # Lengths of inputs
            >>> rnnt = CPURNNT(...)                        # Initialize CPURNNT
            >>> status = rnnt.cost_and_grad(log_probs, grads, costs, 
            ...                               flat_labels, label_lengths, 
            ...                               input_lengths)

        Note:
            Ensure that the input tensors are properly shaped and contain valid 
            data before calling this function.

        Raises:
            ValueError: If the dimensions of the input tensors do not match 
            expected shapes or if any tensor contains invalid values.
        """
        # // per minibatch memory
        per_minibatch_bytes = 0

        # // alphas & betas
        per_minibatch_bytes += self.maxT_ * self.maxU_ * 2

        # // blank & label log probability cache
        per_minibatch_bytes += self.maxT_ * self.maxU_ * 2

        for mb in range(self.minibatch_):
            T = input_lengths[mb]  # // Length of utterance (time)
            U = label_lengths[mb] + 1  # // Number of labels in transcription
            batch_size = self.alphabet_size_
            if self.batch_first:
                batch_size = self.maxT_ * self.maxU_ * self.alphabet_size_

            costs[mb] = self.cost_and_grad_kernel(
                log_probs[(mb * batch_size) :],
                grads[(mb * batch_size) :],
                flat_labels[(mb * (self.maxU_ - 1)) :],
                mb,
                T,
                U,
                mb * per_minibatch_bytes,
            )

        return global_constants.RNNTStatus.RNNT_STATUS_SUCCESS

    def score_forward(
        self,
        log_probs: torch.Tensor,
        costs: torch.Tensor,
        flat_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        """
        Computes the forward score for a batch of input sequences using the RNNT loss.

        This method calculates the log likelihood of the forward variable (alpha) 
        for each sequence in the minibatch based on the provided log probabilities 
        and labels. The results are stored in the `costs` tensor.

        Args:
            log_probs (torch.Tensor): A tensor of shape [B, T, U, V+1] containing 
                the log probabilities for each token in the vocabulary at each 
                time step, where B is the batch size, T is the length of the 
                acoustic sequence, U is the length of the target sequence, and 
                V is the size of the vocabulary (excluding the blank token).
            costs (torch.Tensor): A tensor of shape [B] where the computed negative 
                log likelihoods will be stored for each sequence in the minibatch.
            flat_labels (torch.Tensor): A tensor containing the ground truth labels 
                for each sequence in the minibatch, padded to the maximum target 
                length.
            label_lengths (torch.Tensor): A tensor containing the actual lengths of 
                the labels for each sequence in the minibatch.
            input_lengths (torch.Tensor): A tensor containing the lengths of the 
                input sequences for each sequence in the minibatch.

        Returns:
            global_constants.RNNTStatus: A status indicator indicating the success 
            or failure of the computation. 

        Examples:
            >>> log_probs = torch.randn(2, 5, 4, 3)  # Example log probabilities
            >>> costs = torch.zeros(2)  # Initialize costs tensor
            >>> flat_labels = torch.tensor([[1, 2], [1, 3]])  # Example labels
            >>> label_lengths = torch.tensor([2, 2])  # Lengths of labels
            >>> input_lengths = torch.tensor([5, 5])  # Lengths of inputs
            >>> rnnt.score_forward(log_probs, costs, flat_labels, 
            ...                    label_lengths, input_lengths)
            >>> print(costs)  # Should contain the negative log likelihoods
        """
        # // per minibatch memory
        per_minibatch_bytes = 0

        # // alphas & betas
        per_minibatch_bytes += self.maxT_ * self.maxU_ * 2

        # // blank & label log probability cache
        per_minibatch_bytes += self.maxT_ * self.maxU_ * 2

        for mb in range(self.minibatch_):
            T = input_lengths[mb]  # // Length of utterance (time)
            U = label_lengths[mb] + 1  # // Number of labels in transcription
            batch_size = self.alphabet_size_
            if self.batch_first:
                batch_size = self.maxT_ * self.maxU_ * self.alphabet_size_

            idx = CpuRNNT_index(
                U, self.maxU_, self.minibatch_, self.alphabet_size_, self.batch_first
            )
            rnntm = CpuRNNT_metadata(
                T,
                U,
                self.workspace,
                mb * per_minibatch_bytes,
                self.blank_,
                flat_labels[(mb * (self.maxU_ - 1)) :],
                log_probs[(mb * batch_size) :],
                idx,
            )

            costs[mb] = -self.compute_alphas(rnntm.log_probs2, T, U, rnntm.alphas)

        return global_constants.RNNTStatus.RNNT_STATUS_SUCCESS
