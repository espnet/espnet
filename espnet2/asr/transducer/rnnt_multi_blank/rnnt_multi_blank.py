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

import torch
from torch.autograd import Function
from torch.nn import Module

from espnet2.asr.transducer.rnnt_multi_blank import rnnt
from espnet2.asr.transducer.rnnt_multi_blank.utils.cpu_utils import cpu_rnnt

__all__ = ["rnnt_loss", "RNNTLossNumba", "MultiblankRNNTLossNumba"]


class _RNNTNumba(Function):
    """
    A custom autograd function for computing RNN Transducer (RNNT) loss using Numba.

    This class implements the forward and backward passes for the RNNT loss
    computation, leveraging Numba for efficient CPU and GPU implementations.
    It is designed to be used within PyTorch's autograd system.

    The class supports both standard RNNT loss and FastEmit regularization,
    with options for different reduction methods and gradient clamping.

    Note:
        This is an internal class and should not be instantiated directly.
        Instead, use the `rnnt_loss` function or `RNNTLossNumba` module.
    """

    @staticmethod
    def forward(
        ctx,
        acts,
        labels,
        act_lens,
        label_lens,
        blank,
        reduction,
        fastemit_lambda,
        clamp,
    ):
        """
            Forward pass for the RNN Transducer loss computation.

        This method computes the RNNT loss given the network outputs and labels.
        It supports both CPU and GPU implementations.

        Args:
            ctx (object): Context object to save information for backward pass.
            acts (torch.Tensor): A 4D tensor (batch x seqLength x labelLength x outputDim)
                containing output from network.
            labels (torch.Tensor): 2D tensor containing all the targets of the batch
                with zero padded.
            act_lens (torch.Tensor): 1D tensor of size (batch) containing size of each
                output sequence from the network.
            label_lens (torch.Tensor): 1D tensor of (batch) containing label length
                of each example.
            blank (int): The blank label index.
            reduction (str): Specifies the reduction to apply to the output.
            fastemit_lambda (float): Scaling factor for FastEmit regularization.
            clamp (float): Value for gradient clamping.

        Returns:
            torch.Tensor: The computed RNNT loss.

        Note:
            This method saves gradients in the context for use in the backward pass.
            The actual loss computation is delegated to CUDA or CPU implementations
            based on the input tensor's device.
        """

        is_cuda = acts.is_cuda

        certify_inputs(acts, labels, act_lens, label_lens)
        if clamp < 0:
            raise ValueError("`clamp` must be 0.0 or positive float value.")

        loss_func = rnnt.rnnt_loss_gpu if is_cuda else rnnt.rnnt_loss_cpu
        grads = torch.zeros_like(acts) if acts.requires_grad else None
        minibatch_size = acts.size(0)
        costs = torch.zeros(minibatch_size, device=acts.device, dtype=acts.dtype)

        loss_func(
            acts,
            labels=labels,
            input_lengths=act_lens,
            label_lengths=label_lens,
            costs=costs,
            grads=grads,
            blank_label=blank,
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            num_threads=0,
        )

        if reduction in ["sum", "mean"]:
            costs = costs.sum().unsqueeze_(-1)
            if reduction == "mean":
                costs /= minibatch_size

                if grads is not None:
                    grads /= minibatch_size

        ctx.grads = grads

        return costs

    @staticmethod
    def backward(ctx, grad_output):
        """
            Backward pass for the RNN Transducer loss computation.

        This method computes the gradients of the RNNT loss with respect to the inputs.

        Args:
            ctx (object): Context object containing saved tensors from the forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output
                of the forward pass.

        Returns:
            tuple: A tuple containing the gradients with respect to each input of the
            forward function. The gradients for non-tensor inputs are None.

        Note:
            This method relies on the gradients computed and saved during the forward pass.
            It scales the saved gradients by the incoming gradient and returns them.
            The gradient computation is automatically handled by PyTorch's autograd system.
        """
        if grad_output is not None and ctx.grads is not None:
            grad_output = grad_output.view(-1, 1, 1, 1).to(ctx.grads)
            return ctx.grads.mul_(grad_output), None, None, None, None, None, None, None


class _MultiblankRNNTNumba(Function):
    """
    A custom autograd function for computing Multi-blank RNN Transducer (RNNT) loss using Numba.

    This class implements the forward and backward passes for the Multi-blank RNNT loss
    computation, leveraging Numba for efficient GPU implementations. It is designed to
    be used within PyTorch's autograd system.

    The Multi-blank RNNT loss is an extension of the standard RNNT loss that incorporates
    multiple blank symbols with different durations. This approach can improve the
    performance of speech recognition systems, especially in streaming scenarios.

    The class supports both standard Multi-blank RNNT loss and FastEmit regularization,
    with options for different reduction methods, gradient clamping, and logit
    under-normalization.

    Note:
        This is an internal class and should not be instantiated directly.
        Instead, use the `multiblank_rnnt_loss` function or `MultiblankRNNTLossNumba` module.

    Reference:
        https://arxiv.org/pdf/2211.03541.pdf
    """

    @staticmethod
    def forward(
        ctx,
        acts,
        labels,
        act_lens,
        label_lens,
        blank,
        big_blank_durations,
        reduction,
        fastemit_lambda,
        clamp,
        sigma,
    ):
        """
            Forward pass for the Multi-blank RNN Transducer loss computation.

        This method computes the Multi-blank RNNT loss given the network outputs and labels.
        It currently supports only GPU implementations.

        Args:
            ctx (object): Context object to save information for backward pass.
            acts (torch.Tensor): A 4D tensor (batch x seqLength x labelLength x outputDim)
                containing output from network.
            labels (torch.Tensor): 2D tensor containing all the targets of the batch
                with zero padded.
            act_lens (torch.Tensor): 1D tensor of size (batch) containing size of each
                output sequence from the network.
            label_lens (torch.Tensor): 1D tensor of (batch) containing label length
                of each example.
            blank (int): The standard blank label index.
            big_blank_durations (list): List of durations for multi-blank transducer.
            reduction (str): Specifies the reduction to apply to the output.
            fastemit_lambda (float): Scaling factor for FastEmit regularization.
            clamp (float): Value for gradient clamping.
            sigma (float): Hyper-parameter for logit under-normalization method.

        Returns:
            torch.Tensor: The computed Multi-blank RNNT loss.

        Raises:
            NotImplementedError: If attempting to use CPU implementation.

        Note:
            This method saves gradients in the context for use in the backward pass.
            The actual loss computation is delegated to the GPU implementation.
        """

        is_cuda = acts.is_cuda

        certify_inputs(acts, labels, act_lens, label_lens)
        if clamp < 0:
            raise ValueError("`clamp` must be 0.0 or positive float value.")

        if is_cuda:
            loss_func = rnnt.multiblank_rnnt_loss_gpu
        else:
            raise NotImplementedError()

        grads = torch.zeros_like(acts) if acts.requires_grad else None
        minibatch_size = acts.size(0)
        costs = torch.zeros(minibatch_size, device=acts.device, dtype=acts.dtype)

        loss_func(
            acts,
            labels=labels,
            input_lengths=act_lens,
            label_lengths=label_lens,
            costs=costs,
            grads=grads,
            blank_label=blank,
            big_blank_durations=big_blank_durations,
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            sigma=sigma,
            num_threads=0,
        )

        if reduction in ["sum", "mean"]:
            costs = costs.sum().unsqueeze_(-1)
            if reduction == "mean":
                costs /= minibatch_size

                if grads is not None:
                    grads /= minibatch_size

        ctx.grads = grads

        return costs

    @staticmethod
    def backward(ctx, grad_output):
        """
            Backward pass for the Multi-blank RNN Transducer loss computation.

        This method computes the gradients of the Multi-blank RNNT loss with respect to the inputs.

        Args:
            ctx (object): Context object containing saved tensors from the forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output
                of the forward pass.

        Returns:
            tuple: A tuple containing the gradients with respect to each input of the
            forward function. The gradients for non-tensor inputs are None.

        Note:
            This method relies on the gradients computed and saved during the forward pass.
            It scales the saved gradients by the incoming gradient and returns them.
            The gradient computation is automatically handled by PyTorch's autograd system.

            The returned tuple has 11 elements to match the number of inputs in the forward method,
            with None values for inputs that don't require gradients.
        """
        if grad_output is not None and ctx.grads is not None:
            grad_output = grad_output.view(-1, 1, 1, 1).to(ctx.grads)
            return (
                ctx.grads.mul_(grad_output),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )


def rnnt_loss(
    acts,
    labels,
    act_lens,
    label_lens,
    blank=0,
    reduction="mean",
    fastemit_lambda: float = 0.0,
    clamp: float = 0.0,
):
    """
    Compute the RNN Transducer Loss.

    This function calculates the RNN Transducer Loss, which is commonly used in
    speech recognition tasks. It supports both CPU and GPU computations.

    Args:
        acts (torch.Tensor): A 4D tensor of shape (batch, seqLength, labelLength, outputDim)
            containing the output from the network.
        labels (torch.Tensor): A 2D tensor containing all the targets of the batch
            with zero padding.
        act_lens (torch.Tensor): A 1D tensor of size (batch) containing the size of
            each output sequence from the network.
        label_lens (torch.Tensor): A 1D tensor of size (batch) containing the label
            length of each example.
        blank (int, optional): The blank label index. Defaults to 0.
        reduction (str, optional): Specifies the reduction to apply to the output.
            Options are 'none', 'mean', or 'sum'. Defaults to 'mean'.
        fastemit_lambda (float, optional): Scaling factor for FastEmit regularization.
            Defaults to 0.0.
        clamp (float, optional): Value for gradient clamping. If positive, gradients
            will be clamped to [-clamp, clamp]. Defaults to 0.0.

    Returns:
        torch.Tensor: The computed RNN Transducer Loss.

    Raises:
        ValueError: If `clamp` is negative.

    Note:
        For CPU computations, log_softmax is applied manually, while for GPU
        computations, it's computed within the CUDA kernel.

    Example:
        >>> acts = torch.randn(2, 10, 5, 20)
        >>> labels = torch.randint(0, 19, (2, 5))
        >>> act_lens = torch.tensor([10, 8])
        >>> label_lens = torch.tensor([5, 4])
        >>> loss = rnnt_loss(acts, labels, act_lens, label_lens)
    """

    if not acts.is_cuda:
        # Since CPU requires log_softmax to be computed explicitly,
        # we need to perform grad clipping
        # *after* we have obtained the gradients of loss(logsoftmax()).
        # This is highly wasteful since it requires a copy of the entire joint
        # tensor which is expensive. CUDA version is much more efficient since
        # it performs an inplace logsoftmax, and therefore
        # can inplace clamp the gradient.
        if clamp > 0.0:
            acts = cpu_rnnt.LogSoftmaxGradModification.apply(acts, clamp)

        # NOTE manually done log_softmax for CPU version,
        # log_softmax is computed within GPU version.
        acts = torch.nn.functional.log_softmax(acts, -1)

    return _RNNTNumba.apply(
        acts, labels, act_lens, label_lens, blank, reduction, fastemit_lambda, clamp
    )


def multiblank_rnnt_loss(
    acts,
    labels,
    act_lens,
    label_lens,
    blank,
    big_blank_durations=[],
    reduction="mean",
    fastemit_lambda: float = 0.0,
    clamp: float = 0.0,
):
    """
    Compute the Multi-blank RNN Transducer Loss.

    This function calculates the Multi-blank RNN Transducer Loss, which is an extension
    of the standard RNN Transducer Loss that incorporates multiple blank symbols with
    different durations. It is designed to improve the performance of speech recognition
    systems, especially for streaming scenarios.

    Args:
        acts (torch.Tensor): A 4D tensor of shape (batch, seqLength, labelLength, outputDim)
            containing the output from the network.
        labels (torch.Tensor): A 2D tensor containing all the targets of the batch
            with zero padding.
        act_lens (torch.Tensor): A 1D tensor of size (batch) containing the size of
            each output sequence from the network.
        label_lens (torch.Tensor): A 1D tensor of size (batch) containing the label
            length of each example.
        blank (int): The standard blank label index.
        big_blank_durations (list, optional): List of durations for multi-blank transducer,
            e.g., [2, 4, 8]. Defaults to an empty list.
        reduction (str, optional): Specifies the reduction to apply to the output.
            Options are 'none', 'mean', or 'sum'. Defaults to 'mean'.
        fastemit_lambda (float, optional): Scaling factor for FastEmit regularization.
            Defaults to 0.0.
        clamp (float, optional): Value for gradient clamping. If positive, gradients
            will be clamped to [-clamp, clamp]. Defaults to 0.0.

    Returns:
        torch.Tensor: The computed Multi-blank RNN Transducer Loss.

    Raises:
        ValueError: If `clamp` is negative.
        NotImplementedError: If trying to use CPU for computation (currently only GPU is supported).

    Note:
        This implementation is based on the paper "Multi-blank Transducers for Speech Recognition"
        (https://arxiv.org/pdf/2211.03541.pdf). It's designed to work with CUDA-enabled devices.

    Example:
        >>> acts = torch.randn(2, 10, 5, 20).cuda()
        >>> labels = torch.randint(0, 19, (2, 5)).cuda()
        >>> act_lens = torch.tensor([10, 8]).cuda()
        >>> label_lens = torch.tensor([5, 4]).cuda()
        >>> blank = 0
        >>> big_blank_durations = [2, 4]
        >>> loss = multiblank_rnnt_loss(acts, labels, act_lens, label_lens, blank, big_blank_durations)
    """

    if not acts.is_cuda:
        # Since CPU requires log_softmax to be computed explicitly,
        # we need to perform grad clipping
        # *after* we have obtained the gradients of loss(logsoftmax()).
        # This is highly wasteful since it requires a copy of the entire
        # joint tensor which is expensive.
        # CUDA version is much more efficient since it performs an inplace
        # logsoftmax, and therefore can inplace clamp the gradient.
        if clamp > 0.0:
            acts = cpu_rnnt.LogSoftmaxGradModification.apply(acts, clamp)

        # NOTE manually done log_softmax for CPU version,
        # log_softmax is computed within GPU version.
        acts = torch.nn.functional.log_softmax(acts, -1)

    return _MultiblankRNNTNumba.apply(
        acts,
        labels,
        act_lens,
        label_lens,
        blank,
        big_blank_durations,
        reduction,
        fastemit_lambda,
        clamp,
    )


class RNNTLossNumba(Module):
    """
    A PyTorch module for computing RNN Transducer (RNNT) loss using Numba.

    This module provides an efficient implementation of the RNNT loss function,
    leveraging Numba for improved performance. It supports both CPU and GPU
    computations, with options for different reduction methods, FastEmit
    regularization, and gradient clamping.

    The RNNT loss is commonly used in speech recognition tasks, particularly
    for training end-to-end models.

    Attributes:
        blank (int): The blank label index. Default is 0.
        reduction (str): Specifies the reduction to apply to the output.
            Can be 'none', 'mean', or 'sum'. Default is 'mean'.
        fastemit_lambda (float): Scaling factor for FastEmit regularization.
            Default is 0.0.
        clamp (float): Value for gradient clamping. When set to a value >= 0.0,
            will clamp the gradient to [-clamp, clamp]. Default is -1 (no clamping).

    Note:
        This module uses the `_RNNTNumba` function for the actual loss computation.
    """

    def __init__(
        self, blank=0, reduction="mean", fastemit_lambda: float = 0.0, clamp: float = -1
    ):
        super(RNNTLossNumba, self).__init__()
        self.blank = blank
        self.fastemit_lambda = fastemit_lambda
        self.clamp = float(clamp) if clamp > 0 else 0.0
        self.reduction = reduction
        self.loss = _RNNTNumba.apply

    def forward(self, acts, labels, act_lens, label_lens):
        """
            Forward pass for computing the RNN Transducer loss.

        This method calculates the RNNT loss given the network outputs and labels.
        It handles both CPU and GPU implementations, applying necessary preprocessing
        steps depending on the device.

        Args:
            acts (torch.Tensor): A 4D tensor of shape (batch x seqLength x labelLength x outputDim)
                containing output from the network.
            labels (torch.Tensor): A 2D tensor containing all the targets of the batch
                with zero padding.
            act_lens (torch.Tensor): A 1D tensor of size (batch) containing the size of each
                output sequence from the network.
            label_lens (torch.Tensor): A 1D tensor of size (batch) containing the label
                length of each example.

        Returns:
            torch.Tensor: The computed RNNT loss.

        Note:
            For CPU computations, this method manually applies log_softmax and handles
            gradient clamping if specified. For GPU computations, these operations are
            performed within the CUDA kernel for efficiency.
        """

        if not acts.is_cuda:
            # Since CPU requires log_softmax to be computed explicitly,
            # we need to perform grad clipping
            # *after* we have obtained the gradients of loss(logsoftmax()).
            # This is highly wasteful since it requires a copy of the entire
            # joint tensor which is expensive.
            # CUDA version is much more efficient since it performs an
            # inplace logsoftmax, and therefore can inplace clamp the gradient.
            if self.clamp > 0.0:
                acts = cpu_rnnt.LogSoftmaxGradModification.apply(acts, self.clamp)

            # NOTE manually done log_softmax for CPU version,
            # log_softmax is computed within GPU version.
            acts = torch.nn.functional.log_softmax(acts, -1)

        return self.loss(
            acts,
            labels,
            act_lens,
            label_lens,
            self.blank,
            self.reduction,
            self.fastemit_lambda,
            self.clamp,
        )


class MultiblankRNNTLossNumba(Module):
    """
    A PyTorch module for computing Multi-blank RNN Transducer (RNNT) loss using Numba.

    This module implements the Multi-blank RNNT loss function, which is an extension
    of the standard RNNT loss that incorporates multiple blank symbols with different
    durations. It is designed to improve the performance of speech recognition systems,
    especially in streaming scenarios.

    The implementation uses Numba for efficient computation and currently supports
    GPU operations only.

    Attributes:
        blank (int): The standard blank label index.
        big_blank_durations (list): List of durations for multi-blank transducer, e.g., [2, 4, 8].
        reduction (str): Specifies the reduction to apply to the output.
            Can be 'none', 'mean', or 'sum'. Default is 'mean'.
        fastemit_lambda (float): Scaling factor for FastEmit regularization. Default is 0.0.
        clamp (float): Value for gradient clamping. When set to a value >= 0.0,
            will clamp the gradient to [-clamp, clamp]. Default is -1 (no clamping).
        sigma (float): Hyper-parameter for logit under-normalization method. Default is 0.0.

    Note:
        This module uses the `_MultiblankRNNTNumba` function for the actual loss computation.

    Reference:
        https://arxiv.org/pdf/2211.03541.pdf
    """

    def __init__(
        self,
        blank,
        big_blank_durations,
        reduction="mean",
        fastemit_lambda: float = 0.0,
        clamp: float = -1,
        sigma: float = 0.0,
    ):
        super(MultiblankRNNTLossNumba, self).__init__()
        self.blank = blank
        self.big_blank_durations = big_blank_durations
        self.fastemit_lambda = fastemit_lambda
        self.clamp = float(clamp) if clamp > 0 else 0.0
        self.reduction = reduction
        self.loss = _MultiblankRNNTNumba.apply
        self.sigma = sigma

    def forward(self, acts, labels, act_lens, label_lens):
        """
            Forward pass for computing the Multi-blank RNN Transducer loss.

        This method calculates the Multi-blank RNNT loss given the network outputs and labels.
        It currently supports only GPU implementations, applying necessary preprocessing
        steps before the loss computation.

        Args:
            acts (torch.Tensor): A 4D tensor of shape (batch x seqLength x labelLength x outputDim)
                containing output from the network.
            labels (torch.Tensor): A 2D tensor containing all the targets of the batch
                with zero padding.
            act_lens (torch.Tensor): A 1D tensor of size (batch) containing the size of each
                output sequence from the network.
            label_lens (torch.Tensor): A 1D tensor of size (batch) containing the label
                length of each example.

        Returns:
            torch.Tensor: The computed Multi-blank RNNT loss.

        Raises:
            NotImplementedError: If attempting to use CPU for computation.

        Note:
            For GPU computations, this method manually applies log_softmax and handles
            gradient clamping if specified. The actual loss computation is performed
            within the CUDA kernel for efficiency.
        """

        if not acts.is_cuda:
            # Since CPU requires log_softmax to be computed explicitly,
            # we need to perform grad clipping
            # *after* we have obtained the gradients of loss(logsoftmax()).
            # This is highly wasteful since it requires a copy of the entire
            # joint tensor which is expensive.
            # CUDA version is much more efficient since it performs an
            # inplace logsoftmax, and therefore can inplace clamp the gradient.
            if self.clamp > 0.0:
                acts = cpu_rnnt.LogSoftmaxGradModification.apply(acts, self.clamp)

            # NOTE manually done log_softmax for CPU version,
            # log_softmax is computed within GPU version.
            acts = torch.nn.functional.log_softmax(acts, -1)

        return self.loss(
            acts,
            labels,
            act_lens,
            label_lens,
            self.blank,
            self.big_blank_durations,
            self.reduction,
            self.fastemit_lambda,
            self.clamp,
            self.sigma,
        )


def check_type(var, t, name):
    """
    Check if a variable has the expected data type.

    This function verifies whether the given variable has the specified data type.
    If the variable's type doesn't match the expected type, it raises a TypeError.

    Args:
        var (Any): The variable to check.
        t (type): The expected data type.
        name (str): The name of the variable (used in the error message).

    Raises:
        TypeError: If the variable's type doesn't match the expected type.

    Example:
        >>> import torch
        >>> tensor = torch.tensor([1, 2, 3])
        >>> check_type(tensor, torch.Tensor, "tensor")
        >>> check_type(tensor, torch.float32, "tensor")
        TypeError: tensor must be torch.float32
    """
    if var.dtype is not t:
        raise TypeError("{} must be {}".format(name, t))


def check_contiguous(var, name):
    """
    Check if a tensor is contiguous in memory.

    This function verifies whether the given tensor is contiguous in memory.
    If the tensor is not contiguous, it raises a ValueError.

    Args:
        var (torch.Tensor): The tensor to check for contiguity.
        name (str): The name of the tensor (used in the error message).

    Raises:
        ValueError: If the tensor is not contiguous in memory.

    Example:
        >>> import torch
        >>> tensor = torch.tensor([[1, 2], [3, 4]])
        >>> check_contiguous(tensor, "tensor")
        >>> non_contiguous = tensor.t()
        >>> check_contiguous(non_contiguous, "non_contiguous")
        ValueError: non_contiguous must be contiguous
    """
    if not var.is_contiguous():
        raise ValueError("{} must be contiguous".format(name))


def check_dim(var, dim, name):
    """
    Check if a tensor has the expected number of dimensions.

    This function verifies whether the given tensor has the specified number of dimensions.
    If the tensor's dimensionality doesn't match the expected value, it raises a ValueError.

    Args:
        var (torch.Tensor): The tensor to check.
        dim (int): The expected number of dimensions.
        name (str): The name of the tensor (used in the error message).

    Raises:
        ValueError: If the tensor's number of dimensions doesn't match the expected value.

    Example:
        >>> import torch
        >>> tensor_2d = torch.tensor([[1, 2], [3, 4]])
        >>> check_dim(tensor_2d, 2, "tensor_2d")
        >>> tensor_3d = torch.ones(2, 3, 4)
        >>> check_dim(tensor_3d, 2, "tensor_3d")
        ValueError: tensor_3d must be 2D
    """
    if len(var.shape) != dim:
        raise ValueError("{} must be {}D".format(name, dim))


def certify_inputs(log_probs, labels, lengths, label_lengths):
    # check_type(log_probs, torch.float32, "log_probs")
    """
    Certify that the input tensors meet the required specifications for RNNT loss computation.

    This function performs a series of checks on the input tensors to ensure they meet
    the necessary requirements for computing the RNNT loss. It verifies data types,
    contiguity, dimensions, and shape consistency.

    Args:
        log_probs (torch.Tensor): A 4D tensor of log probabilities from the network output.
        labels (torch.Tensor): A 2D tensor of label sequences.
        lengths (torch.Tensor): A 1D tensor of input sequence lengths.
        label_lengths (torch.Tensor): A 1D tensor of label sequence lengths.

    Raises:
        TypeError: If any tensor has an incorrect data type.
        ValueError: If any tensor is not contiguous, has incorrect dimensions,
                    or if there's a mismatch in batch sizes or sequence lengths.

    Note:
        This function is typically called internally by the RNNT loss function
        to validate inputs before computation.

    Example:
        >>> log_probs = torch.randn(2, 10, 5, 20, dtype=torch.float32)
        >>> labels = torch.randint(0, 19, (2, 5), dtype=torch.int32)
        >>> lengths = torch.tensor([10, 8], dtype=torch.int32)
        >>> label_lengths = torch.tensor([5, 4], dtype=torch.int32)
        >>> certify_inputs(log_probs, labels, lengths, label_lengths)
    """
    check_type(label_lengths, torch.int32, "label_lengths")
    check_type(lengths, torch.int32, "lengths")
    check_contiguous(log_probs, "log_probs")
    check_contiguous(labels, "labels")
    check_contiguous(label_lengths, "label_lengths")
    check_contiguous(lengths, "lengths")

    if lengths.shape[0] != log_probs.shape[0]:
        raise ValueError(
            f"Must have a length per example. "
            f"Given lengths dim: {lengths.shape[0]}, "
            f"Log probs dim : {log_probs.shape[0]}"
        )
    if label_lengths.shape[0] != log_probs.shape[0]:
        raise ValueError(
            "Must have a label length per example. "
            f"Given label lengths dim : {label_lengths.shape[0]}, "
            f"Log probs dim : {log_probs.shape[0]}"
        )

    check_dim(log_probs, 4, "log_probs")
    check_dim(labels, 2, "labels")
    check_dim(lengths, 1, "lenghts")
    check_dim(label_lengths, 1, "label_lenghts")
    max_T = torch.max(lengths)
    max_U = torch.max(label_lengths)
    T, U = log_probs.shape[1:3]
    if T != max_T:
        raise ValueError(
            f"Input length mismatch! Given T: {T}, Expected max T from input \
             lengths: {max_T}"
        )
    if U != max_U + 1:
        raise ValueError(
            f"Output length mismatch! Given U: {U}, Expected max U from target \
             lengths: {max_U} + 1"
        )
