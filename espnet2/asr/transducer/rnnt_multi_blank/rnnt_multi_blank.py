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
    RNNT Loss Numba class for computing RNN Transducer loss using Numba.

    This class implements the forward and backward passes for the RNNT loss
    computation, leveraging Numba for performance optimizations. It takes as
    input the output activations from a neural network and the corresponding
    labels to compute the loss and gradients.

    Attributes:
        None

    Args:
        ctx: The context object that can be used to store information for the
            backward pass.
        acts (Tensor): Tensor of shape (batch x seqLength x labelLength x
            outputDim) containing the output from the network.
        labels (Tensor): 2D Tensor containing all the targets of the batch
            with zero padding.
        act_lens (Tensor): 1D Tensor of size (batch) containing the size of
            each output sequence from the network.
        label_lens (Tensor): 1D Tensor of size (batch) containing the label
            lengths of each example.
        blank (int): The label index for the blank token in the RNNT loss.
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.
        fastemit_lambda (float): Scaling factor for FastEmit regularization.
            Refer to FastEmit: Low-latency Streaming ASR with Sequence-level
            Emission Regularization.
        clamp (float): Value to clamp gradients to. Must be non-negative.

    Returns:
        Tensor: The computed loss values for the input batch.

    Raises:
        ValueError: If `clamp` is less than 0.0.

    Examples:
        # Example usage:
        acts = torch.randn(2, 10, 5, 20)  # (batch x seqLength x labelLength x outputDim)
        labels = torch.randint(0, 5, (2, 3))  # (batch x maxLabelLength)
        act_lens = torch.tensor([10, 8])  # Lengths of each output sequence
        label_lens = torch.tensor([3, 2])  # Lengths of each label sequence
        loss = _RNNTNumba.apply(acts, labels, act_lens, label_lens,
                                 blank=0, reduction='mean',
                                 fastemit_lambda=0.0, clamp=0.0)
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
        Compute the forward pass of the RNNT loss using Numba.

        This method computes the RNN Transducer (RNNT) loss for a given
        set of inputs. The RNNT loss is useful for training sequence-to-sequence
        models, particularly in automatic speech recognition tasks.

        Args:
            ctx: Context object that can be used to store information
                for backward computation.
            acts (Tensor): A tensor of shape (batch x seqLength x labelLength x outputDim)
                containing the output probabilities from the network.
            labels (Tensor): A 2-dimensional tensor containing the target
                labels for each example in the batch, padded with zeros.
            act_lens (Tensor): A tensor of size (batch) that indicates the
                actual length of each output sequence from the network.
            label_lens (Tensor): A tensor of size (batch) that contains the
                length of each target label sequence.
            blank (int): The index of the blank label. Default is 0.
            reduction (str): Specifies the reduction method to apply to the output:
                'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                'mean': the output losses will be divided by the number of target
                lengths and then the mean over the batch is taken. Default is 'mean'.
            fastemit_lambda (float): A scaling factor for FastEmit regularization.
                Refer to the FastEmit paper for details.
            clamp (float): A value to clamp the gradients. Must be 0.0 or greater.

        Returns:
            Tensor: A tensor containing the computed RNNT loss for the batch.

        Raises:
            ValueError: If `clamp` is negative.

        Examples:
            >>> acts = torch.randn(2, 10, 5, 20)  # Example output from the network
            >>> labels = torch.tensor([[1, 2, 0], [1, 0, 0]])  # Padded labels
            >>> act_lens = torch.tensor([10, 5])  # Actual lengths of acts
            >>> label_lens = torch.tensor([2, 1])  # Actual lengths of labels
            >>> loss = _RNNTNumba.forward(None, acts, labels, act_lens, label_lens,
            ...                            blank=0, reduction='mean',
            ...                            fastemit_lambda=0.0, clamp=0.0)
            >>> print(loss)
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
        Backward pass for the RNNTNumba function.

        This method computes the gradients of the RNNT loss with respect to the
        input activations. It uses the stored gradients from the forward pass to
        apply the chain rule, allowing for backpropagation through the network.

        Args:
            ctx: The context object that contains information from the forward pass.
            grad_output: A tensor containing the gradient of the loss with respect
                to the output of the RNNT loss function.

        Returns:
            A tuple containing the gradients of the input activations and None for
            the other parameters that do not require gradients.

        Note:
            This method assumes that the gradient output and stored gradients are
            not None. If grad_output is None, it will not compute any gradients.

        Examples:
            >>> # Assuming acts, labels, act_lens, label_lens, and other required
            >>> # parameters have been defined and used in the forward pass
            >>> loss_fn = _RNNTNumba.apply(acts, labels, act_lens, label_lens)
            >>> # Now compute gradients during the backward pass
            >>> loss_fn.backward(grad_output)
        """
        if grad_output is not None and ctx.grads is not None:
            grad_output = grad_output.view(-1, 1, 1, 1).to(ctx.grads)
            return ctx.grads.mul_(grad_output), None, None, None, None, None, None, None


class _MultiblankRNNTNumba(Function):
    """
    Numba class for multi-blank RNN Transducer loss.

    This class implements the forward and backward passes for a multi-blank
    RNN Transducer (RNNT) loss function using Numba. The multi-blank RNNT
    allows for multiple blank durations during training, which can improve
    the performance of automatic speech recognition models by better
    accommodating varying input lengths.

    References:
        - https://arxiv.org/pdf/2211.03541.pdf

    Attributes:
        None

    Args:
        ctx: The context object for storing information for the backward pass.
        acts (torch.Tensor): A tensor of shape (batch x seqLength x labelLength x
            outputDim) containing the output from the network.
        labels (torch.Tensor): A 2D tensor containing the targets for the
            batch, zero-padded.
        act_lens (torch.Tensor): A tensor of size (batch) containing the size
            of each output sequence from the network.
        label_lens (torch.Tensor): A tensor of size (batch) containing the
            label lengths for each example.
        blank (int): The index of the blank label.
        big_blank_durations (list): A list of durations for the multi-blank
            transducer, e.g., [2, 4, 8].
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default is 'mean'.
        fastemit_lambda (float): Scaling factor for FastEmit regularization.
        clamp (float): A non-negative float value for clamping gradients.
            Must be >= 0.
        sigma (float): Hyper-parameter for logit under-normalization method for
            training multi-blank transducers. Recommended value is 0.05.

    Returns:
        torch.Tensor: The computed costs as a tensor.

    Raises:
        ValueError: If `clamp` is negative or if any of the tensor shapes
        do not match the expected dimensions.

    Examples:
        # Example usage of the forward method:
        loss = _MultiblankRNNTNumba.forward(
            ctx,
            acts,
            labels,
            act_lens,
            label_lens,
            blank=0,
            big_blank_durations=[2, 4, 8],
            reduction='mean',
            fastemit_lambda=0.0,
            clamp=0.0,
            sigma=0.05
        )

    Note:
        This implementation is specifically designed for GPU usage.
        If inputs are on CPU, consider using the appropriate CPU loss
        functions instead.
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
            MultiblankRNNTNumba Forward.

        This method computes the forward pass for the multi-blank RNNT loss
        using Numba for performance optimization. It takes the predicted
        logits, target labels, and their respective lengths to calculate
        the loss values.

        Args:
            ctx: The context object to save information for backward pass.
            acts (Tensor): Tensor of shape (batch x seqLength x labelLength x outputDim)
                containing output logits from the network.
            labels (Tensor): 2D tensor containing target labels for the batch,
                zero-padded as necessary.
            act_lens (Tensor): 1D tensor of size (batch) containing the
                lengths of each output sequence from the network.
            label_lens (Tensor): 1D tensor of size (batch) containing the
                lengths of the labels for each example.
            blank (int): The blank label used in RNNT. Default is 0.
            big_blank_durations (list): List of durations for multi-blank
                transducer, e.g., [2, 4, 8].
            reduction (str): Specifies the reduction method to apply to the output:
                'none', 'mean', or 'sum'. Default is 'mean'.
            fastemit_lambda (float): Scaling factor for FastEmit regularization.
                Refer to the FastEmit paper for more details.
            clamp (float): Value to clamp the gradients to, must be >= 0.0.
            sigma (float): Hyper-parameter for logit under-normalization method
                for training multi-blank transducers. Recommended value is 0.05.

        Returns:
            Tensor: A tensor containing the computed loss for each example in
            the batch, reduced according to the specified method.

        Raises:
            ValueError: If `clamp` is negative, or if the input tensors are
            not of the expected dimensions or types.

        Examples:
            >>> acts = torch.randn(4, 10, 5, 20)  # Example logits
            >>> labels = torch.tensor([[1, 2, 3, 0], [1, 0, 0, 0],
            ...                         [2, 3, 0, 0], [0, 0, 0, 0]])
            >>> act_lens = torch.tensor([10, 5, 10, 0])
            >>> label_lens = torch.tensor([3, 1, 2, 0])
            >>> blank = 0
            >>> big_blank_durations = [2, 4, 8]
            >>> reduction = "mean"
            >>> fastemit_lambda = 0.0
            >>> clamp = 0.0
            >>> sigma = 0.05
            >>> loss = _MultiblankRNNTNumba.forward(
            ...     None, acts, labels, act_lens, label_lens,
            ...     blank, big_blank_durations, reduction,
            ...     fastemit_lambda, clamp, sigma)
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
        Compute the gradients for the backward pass of the RNNT loss.

        This method computes the gradients of the loss with respect to the
        inputs of the forward pass. It uses the stored gradients from the
        forward context to compute the final gradient output.

        Args:
            ctx: The context object containing information from the forward
                pass, including stored gradients.
            grad_output: The gradient of the loss with respect to the output
                of the forward pass. This is typically provided by PyTorch
                during the backward pass.

        Returns:
            Tuple[Tensor, None, None, None, None, None, None, None]: A tuple
            containing the gradients with respect to the inputs of the forward
            pass. The inputs that are not relevant for the gradient are
            returned as None.

        Examples:
            # Example usage in a PyTorch training loop
            loss = RNNTLossNumba()
            output = model(input)
            loss_value = loss(output, labels, act_lens, label_lens)
            loss_value.backward()  # This calls the backward method to compute gradients

        Note:
            Ensure that grad_output is not None and ctx.grads is not None
            before attempting to compute the gradients. If either is None,
            the function will not perform any operations.
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
    RNN Transducer Loss (functional form).

    This function computes the RNN Transducer loss for a given batch of
    inputs and targets. The loss can be reduced using different methods
    specified by the `reduction` parameter.

    Args:
        acts (Tensor): A tensor of shape (batch x seqLength x labelLength x outputDim)
            containing output from the network.
        labels (Tensor): A 2-dimensional tensor containing all the targets of
            the batch with zero padding.
        act_lens (Tensor): A tensor of size (batch) containing the size of
            each output sequence from the network.
        label_lens (Tensor): A tensor of size (batch) containing the label
            length of each example.
        blank (int, optional): The blank label. Default is 0.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none', 'mean', or 'sum'. 'none': no reduction will be applied,
            'mean': the output losses will be divided by the target lengths and
            then the mean over the batch is taken. Default is 'mean'.
        fastemit_lambda (float, optional): A scaling factor for FastEmit
            regularization. Default is 0.0.
        clamp (float, optional): A float value to clamp the gradient to
            [-clamp, clamp]. Default is 0.0.

    Returns:
        Tensor: The computed loss value after applying the specified reduction.

    Raises:
        ValueError: If `clamp` is negative or if the input dimensions do not
            match the expected shapes.

    Examples:
        >>> acts = torch.randn(2, 5, 10, 20)  # Example logits
        >>> labels = torch.randint(0, 10, (2, 8))  # Example labels
        >>> act_lens = torch.tensor([5, 5])  # Lengths of output sequences
        >>> label_lens = torch.tensor([8, 6])  # Lengths of label sequences
        >>> loss = rnnt_loss(acts, labels, act_lens, label_lens)
        >>> print(loss)

    Note:
        This function requires the input tensors to be contiguous and of the
        correct dtype. Ensure that the input tensors are properly formatted
        before calling this function.
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
    Compute the multi-blank RNN Transducer loss.

    This loss function is designed for training models using a multi-blank
    RNN transducer approach, as described in the paper
    "Multi-blank RNN Transducer" (https://arxiv.org/pdf/2211.03541.pdf).

    Args:
        acts (torch.Tensor): A tensor of shape (batch x seqLength x labelLength x
            outputDim) containing the output from the network.
        labels (torch.Tensor): A 2D tensor containing all the targets of the
            batch, zero-padded.
        act_lens (torch.Tensor): A tensor of size (batch) containing the size
            of each output sequence from the network.
        label_lens (torch.Tensor): A tensor of size (batch) containing the
            label length of each example.
        blank (int): The standard blank label used in the loss calculation.
        big_blank_durations (list of int, optional): A list of durations for
            the multi-blank transducer, e.g., [2, 4, 8]. Default is an empty list.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default is 'mean'.
        fastemit_lambda (float, optional): A float scaling factor for FastEmit
            regularization. Default is 0.0.
        clamp (float, optional): A float value. When set to a value >= 0.0,
            it will clamp the gradient to [-clamp, clamp]. Default is 0.0.

    Returns:
        torch.Tensor: The computed loss value.

    Raises:
        ValueError: If the input tensors do not have the expected shapes or
            types.
        NotImplementedError: If the function is called with non-CUDA tensors.

    Examples:
        >>> acts = torch.rand(32, 100, 20, 50)  # Example output from network
        >>> labels = torch.randint(0, 20, (32, 15))  # Example labels
        >>> act_lens = torch.randint(1, 100, (32,))  # Lengths of acts
        >>> label_lens = torch.randint(1, 15, (32,))  # Lengths of labels
        >>> loss = multiblank_rnnt_loss(acts, labels, act_lens, label_lens,
        ...                              blank=0, big_blank_durations=[2, 4],
        ...                              reduction='mean')
        >>> print(loss)
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
    RNNTLossNumba is a PyTorch module that computes the RNN Transducer Loss using Numba.

    This module leverages the Numba JIT compiler to optimize the forward and backward
    pass for RNN Transducer Loss, making it suitable for high-performance applications
    in automatic speech recognition (ASR) tasks.

    Attributes:
        blank (int): Standard blank label. Default is 0.
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default is 'mean'.
        fastemit_lambda (float): Scaling factor for FastEmit regularization.
        clamp (float): When set to a value >= 0.0, clamps the gradient to
            [-clamp, clamp].

    Args:
        blank (int, optional): Standard blank label. Default is 0.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default is 'mean'.
        fastemit_lambda (float, optional): Scaling factor for FastEmit regularization.
            Default is 0.0.
        clamp (float, optional): Clamping value for gradients. Default is -1.

    Examples:
        >>> import torch
        >>> loss_fn = RNNTLossNumba(blank=0, reduction='mean')
        >>> acts = torch.randn(10, 20, 15, 30)  # (batch, seqLength, labelLength, outputDim)
        >>> labels = torch.randint(0, 15, (10, 5))  # (batch, labelLength)
        >>> act_lens = torch.randint(1, 21, (10,))  # (batch)
        >>> label_lens = torch.randint(1, 6, (10,))  # (batch)
        >>> loss = loss_fn(acts, labels, act_lens, label_lens)
        >>> print(loss)

    Note:
        Ensure that the input tensors are contiguous and of the correct type before
        passing them to the forward method.

    Todo:
        - Add support for multi-blank RNN Transducer Loss in a future release.
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
        Compute the forward pass of the RNNT loss.

        This method computes the RNN Transducer (RNNT) loss based on the input
        activations (log probabilities) and target labels. It supports both GPU
        and CPU operations. The function returns the computed loss value(s)
        based on the specified reduction method.

        Args:
            ctx: The context object that can be used to store
                information for backward computation.
            acts (Tensor): A tensor of shape (batch x seqLength x
                labelLength x outputDim) containing the output from the
                network.
            labels (Tensor): A 2-dimensional tensor containing all the
                targets of the batch, zero-padded.
            act_lens (Tensor): A tensor of size (batch) containing the
                sizes of each output sequence from the network.
            label_lens (Tensor): A tensor of size (batch) containing the
                label lengths of each example.
            blank (int): The index of the blank label. Default is 0.
            reduction (str): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                'mean': the output losses will be divided by the target lengths
                and then the mean over the batch is taken. Default is 'mean'.
            fastemit_lambda (float): A scaling factor for FastEmit
                regularization. Refer to the FastEmit paper for more details.
            clamp (float): A float value to clamp the gradient to
                [-clamp, clamp]. Must be >= 0.0.

        Returns:
            Tensor: The computed loss value(s), either as a single value
            or as a tensor of values based on the reduction method specified.

        Raises:
            ValueError: If `clamp` is negative or if the input tensors
            do not meet dimensionality or type requirements.

        Examples:
            >>> acts = torch.rand(32, 100, 20, 256)  # Example logits
            >>> labels = torch.randint(0, 20, (32, 10))  # Example labels
            >>> act_lens = torch.full((32,), 100, dtype=torch.int32)
            >>> label_lens = torch.randint(1, 11, (32,), dtype=torch.int32)
            >>> loss = _RNNTNumba.forward(
            ...     None, acts, labels, act_lens, label_lens,
            ...     blank=0, reduction='mean',
            ...     fastemit_lambda=0.0, clamp=0.0
            ... )
            >>> print(loss)

        Note:
            This function is optimized for use with both CUDA and CPU
            tensors, and it is important to ensure that the inputs are
            correctly formatted to avoid runtime errors.
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
    Multiblank RNNT Loss Numba.

    This class implements the Multi-blank RNN Transducer loss using Numba for
    efficient computation. It is designed for use in automatic speech recognition
    tasks where multiple blank labels are utilized to improve the model's
    performance. The loss is computed using a forward pass, and the backward
    pass computes gradients for optimization.

    Attributes:
        blank (int): Standard blank label.
        big_blank_durations (list): List of durations for multi-blank transducer,
            e.g., [2, 4, 8].
        sigma (float): Hyper-parameter for logit under-normalization method for
            training multi-blank transducers. Recommended value: 0.05.
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.
        fastemit_lambda (float): Float scaling factor for FastEmit regularization.
        clamp (float): Float value. When set to value >= 0.0, will clamp the
            gradient to [-clamp, clamp].

    Args:
        blank (int): Standard blank label.
        big_blank_durations (list): List of durations for multi-blank transducer,
            e.g., [2, 4, 8].
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.
        fastemit_lambda (float, optional): Float scaling factor for FastEmit
            regularization. Default: 0.0.
        clamp (float, optional): Float value for gradient clamping. Default: -1.
        sigma (float, optional): Hyper-parameter for logit under-normalization.
            Default: 0.0.

    Examples:
        >>> loss_fn = MultiblankRNNTLossNumba(blank=0, big_blank_durations=[2, 4, 8])
        >>> acts = torch.randn(10, 20, 30, 40)  # Example tensor
        >>> labels = torch.randint(0, 30, (10, 15))
        >>> act_lens = torch.randint(1, 20, (10,))
        >>> label_lens = torch.randint(1, 15, (10,))
        >>> loss = loss_fn(acts, labels, act_lens, label_lens)

    Note:
        Refer to the paper at https://arxiv.org/pdf/2211.03541 for detailed
        explanations regarding the multi-blank transducer and its parameters.
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
        MultiblankRNNTNumba Forward.

        Computes the forward pass of the multi-blank RNNT loss function.

        This function takes in the model's output, the target labels, and their
        respective lengths to compute the loss for the multi-blank RNNT. The
        function supports gradient calculation for backpropagation.

        Args:
            ctx: The context object that can be used to store information
                for backward computation.
            acts (torch.Tensor): A tensor of shape
                (batch x seqLength x labelLength x outputDim) containing
                the output probabilities from the network.
            labels (torch.Tensor): A 2D tensor containing the target labels
                for each element in the batch, zero-padded.
            act_lens (torch.Tensor): A 1D tensor containing the lengths of
                each output sequence from the network (batch size).
            label_lens (torch.Tensor): A 1D tensor containing the lengths
                of the label sequences for each example in the batch.
            blank (int): The blank label used in the RNNT model.
            big_blank_durations (list): A list of durations for multi-blank
                transducer, e.g. [2, 4, 8].
            reduction (str): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. 'none': no reduction will be
                applied, 'mean': the output losses will be divided by the
                target lengths and then the mean over the batch is taken.
                Default: 'mean'.
            fastemit_lambda (float): A scaling factor for FastEmit regularization.
                Refer to the FastEmit paper for details.
            clamp (float): A float value that, when set to >= 0.0, will clamp
                the gradient to the range [-clamp, clamp].
            sigma (float): A hyper-parameter for logit under-normalization method
                for training multi-blank transducers. Recommended value is 0.05.

        Returns:
            torch.Tensor: A tensor containing the computed costs for each
            example in the batch.

        Raises:
            ValueError: If `clamp` is less than 0 or if input tensor shapes
            do not match the expected dimensions.

        Examples:
            >>> acts = torch.randn(2, 5, 3, 10)  # (batch x seqLength x labelLength x outputDim)
            >>> labels = torch.randint(0, 3, (2, 4))  # (batch x maxLabelLength)
            >>> act_lens = torch.tensor([5, 5])  # lengths of output sequences
            >>> label_lens = torch.tensor([4, 3])  # lengths of label sequences
            >>> blank = 0
            >>> big_blank_durations = [2, 4, 8]
            >>> reduction = "mean"
            >>> fastemit_lambda = 0.0
            >>> clamp = 0.0
            >>> sigma = 0.05
            >>> costs = _MultiblankRNNTNumba.forward(ctx, acts, labels, act_lens,
            ... label_lens, blank, big_blank_durations, reduction, fastemit_lambda, clamp, sigma)

        Note:
            This implementation relies on CUDA for performance; make sure
            to run this on a compatible GPU.
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
    Checks if the given variable is of the specified type.

    This function raises a TypeError if the type of the variable `var`
    does not match the expected type `t`. The name of the variable is
    used in the error message for clarity.

    Args:
        var: The variable whose type is to be checked.
        t: The expected type (e.g., `torch.float32`, `torch.int32`).
        name: A string representing the name of the variable, used in
              the error message.

    Raises:
        TypeError: If `var` is not of type `t`.

    Examples:
        >>> check_type(torch.tensor([1, 2, 3], dtype=torch.int32),
        ...            torch.int32, "my_tensor")
        # No exception raised.

        >>> check_type(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        ...            torch.int32, "my_tensor")
        Traceback (most recent call last):
            ...
        TypeError: my_tensor must be <class 'torch.int32'>
    """
    if var.dtype is not t:
        raise TypeError("{} must be {}".format(name, t))


def check_contiguous(var, name):
    """
    Check if the given variable is contiguous in memory.

    This function raises a ValueError if the input tensor is not contiguous.
    Contiguous tensors are stored in a single, contiguous block of memory,
    which can be more efficient for certain operations in PyTorch.

    Args:
        var (torch.Tensor): The tensor to check for contiguity.
        name (str): The name of the tensor being checked, used in the error message.

    Raises:
        ValueError: If the input tensor is not contiguous.

    Examples:
        >>> import torch
        >>> a = torch.tensor([[1, 2], [3, 4]])
        >>> check_contiguous(a, "a")  # This will pass
        >>> b = a.transpose(0, 1)
        >>> check_contiguous(b, "b")  # This will raise a ValueError
    """
    if not var.is_contiguous():
        raise ValueError("{} must be contiguous".format(name))


def check_dim(var, dim, name):
    """
    Check if the tensor has the specified number of dimensions.

    This function verifies whether the input tensor `var` has the expected
    number of dimensions specified by `dim`. If the number of dimensions
    does not match, a ValueError is raised.

    Args:
        var (torch.Tensor): The input tensor to be checked.
        dim (int): The expected number of dimensions.
        name (str): The name of the tensor, used in the error message.

    Raises:
        ValueError: If the number of dimensions of `var` does not match `dim`.

    Examples:
        >>> check_dim(torch.randn(2, 3), 2, "input_tensor")  # No exception
        >>> check_dim(torch.randn(2, 3, 4), 2, "input_tensor")  # Raises ValueError
    """
    if len(var.shape) != dim:
        raise ValueError("{} must be {}D".format(name, dim))


def certify_inputs(log_probs, labels, lengths, label_lengths):
    # check_type(log_probs, torch.float32, "log_probs")
    """
    Validate input tensors for RNNT loss computations.

    This function checks the types, contiguity, and dimensions of the input
    tensors used in RNNT loss calculations. It ensures that the shapes and
    data types of the inputs are compatible with the expected requirements.

    Args:
        log_probs (torch.Tensor): Tensor of shape (batch, seqLength,
            labelLength, outputDim) containing the log probabilities output
            from the network.
        labels (torch.Tensor): 2D tensor of shape (batch, labelLength)
            containing the target labels for each example in the batch,
            zero-padded.
        lengths (torch.Tensor): 1D tensor of shape (batch) containing the
            actual lengths of each output sequence from the network.
        label_lengths (torch.Tensor): 1D tensor of shape (batch) containing
            the length of each target label for the corresponding example.

    Raises:
        TypeError: If any of the input tensors have incorrect data types.
        ValueError: If any of the input tensors have incorrect shapes or
            are not contiguous in memory.

    Note:
        The `log_probs` tensor is expected to be of type float32, while
        `labels`, `lengths`, and `label_lengths` should be of type int32.

    Examples:
        >>> log_probs = torch.randn(2, 10, 5, 20).float()  # Example log probs
        >>> labels = torch.tensor([[1, 2, 3], [0, 1, 2]], dtype=torch.int32)
        >>> lengths = torch.tensor([10, 8], dtype=torch.int32)
        >>> label_lengths = torch.tensor([3, 2], dtype=torch.int32)
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
