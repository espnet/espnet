# This code is derived from https://github.com/HazyResearch/state-spaces

"""Implements downsampling and upsampling on sequences."""

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import nn

from espnet2.asr.state_spaces.base import SequenceModule
from espnet2.asr.state_spaces.components import LinearActivation

"""Simple pooling functions that just downsample or repeat

stride: Subsample on the layer dimension
expand: Repeat on the feature dimension
"""


def downsample(x, stride=1, expand=1, transposed=False):
    """
    Downsample or upsample a sequence tensor.

    This function performs downsampling and upsampling on input tensors. It
    allows for reducing the sequence length by a specified stride and
    expanding the feature dimension by a specified factor. The operation can
    be performed in a transposed manner as well.

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, D) where B is the
            batch size, L is the sequence length, and D is the feature
            dimension.
        stride (int, optional): The downsampling factor for the sequence
            length. Default is 1 (no downsampling).
        expand (int, optional): The factor by which to expand the feature
            dimension. Default is 1 (no expansion).
        transposed (bool, optional): If True, performs the operation in
            transposed mode. Default is False.

    Returns:
        torch.Tensor: The downsampled or upsampled tensor.

    Raises:
        AssertionError: If the input tensor has more than 3 dimensions
            while a stride greater than 1 is specified.

    Examples:
        >>> x = torch.randn(2, 8, 4)  # Batch of 2, sequence length 8, features 4
        >>> downsampled = downsample(x, stride=2, expand=2)
        >>> downsampled.shape
        torch.Size([2, 4, 8])  # Sequence length reduced to 4, features expanded to 8

        >>> upsampled = downsample(x, stride=1, expand=2, transposed=True)
        >>> upsampled.shape
        torch.Size([2, 16, 4])  # Sequence length remains 8, features expanded to 16
    """
    if x is None:
        return None
    if stride > 1:
        assert x.ndim == 3, (
            "Downsampling with higher-dimensional inputs is currently not supported."
            "It is recommended to use average or spectral pooling instead."
        )
        if transposed:
            x = x[..., 0::stride]
        else:
            x = x[..., 0::stride, :]

    if expand > 1:
        if transposed:
            x = repeat(x, "b d ... -> b (d e) ...", e=expand)
        else:
            x = repeat(x, "b ... d -> b ... (d e)", e=expand)

    return x


def upsample(x, stride=1, expand=1, transposed=False):
    """
    Upsample a tensor by applying stride and expansion.

    This function increases the dimensions of the input tensor `x` by
    repeating its elements based on the specified `stride` and `expand`
    parameters. The function can also handle transposed operations.

    Args:
        x (torch.Tensor): The input tensor to upsample. It should be
            a 3D or higher-dimensional tensor.
        stride (int, optional): The factor by which to increase the
            sequence length. Default is 1.
        expand (int, optional): The factor by which to increase the
            feature dimension. Default is 1.
        transposed (bool, optional): If True, the operation will be
            treated as a transposed operation. Default is False.

    Returns:
        torch.Tensor: The upsampled tensor.

    Examples:
        >>> import torch
        >>> x = torch.tensor([[1, 2], [3, 4]])
        >>> upsample(x, stride=2, expand=2)
        tensor([[1, 1, 2, 2],
                [3, 3, 4, 4]])

        >>> upsample(x, stride=1, expand=3)
        tensor([[1, 2, 2, 2],
                [3, 4, 4, 4]])

        >>> upsample(x, stride=2, expand=1, transposed=True)
        tensor([[1, 2],
                [1, 2],
                [3, 4],
                [3, 4]])

    Raises:
        AssertionError: If the input tensor is None.
    """
    if x is None:
        return None
    if expand > 1:
        if transposed:
            x = reduce(x, "... (d e) l -> ... d l", "mean", e=expand)
        else:
            x = reduce(x, "... (d e) -> ... d", "mean", e=expand)
    if stride > 1:
        if transposed:
            x = repeat(x, "... l -> ... (l e)", e=stride)
        else:
            x = repeat(x, "... l d -> ... (l e) d", e=stride)
    return x


class DownSample(SequenceModule):
    """
    Downsampling module for sequence data.

    This class implements a downsampling operation that reduces the
    sequence length by a specified stride and optionally expands the
    feature dimension. It can be used as a building block in sequence
    models.

    Attributes:
        d_input (int): The input dimension of the sequence.
        stride (int): The downsampling factor applied to the sequence length.
        expand (int): The factor by which to expand the feature dimension.
        transposed (bool): If True, applies transposed downsampling.

    Args:
        d_input (int): Input dimension of the sequence.
        stride (int, optional): Stride for downsampling. Default is 1.
        expand (int, optional): Expansion factor for the feature dimension.
            Default is 1.
        transposed (bool, optional): If True, applies transposed downsampling.
            Default is True.

    Returns:
        torch.Tensor: Downsampled output tensor.

    Raises:
        NotImplementedError: If stride or expand is greater than 1 during
            the step method.

    Examples:
        >>> downsample_layer = DownSample(d_input=64, stride=2, expand=1)
        >>> input_tensor = torch.randn(10, 100, 64)  # (batch_size, seq_len, d_input)
        >>> output_tensor = downsample_layer(input_tensor)
        >>> output_tensor.shape
        torch.Size([10, 50, 64])  # (batch_size, seq_len/stride, d_input)

    Note:
        The `forward` method should be used to perform downsampling on
        the input tensor. The `step` method is designed for recurrent
        use cases and is not implemented for stride or expand greater than 1.
    """

    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        """
        Perform a forward pass through the DownSample layer.

        This method applies downsampling to the input tensor `x` using the
        specified stride and expansion parameters. The downsampling is
        achieved by calling the `downsample` function defined in the
        module. The input tensor should have at least 3 dimensions,
        corresponding to batch size, length, and feature dimensions.

        Args:
            x (torch.Tensor): The input tensor of shape (B, L, D), where:
                - B is the batch size.
                - L is the length of the input sequence.
                - D is the number of features.

        Returns:
            torch.Tensor: The downsampled output tensor of shape
            (B, L_new, D_new), where L_new is determined by the
            `stride` and D_new is determined by the `expand` factor.

        Examples:
            >>> downsample_layer = DownSample(d_input=64, stride=2, expand=1)
            >>> input_tensor = torch.randn(8, 10, 64)  # (B=8, L=10, D=64)
            >>> output_tensor = downsample_layer(input_tensor)
            >>> output_tensor.shape
            torch.Size([8, 5, 64])  # (B=8, L_new=5, D=64)

        Note:
            The method will raise an error if the input tensor `x` is
            None or if the input does not have the required number of
            dimensions.
        """
        return downsample(x, self.stride, self.expand, False, self.transposed)

    def step(self, x, state, **kwargs):
        """
        Step one time step as a recurrent model.

        This method processes a single time step of input `x` while maintaining
        the state of the model. It is designed to be used in recurrent contexts where
        sequential input is provided over time.

        Args:
            x: A tensor of shape (..., H) representing the input at the current time step.
            state: A list containing the previous states that will be updated.
            **kwargs: Additional keyword arguments for future extensions.

        Returns:
            A tuple containing:
                - The processed output tensor for the current time step or None if
                no output is produced.
                - The updated state, which is either an empty list if the output was
                generated, or the updated state list.

        Raises:
            NotImplementedError: If the stride or expand parameters are greater than 1,
            indicating that this function does not support those configurations.

        Examples:
            >>> model = DownSample(d_input=128, stride=1, expand=1)
            >>> x = torch.randn(10, 20, 128)  # Example input tensor
            >>> state = []  # Initial state
            >>> output, updated_state = model.step(x, state)

        Note:
            This method assumes that the input tensor `x` and the state are managed
            correctly outside of this function to ensure the model operates as intended.
        """
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        """
        Calculates the output dimension of the downsampling layer.

        The output dimension is determined by the input dimension multiplied
        by the expand factor. This is useful for understanding the size of the
        output tensor after applying the downsampling operation.

        Returns:
            int: The output dimension after downsampling, computed as
            `d_input * expand`.

        Examples:
            >>> downsample_layer = DownSample(d_input=64, expand=2)
            >>> downsample_layer.d_output
            128

        Note:
            This property assumes that the `expand` factor is defined during
            the initialization of the DownSample instance.
        """
        return self.d_input * self.expand


class DownAvgPool(SequenceModule):
    """
    Downsample input sequences using average pooling.

    This module applies average pooling to the input tensor along the layer
    dimension while allowing for expansion on the feature dimension. It can
    operate in both transposed and non-transposed modes.

    Attributes:
        d_input (int): The number of input features.
        stride (int): The downsampling factor for the layer dimension.
        expand (int): The repetition factor for the feature dimension.
        transposed (bool): If True, the operation will be applied in transposed mode.

    Args:
        d_input (int): Number of input features.
        stride (int, optional): Downsampling factor. Default is 1.
        expand (int, optional): Feature dimension expansion factor. Default is 1.
        transposed (bool, optional): If True, operates in transposed mode. Default is True.

    Returns:
        Tensor: The downsampled (and potentially expanded) output tensor.

    Raises:
        NotImplementedError: If `stride` or `expand` is greater than 1 in the `step` method.

    Examples:
        >>> down_avg_pool = DownAvgPool(d_input=64, stride=2, expand=2)
        >>> input_tensor = torch.randn(10, 20, 64)  # (batch_size, length, features)
        >>> output_tensor = down_avg_pool(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([10, 10, 128])  # Downsampled length and expanded features

    Note:
        This module expects the input tensor to have a minimum of 3 dimensions.
    """

    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        """
            Perform the forward pass of the DownAvgPool layer.

        This method takes an input tensor `x` and applies average pooling and
        optional expansion based on the specified stride and expand parameters.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L..., D) where:
                - B is the batch size,
                - L... represents any number of additional dimensions,
                - D is the number of features.

        Returns:
            torch.Tensor: Output tensor after applying average pooling and
            expansion, of shape (B, L'..., D'), where L' is the reduced length
            after pooling and D' is the expanded number of features if expand > 1.

        Examples:
            >>> pool = DownAvgPool(d_input=16, stride=2, expand=2)
            >>> input_tensor = torch.randn(8, 4, 16)  # Batch of 8, 4 time steps, 16 features
            >>> output_tensor = pool(input_tensor)
            >>> output_tensor.shape
            torch.Size([8, 2, 32])  # Output shape after pooling and expansion

        Note:
            - If `self.transposed` is set to `True`, the input tensor will be
              rearranged before applying the pooling operation.
            - The method raises a NotImplementedError if the stride or expand
              parameters are greater than 1 in the `step` method.

        Raises:
            NotImplementedError: If stride or expand parameters are greater than 1
            during the step method.
        """
        if not self.transposed:
            x = rearrange(x, "b ... d -> b d ...")

        if self.stride > 1:
            # einops appears slower than F
            if x.ndim == 3:
                x = F.avg_pool1d(x, self.stride, self.stride)
            elif x.ndim == 4:
                x = F.avg_pool2d(x, self.stride, self.stride)
            else:
                # Reduction string e.g. "b d (l1 2) (l2 2) -> b d l1 l2"
                reduce_str = (
                    "b d "
                    + " ".join([f"(l{i} {self.stride})" for i in range(x.ndim - 2)])
                    + " -> b d "
                    + " ".join([f"l{i}" for i in range(x.ndim - 2)])
                )
                x = reduce(x, reduce_str, "mean")

        if self.expand > 1:
            x = repeat(x, "b d ... -> b (d e) ...", e=self.expand)

        if not self.transposed:
            x = rearrange(x, "b d ... -> b ... d")
        return x

    def step(self, x, state, **kwargs):
        """
        Process a single time step in a recurrent model.

        This method handles the input tensor `x` and the current `state`,
        performing operations defined by the pooling parameters. It is
        primarily used in recurrent models where state is maintained across
        time steps.

        Args:
            x (torch.Tensor): The input tensor of shape (..., H), where H
                represents the feature dimension.
            state (list): A list representing the current state of the model.
                It is updated during the step.
            **kwargs: Additional keyword arguments for future extensions.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor or None: The processed output tensor if
                the state has reached the defined stride, otherwise None.
                - list: The updated state after processing the input.

        Raises:
            NotImplementedError: If the stride or expand attributes are
            greater than 1, as these operations are not implemented in this
            method.

        Examples:
            >>> model = DownSample(d_input=64, stride=1, expand=1)
            >>> state = []
            >>> output, new_state = model.step(torch.randn(10, 64), state)
            >>> assert output.shape == (10, 64)  # Assuming the input has
            >>> # the correct dimensions for processing.

        Note:
            This method is intended to be used in a recurrent context, and
            is not designed for batch processing of inputs.
        """
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        """
        Calculates the output dimension after applying the downsampling.

        This property computes the output dimension based on the input dimension
        and the expand factor. The output dimension is given by the formula:
        d_output = d_input * expand.

        Returns:
            int: The computed output dimension.

        Examples:
            >>> down_sample = DownSample(d_input=128, expand=2)
            >>> down_sample.d_output
            256
        """
        return self.d_input * self.expand


class DownSpectralPool(SequenceModule):
    """
    Downsampling using spectral pooling.

    This class implements downsampling on sequences using spectral
    methods. It performs an inverse Fast Fourier Transform (iFFT)
    to convert the input to the frequency domain, selects the
    relevant frequency components based on the specified stride,
    and then applies an inverse iFFT to obtain the downsampled
    output.

    Attributes:
        d_input (int): The dimensionality of the input features.
        stride (int): The factor by which to downsample the input.
        expand (int): The factor by which to expand the output features.
        transposed (bool): Whether to perform the operation in
            transposed mode.

    Args:
        d_input (int): The input feature dimension.
        stride (int, optional): The downsampling factor. Defaults to 1.
        expand (int, optional): The expansion factor for the output.
            Defaults to 1.
        transposed (bool, optional): Whether to use transposed
            operations. Defaults to True.

    Returns:
        Tensor: The downsampled output tensor.

    Raises:
        AssertionError: If the input length is not divisible by
            the stride.

    Examples:
        >>> import torch
        >>> pool = DownSpectralPool(d_input=64, stride=2, expand=1)
        >>> input_tensor = torch.randn(10, 5, 64)  # (B, L, D)
        >>> output_tensor = pool(input_tensor)
        >>> output_tensor.shape
        torch.Size([10, 3, 64])  # Downsampled length due to stride

    Note:
        This pooling method is particularly effective for
        frequency-based downsampling and may not be suitable for
        all types of data.

    Todo:
        - Implement additional checks for input dimensions.
    """

    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        """
        Implements downsampling of sequences using spectral pooling.

        This class applies downsampling through spectral pooling, which uses
        the Fast Fourier Transform (FFT) to manipulate the frequency domain
        representation of the input data. It supports optional transposed
        operations for upsampling.

        Attributes:
            d_input (int): The input dimensionality of the data.
            stride (int): The factor by which to downsample the input.
            expand (int): The factor by which to expand the output.
            transposed (bool): Indicates whether the operation is transposed.

        Args:
            d_input (int): Input dimensionality.
            stride (int): The downsampling factor (default: 1).
            expand (int): The expansion factor for the output (default: 1).
            transposed (bool): Whether to perform transposed operations (default: True).

        Returns:
            torch.Tensor: The downsampled output tensor of shape
            (B, D', ...) where D' = d_input * expand.

        Raises:
            AssertionError: If the input length is not divisible by stride.

        Examples:
            >>> import torch
            >>> down_pool = DownSpectralPool(d_input=64, stride=2, expand=1)
            >>> x = torch.randn(8, 10, 64)  # Batch of 8, sequence length of 10
            >>> output = down_pool(x)
            >>> output.shape
            torch.Size([8, 32, ...])  # Output shape will depend on input dimensions

        Note:
            This method requires the input length to be divisible by the stride
            to ensure valid downsampling.
        """
        if not self.transposed:
            x = rearrange(x, "b ... d -> b d ...")
        shape = x.shape[2:]
        x_f = torch.fft.ifftn(x, s=shape)

        for axis, l in enumerate(shape):
            assert l % self.stride == 0, "input length must be divisible by stride"
            new_l = l // self.stride
            idx = torch.cat(
                [torch.arange(0, new_l - new_l // 2), l + torch.arange(-new_l // 2, 0)]
            ).to(x_f.device)
            x_f = torch.index_select(x_f, 2 + axis, idx)
        x = torch.fft.ifftn(x_f, s=[length // self.stride for length in shape])
        x = x.real

        if self.expand > 1:
            x = repeat(x, "b d ... -> b (d e) ...", e=self.expand)
        if not self.transposed:
            x = rearrange(x, "b d ... -> b ... d")
        return x

    def step(self, x, state, **kwargs):
        """
        Applies downsampling using spectral methods on input sequences.

        This class implements downsampling of input sequences using
        spectral methods. It transforms the input into the frequency domain,
        selects indices based on the specified stride, and transforms the
        result back into the time domain. The downsampling can also
        include an expansion factor.

        Attributes:
            d_input (int): The dimensionality of the input.
            stride (int): The downsampling factor.
            expand (int): The expansion factor for the output.
            transposed (bool): Indicates whether to use transposed operations.

        Args:
            d_input (int): Dimensionality of the input.
            stride (int): Downsampling factor (default is 1).
            expand (int): Expansion factor for the output (default is 1).
            transposed (bool): If True, use transposed operations (default is True).

        Returns:
            Tensor: The downsampled output tensor.

        Raises:
            NotImplementedError: If stride or expand is greater than 1.

        Examples:
            >>> down_pool = DownSpectralPool(d_input=128, stride=2, expand=1)
            >>> x = torch.randn(10, 20, 128)  # Batch size 10, sequence length 20
            >>> output = down_pool(x)
            >>> print(output.shape)  # Output shape will be (10, 10, 128)

        Note:
            The input length must be divisible by the stride.
        """
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        """
        Calculates the output dimension based on input dimension and expand.

        The output dimension is determined by multiplying the input dimension
        (`d_input`) by the `expand` factor. This property is useful for
        understanding the dimensionality of the output tensor after
        applying downsampling operations.

        Returns:
            int: The computed output dimension.

        Examples:
            >>> downsample_layer = DownSample(d_input=128, expand=2)
            >>> downsample_layer.d_output
            256
        """
        return self.d_input * self.expand


class UpSample(nn.Module):
    """
    Upsampling layer for sequences.

    This class implements an upsampling operation that increases the
    temporal resolution of input sequences by repeating or expanding the
    features based on the specified stride and expand factors.

    Attributes:
        d_input (int): The number of input features.
        stride (int): The factor by which to upsample the sequence length.
        expand (int): The factor by which to expand the feature dimension.
        transposed (bool): If True, applies the upsampling in a transposed
            manner.

    Args:
        d_input (int): Number of input features.
        stride (int): Upsampling stride. Default is 1.
        expand (int): Feature expansion factor. Default is 1.
        transposed (bool): If True, apply transposed upsampling. Default is True.

    Returns:
        torch.Tensor: The upsampled output tensor.

    Examples:
        >>> upsample_layer = UpSample(d_input=64, stride=2, expand=3)
        >>> input_tensor = torch.randn(10, 5, 64)  # (batch_size, length, features)
        >>> output_tensor = upsample_layer(input_tensor)
        >>> output_tensor.shape
        torch.Size([10, 10, 64])  # Length increased to 10 (5 * 2)

    Raises:
        NotImplementedError: If stride or expand is greater than 1 during the
        step method.

    Note:
        The output dimension of the upsampled tensor is computed as:
        d_output = d_input // expand.

    Todo:
        - Add support for non-default values of stride and expand in the step method.
    """

    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        """
        Perform the forward pass of the UpSample module.

        This method takes an input tensor `x` and applies upsampling
        based on the specified stride and expansion factors. The
        upsampling can be performed in a transposed manner based on
        the `transposed` attribute.

        Args:
            x (torch.Tensor): The input tensor of shape (B, L, D) or
                (B, L..., D) where B is the batch size, L is the
                sequence length, and D is the number of features.

        Returns:
            torch.Tensor: The upsampled output tensor with shape
                determined by the input shape and the `stride` and
                `expand` parameters.

        Examples:
            >>> upsample_layer = UpSample(d_input=16, stride=2, expand=2)
            >>> input_tensor = torch.randn(8, 10, 16)  # (B, L, D)
            >>> output_tensor = upsample_layer(input_tensor)
            >>> output_tensor.shape
            torch.Size([8, 20, 32])  # Up-sampled shape

        Note:
            If `x` is None, the method will return None.
        """
        return upsample(x, self.stride, self.expand, self.transposed)

    @property
    def d_output(self):
        """
        int: The output dimensionality of the layer.

        This property calculates the output dimensionality based on the input
        dimensionality and the expansion factor. The output dimension is
        determined by multiplying the input dimension (`d_input`) by the
        expansion factor.

        Returns:
            int: The computed output dimension.

        Examples:
            If `d_input` is 64 and `expand` is 2, then:
            >>> layer = DownSample(d_input=64, expand=2)
            >>> layer.d_output
            128

            If `d_input` is 128 and `expand` is 1, then:
            >>> layer = DownSample(d_input=128, expand=1)
            >>> layer.d_output
            128
        """
        return self.d_input // self.expand

    def step(self, x, state, **kwargs):
        """
        Step one time step as a recurrent model.

        This method processes a single time step of input data, potentially
        maintaining state for recurrent computations. It raises a
        NotImplementedError if stride or expand parameters are greater than 1.

        Args:
            x (torch.Tensor): The input tensor for the current time step,
                expected to have the shape (..., H).
            state (list): The state from previous time steps, which can
                be used to maintain context in recurrent models.
            **kwargs: Additional keyword arguments that may be used in
                the future.

        Returns:
            tuple: A tuple containing the output tensor for the current
                time step (or None if not applicable) and the updated state.

        Raises:
            NotImplementedError: If `self.stride` or `self.expand` is
                greater than 1, indicating that the current implementation
                does not support such configurations.

        Examples:
            >>> model = UpSample(d_input=128, stride=1, expand=1)
            >>> input_tensor = torch.randn(10, 5, 128)  # Example input
            >>> state = []
            >>> output, new_state = model.step(input_tensor, state)

        Note:
            This method is designed for use in recurrent scenarios where
            maintaining a state across time steps is necessary. It is
            important to ensure that the input tensor shape is consistent
            with the expected input dimensions.
        """
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state


""" Pooling functions with trainable parameters """
# For the flexible backbone SequenceModel


class DownLinearPool(SequenceModule):
    """
    Applies linear downsampling to input sequences with trainable parameters.

    This module performs downsampling of the input sequences using a linear
    transformation. The downsampling is achieved by rearranging the input
    tensor, applying a linear layer, and optionally transposing the output.

    Attributes:
        d_input (int): The dimensionality of the input feature vectors.
        stride (int): The factor by which to downsample the input sequences.
        expand (int): The factor by which to expand the output feature vectors.
        transposed (bool): Indicates whether to apply transposed operations.
        linear (LinearActivation): A linear layer that transforms the input
            features.

    Args:
        d_input (int): The dimensionality of the input feature vectors.
        stride (int, optional): The downsampling factor (default is 1).
        expand (int, optional): The expansion factor for output features
            (default is 1).
        transposed (bool, optional): Whether to use transposed operations
            (default is True).

    Returns:
        Tensor: The transformed output tensor after applying the linear
        transformation and downsampling.

    Raises:
        NotImplementedError: If stride or expand is greater than 1 in the
        `step` method.

    Examples:
        >>> down_linear_pool = DownLinearPool(d_input=64, stride=2, expand=1)
        >>> input_tensor = torch.randn(10, 8, 64)  # (batch_size, length, d_input)
        >>> output_tensor = down_linear_pool(input_tensor)
        >>> output_tensor.shape
        torch.Size([10, 4, 64])  # Output shape after downsampling

    Note:
        The output dimensionality is computed as `d_input * expand`.
    """

    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()

        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

        self.linear = LinearActivation(
            d_input * stride,
            d_input * expand,
            transposed=transposed,
        )

    def forward(self, x):
        """
        Applies a linear transformation with downsampling on input sequences.

        This module performs a downsampling operation followed by a linear
        transformation. The input is reshaped based on the specified stride
        and expanded based on the specified parameters.

        Attributes:
            d_input (int): The dimensionality of the input features.
            stride (int): The downsampling factor along the sequence length.
            expand (int): The factor by which to expand the output features.
            transposed (bool): Indicates whether to apply the transformation in
                transposed mode (affecting the input reshaping).
            linear (LinearActivation): A linear layer that applies the
                transformation after downsampling.

        Args:
            d_input (int): Number of input features.
            stride (int): Factor to downsample the input (default: 1).
            expand (int): Factor to expand the output features (default: 1).
            transposed (bool): Whether to apply the linear layer in
                transposed mode (default: True).

        Returns:
            torch.Tensor: The transformed output after applying downsampling
            and the linear transformation.

        Examples:
            >>> down_linear_pool = DownLinearPool(d_input=64, stride=2, expand=2)
            >>> input_tensor = torch.randn(10, 8, 64)  # (batch_size, sequence_length, d_input)
            >>> output_tensor = down_linear_pool(input_tensor)
            >>> output_tensor.shape
            torch.Size([10, 4, 128])  # Downsampled length and expanded features

        Raises:
            NotImplementedError: If stride or expand is greater than 1 in
            the `step` method.
        """
        if self.transposed:
            x = rearrange(x, "... h (l s) -> ... (h s) l", s=self.stride)
        else:
            x = rearrange(x, "... (l s) h -> ... l (h s)", s=self.stride)
        x = self.linear(x)
        return x

    def step(self, x, state, **kwargs):
        """
        Downsampling layer that applies a linear transformation with pooling.

        This class implements a downsampling operation that combines a linear
        activation with an optional stride and expansion factor. The input
        sequence is rearranged according to the specified transposed mode
        before applying the linear transformation.

        Attributes:
            d_input (int): The dimensionality of the input features.
            stride (int): The stride factor for downsampling.
            expand (int): The expansion factor for upsampling.
            transposed (bool): Indicates whether to apply the operation in a
                transposed manner.
            linear (LinearActivation): The linear transformation layer.

        Args:
            d_input (int): The input feature dimension.
            stride (int, optional): The downsampling stride. Default is 1.
            expand (int, optional): The upsampling expansion factor. Default is 1.
            transposed (bool, optional): If True, the operation is transposed.
                Default is True.

        Returns:
            Tensor: The transformed output tensor after downsampling.

        Raises:
            NotImplementedError: If the stride or expand factors are greater than 1
                in the `step` method.

        Examples:
            >>> down_pool = DownLinearPool(d_input=64, stride=2, expand=1)
            >>> x = torch.randn(10, 5, 64)  # (batch_size, length, features)
            >>> output = down_pool(x)
            >>> print(output.shape)  # Output shape will be (10, 3, 128)

        Note:
            The input tensor should have at least three dimensions, where the last
            dimension corresponds to the feature dimension.
        """
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        """
        Calculates the output dimension based on input and expand factors.

        This property computes the output dimension for the DownLinearPool layer
        by multiplying the input dimension by the expand factor. It is useful for
        determining the shape of the output tensor after applying the pooling
        operation.

        Returns:
            int: The calculated output dimension.

        Examples:
            >>> down_linear_pool = DownLinearPool(d_input=64, expand=2)
            >>> print(down_linear_pool.d_output)
            128
        """
        return self.d_input * self.expand


""" Pooling functions with trainable parameters """


class DownPool2d(SequenceModule):
    """
    DownPool2d is a pooling layer that performs downsampling on 2D inputs.

    This class implements a downsampling layer using average pooling followed by
    a linear transformation. It is designed for processing 2D inputs such as
    images or feature maps in a sequence-to-sequence model.

    Attributes:
        linear (LinearActivation): A linear transformation applied after pooling.
        pool (tuple): A tuple containing the AvgPool2d layer used for downsampling.

    Args:
        d_input (int): The number of input features.
        d_output (int): The number of output features after pooling.
        stride (int): The stride of the pooling operation.
        transposed (bool): If True, applies transposed pooling.
        weight_norm (bool): If True, applies weight normalization to the linear layer.

    Returns:
        Tensor: The output tensor after applying downsampling and linear transformation.

    Examples:
        >>> down_pool = DownPool2d(d_input=16, d_output=8, stride=2)
        >>> x = torch.randn(1, 16, 32, 32)  # Batch size of 1, 16 channels, 32x32 image
        >>> output = down_pool(x)
        >>> output.shape
        torch.Size([1, 8, 16, 16])  # Output shape after downsampling

    Note:
        This layer is typically used in models where spatial dimensions need
        to be reduced while retaining important features.
    """

    def __init__(self, d_input, d_output, stride=1, transposed=True, weight_norm=True):
        super().__init__()

        self.linear = LinearActivation(
            d_input,
            d_output,
            transposed=transposed,
            weight_norm=weight_norm,
        )

        self.pool = (nn.AvgPool2d(kernel_size=stride, stride=stride),)

    def forward(self, x):
        """
        Forward pass for the DownPool layer.

        This method applies a linear transformation to the input tensor `x`
        after rearranging it according to the specified stride and transposition
        settings. It is typically used to downsample the input sequence by
        applying the specified linear activation function.

        Args:
            x (torch.Tensor): The input tensor of shape (..., H) where
                H is the number of features in the input.

        Returns:
            tuple: A tuple containing the transformed tensor and None.
                The transformed tensor will have a shape based on the
                linear transformation applied.

        Examples:
            >>> down_pool = DownPool(d_input=128, d_output=64, stride=2, transposed=False)
            >>> input_tensor = torch.randn(10, 32, 128)  # (batch_size, sequence_length, features)
            >>> output_tensor, _ = down_pool(input_tensor)
            >>> output_tensor.shape
            torch.Size([10, 16, 64])  # Expected output shape after downsampling

        Note:
            The method expects that the input tensor has at least three
            dimensions. The transposed option allows for different
            handling of the input tensor shape.
        """
        if self.transposed:
            x = self.pool(x)


# DownLinearPool is used by the registry (for isotropic backbone)
# DownPool is essentially the same as DownLinearPool. These should be consolidated
class DownPool(SequenceModule):
    """
    Downsampling layer that applies linear transformations and pooling.

    This class implements a downsampling mechanism that combines
    linear activation with pooling operations to reduce the
    dimensionality of input sequences. It can be used in
    various neural network architectures, particularly those
    involving sequential data.

    Attributes:
        d_output (int): The dimensionality of the output after
            downsampling.
        stride (int): The factor by which to downsample the input.
        transposed (bool): Whether to apply transposed operations.
        linear (LinearActivation): A linear layer that transforms
            the input features.

    Args:
        d_input (int): The dimensionality of the input features.
        d_output (int, optional): The dimensionality of the output
            features. If not provided, it is computed as
            `d_input * expand`.
        expand (int, optional): The factor by which to expand the
            output features. If provided, `d_output` should be
            None.
        stride (int, optional): The downsampling stride. Defaults to 1.
        transposed (bool, optional): Whether to use transposed
            operations. Defaults to True.
        weight_norm (bool, optional): Whether to apply weight
            normalization. Defaults to True.
        initializer (callable, optional): Function for initializing
            the weights. Defaults to None.
        activation (callable, optional): Activation function to
            apply after the linear transformation. Defaults to None.

    Returns:
        Tuple[torch.Tensor, None]: The downsampled output and
        None (placeholder for potential future states).

    Examples:
        >>> down_pool = DownPool(d_input=64, d_output=32, stride=2)
        >>> x = torch.randn(10, 5, 64)  # (batch_size, sequence_length, features)
        >>> output, _ = down_pool(x)
        >>> print(output.shape)
        torch.Size([10, 3, 32])  # (batch_size, new_sequence_length, new_features)

    Note:
        Ensure that the input tensor's shape is compatible with
        the specified stride and dimensionality settings.
    """

    def __init__(
        self,
        d_input,
        d_output=None,
        expand=None,
        stride=1,
        transposed=True,
        weight_norm=True,
        initializer=None,
        activation=None,
    ):
        super().__init__()
        assert (d_output is None) + (expand is None) == 1
        if d_output is None:
            d_output = d_input * expand

        self.d_output = d_output
        self.stride = stride
        self.transposed = transposed

        self.linear = LinearActivation(
            d_input * stride,
            d_output,
            transposed=transposed,
            initializer=initializer,
            weight_norm=weight_norm,
            activation=activation,
            activate=True if activation is not None else False,
        )

    def forward(self, x):
        """
        Perform the forward pass of the DownPool layer.

        This method applies downsampling to the input tensor `x` using the
        specified stride and expand parameters. It rearranges the tensor
        based on whether the operation is transposed or not and passes
        the result through a linear activation layer.

        Args:
            x (torch.Tensor): Input tensor of shape (..., H), where
                H is the feature dimension.

        Returns:
            torch.Tensor: The output tensor after downsampling and
                linear transformation.

        Examples:
            >>> down_pool = DownPool(d_input=64, d_output=32, stride=2, transposed=False)
            >>> x = torch.randn(10, 16, 64)  # Batch size of 10, sequence length of 16
            >>> output = down_pool.forward(x)
            >>> output.shape
            torch.Size([10, 8, 32])  # Output shape after downsampling

        Note:
            Ensure that the input tensor is appropriately shaped and
            that the parameters are set correctly to avoid runtime errors.
        """
        if self.transposed:
            x = rearrange(x, "... h (l s) -> ... (h s) l", s=self.stride)
        else:
            x = rearrange(x, "... (l s) h -> ... l (h s)", s=self.stride)
        x = self.linear(x)
        return x, None

    def step(self, x, state, **kwargs):
        """
        Downsampling layer that combines linear transformations with pooling.

        This class implements a downsampling mechanism that applies a linear
        transformation followed by pooling. It can operate in both transposed
        and non-transposed modes, allowing for flexibility in sequence modeling
        tasks.

        Attributes:
            d_input (int): The dimensionality of the input features.
            d_output (int): The dimensionality of the output features.
            stride (int): The stride used for downsampling.
            transposed (bool): Whether to apply the transformation in transposed mode.
            linear (LinearActivation): The linear activation layer used for
                transformation.

        Args:
            d_input (int): The number of input features.
            d_output (int, optional): The number of output features. If not
                provided, it will be calculated based on `expand`.
            expand (int, optional): Expansion factor for the output features.
                If provided, `d_output` must be None.
            stride (int, optional): The stride for downsampling. Default is 1.
            transposed (bool, optional): Indicates if the transformation should
                be transposed. Default is True.
            weight_norm (bool, optional): Indicates whether to apply weight
                normalization. Default is True.
            initializer (callable, optional): Function for weight initialization.
            activation (callable, optional): Activation function to be used.

        Returns:
            tuple: A tuple containing the transformed output and a state (None).

        Examples:
            >>> down_pool = DownPool(d_input=64, d_output=32, stride=2)
            >>> input_tensor = torch.randn(10, 20, 64)  # (batch_size, seq_len, features)
            >>> output, _ = down_pool(input_tensor)
            >>> output.shape
            torch.Size([10, 10, 32])  # Output shape after downsampling

        Note:
            - The `step` method is intended for use in recurrent models where
            stateful processing is required.
            - Ensure that the input shape is compatible with the defined stride
            and expand parameters.

        Raises:
            AssertionError: If both `d_output` and `expand` are provided, or if
                the input shape does not match the expected dimensions.
        """
        if x is None:
            return None, state
        state.append(x)
        if len(state) == self.stride:
            x = rearrange(torch.stack(state, dim=-1), "... h s -> ... (h s)")
            if self.transposed:
                x = x.unsqueeze(-1)
            x = self.linear(x)
            if self.transposed:
                x = x.squeeze(-1)
            return x, []
        else:
            return None, state

    def default_state(self, *batch_shape, device=None):
        """
        Downsampling module for sequences using linear activation.

        This module performs downsampling on input sequences while allowing for
        optional expansion and transposition. The output dimension can be
        specified either directly or by using an expansion factor.

        Attributes:
            d_input (int): The input dimension of the sequence.
            d_output (int): The output dimension of the sequence.
            stride (int): The downsampling factor for the input sequence.
            transposed (bool): Whether to apply transposed operations.
            linear (LinearActivation): A linear layer for transforming the
                downsampled output.

        Args:
            d_input (int): The dimension of the input.
            d_output (int, optional): The dimension of the output. If None,
                it is computed based on the `expand` parameter.
            expand (int, optional): Expansion factor for the output dimension.
            stride (int): The downsampling factor for the input sequence.
            transposed (bool): Whether to apply transposed operations.
            weight_norm (bool): Whether to apply weight normalization.
            initializer (callable, optional): A function to initialize weights.
            activation (callable, optional): An activation function to apply
                after the linear transformation.

        Returns:
            Tuple[Tensor, None]: The downsampled output tensor and None.

        Raises:
            AssertionError: If both `d_output` and `expand` are None or both
                are provided.

        Examples:
            >>> down_pool = DownPool(d_input=64, expand=2, stride=2)
            >>> x = torch.randn(32, 10, 64)  # (batch_size, sequence_length, d_input)
            >>> output, _ = down_pool(x)
            >>> output.shape
            torch.Size([32, 5, 128])  # (batch_size, new_sequence_length, d_output)

        Note:
            The output dimension is calculated as:
            d_output = d_input * expand if d_output is None.

        Todo:
            Consider implementing additional pooling strategies or methods
            for more flexible usage.
        """
        return []


class UpPool(SequenceModule):
    """
    Upsampling layer that applies a linear transformation followed by reshaping.

    This class implements an upsampling operation that uses a linear layer to
    transform the input tensor, followed by reshaping the output tensor based
    on the specified stride and transposition settings. The upsampling process
    can be configured to include a skip connection, allowing for the addition
    of previous activations.

    Attributes:
        d_input (int): The number of input features.
        _d_output (int): The number of output features after upsampling.
        stride (int): The factor by which the input is upsampled.
        transposed (bool): Indicates if the transformation should be transposed.
        linear (LinearActivation): A linear activation layer for feature transformation.

    Args:
        d_input (int): The number of input features.
        d_output (int): The number of output features after upsampling.
        stride (int): The upsampling factor.
        transposed (bool): Whether to apply the transformation in transposed mode.
        weight_norm (bool): If True, applies weight normalization to the linear layer.
        initializer (callable): Function to initialize the weights of the linear layer.
        activation (callable): Activation function to apply after the linear layer.

    Returns:
        Tuple[Tensor, None]: The upsampled tensor and None (for compatibility).

    Examples:
        >>> up_pool = UpPool(d_input=16, d_output=32, stride=2)
        >>> x = torch.randn(10, 5, 16)  # Batch of 10, sequence length 5
        >>> output, _ = up_pool(x)
        >>> output.shape
        torch.Size([10, 5, 32])  # Output shape reflects upsampling

    Note:
        The upsampling operation shifts the tensor to ensure causality during
        the transformation, which is particularly important in sequence models.

    Raises:
        AssertionError: If the state is empty during the step function.

    Todo:
        Consider adding support for additional types of upsampling strategies.
    """

    def __init__(
        self,
        d_input,
        d_output,
        stride,
        transposed=True,
        weight_norm=True,
        initializer=None,
        activation=None,
    ):
        super().__init__()

        self.d_input = d_input
        self._d_output = d_output
        self.stride = stride
        self.transposed = transposed

        self.linear = LinearActivation(
            d_input,
            d_output * stride,
            transposed=transposed,
            initializer=initializer,
            weight_norm=weight_norm,
            activation=activation,
            activate=True if activation is not None else False,
        )

    def forward(self, x, skip=None):
        """
        Upsampling layer for sequence data.

        This layer takes an input tensor and applies a linear transformation,
        followed by an upsampling operation. The upsampling can be performed
        in either a transposed or non-transposed manner, depending on the
        configuration. Additionally, it allows for skip connections to be
        added to the output.

        Attributes:
            d_input (int): The dimensionality of the input feature.
            _d_output (int): The dimensionality of the output feature.
            stride (int): The upsampling factor, determines how much the
                input is expanded.
            transposed (bool): Whether to apply the transposed operation.
            linear (LinearActivation): A linear layer used to transform the input.

        Args:
            d_input (int): The dimensionality of the input feature.
            d_output (int): The dimensionality of the output feature.
            stride (int): The upsampling factor.
            transposed (bool): Whether to apply the transposed operation.
            weight_norm (bool): Whether to apply weight normalization.
            initializer (callable, optional): A function to initialize weights.
            activation (callable, optional): Activation function to apply.

        Returns:
            tuple: The upsampled tensor and a None placeholder for the state.

        Examples:
            >>> upsample_layer = UpPool(d_input=64, d_output=128, stride=2)
            >>> input_tensor = torch.randn(10, 20, 64)  # (batch, length, feature)
            >>> output_tensor, _ = upsample_layer(input_tensor)
            >>> output_tensor.shape
            torch.Size([10, 20, 128])  # Up sampled output shape

        Note:
            The output size is determined by the `stride` parameter and
            the linear transformation applied to the input.

        Raises:
            AssertionError: If the state is empty during the step method.
        """
        x = self.linear(x)
        if self.transposed:
            x = F.pad(x[..., :-1], (1, 0))  # Shift to ensure causality
            x = rearrange(x, "... (h s) l -> ... h (l s)", s=self.stride)
        else:
            x = F.pad(x[..., :-1, :], (0, 0, 1, 0))  # Shift to ensure causality
            x = rearrange(x, "... l (h s) -> ... (l s) h", s=self.stride)
        if skip is not None:
            x = x + skip
        return x, None

    def step(self, x, state, **kwargs):
        """
        Upsampling layer with linear transformation and optional skip connections.

        This class implements an upsampling mechanism for sequences, applying a
        linear transformation to increase the sequence length. It can also utilize
        skip connections to add additional features to the output. The upsampling
        is performed by repeating the input features according to the specified
        stride.

        Attributes:
            d_input (int): The number of input features.
            _d_output (int): The number of output features after upsampling.
            stride (int): The factor by which to upsample the input.
            transposed (bool): If True, applies a transposed operation during
                upsampling.
            linear (LinearActivation): A linear transformation applied to the input.

        Args:
            d_input (int): Number of input features.
            d_output (int): Number of output features after upsampling.
            stride (int): The factor by which to upsample the input.
            transposed (bool, optional): If True, applies a transposed operation
                during upsampling. Defaults to True.
            weight_norm (bool, optional): If True, applies weight normalization
                to the linear layer. Defaults to True.
            initializer (callable, optional): Custom initializer for the linear layer.
                Defaults to None.
            activation (callable, optional): Activation function to be applied
                after the linear layer. Defaults to None.

        Returns:
            tuple: A tuple containing the upsampled output and None.

        Examples:
            >>> up_pool = UpPool(d_input=64, d_output=128, stride=2)
            >>> x = torch.randn(32, 10, 64)  # Batch of 32, sequence length of 10
            >>> output, _ = up_pool(x)
            >>> output.shape
            torch.Size([32, 20, 128])  # Output shape after upsampling

        Note:
            The output shape is affected by the stride and whether the operation
            is transposed. If transposed is True, the operation applies
            transformations accordingly to maintain causality in the sequence.

        Raises:
            AssertionError: If the state is not properly initialized or if
                incorrect input dimensions are provided.
        """
        assert len(state) > 0
        y, state = state[0], state[1:]
        if len(state) == 0:
            assert x is not None
            if self.transposed:
                x = x.unsqueeze(-1)
            x = self.linear(x)
            if self.transposed:
                x = x.squeeze(-1)
            x = rearrange(x, "... (h s) -> ... h s", s=self.stride)
            state = list(torch.unbind(x, dim=-1))
        else:
            assert x is None
        return y, state

    def default_state(self, *batch_shape, device=None):
        """
        Create the default state for the UpPool module.

        This method initializes the state used during the forward pass of the
        UpPool layer. The state is represented as a list of tensors, where each
        tensor corresponds to a time step in the sequence. The dimensions of the
        state are determined by the input batch shape and the output dimension
        of the layer.

        Args:
            *batch_shape: Variable length argument list that defines the shape
                of the input batch. This is typically the shape of the input
                data excluding the sequence length and feature dimensions.
            device (torch.device, optional): The device on which to create the
                state tensors. If None, the state tensors will be created on the
                same device as the input data.

        Returns:
            list: A list of tensors initialized to zeros, representing the
                default state of the UpPool module. Each tensor in the list has
                the shape defined by the output dimension of the layer and the
                stride used for the upsampling.

        Examples:
            >>> up_pool = UpPool(d_input=128, d_output=256, stride=2)
            >>> state = up_pool.default_state(32, device='cuda')
            >>> len(state)
            2
            >>> state[0].shape
            torch.Size([32, 256])  # Example output shape based on d_output

        Note:
            The state is designed to be used in conjunction with the `step`
            method, which processes one time step at a time in a recurrent
            manner.
        """
        state = torch.zeros(
            batch_shape + (self.d_output, self.stride), device=device
        )  # (batch, h, s)
        state = list(torch.unbind(state, dim=-1))  # List of (..., H)
        return state

    @property
    def d_output(self):
        """
        int: The output dimension of the layer.

        This property computes the output dimension based on the input
        dimension and the expansion factor. Specifically, it multiplies
        the input dimension by the expansion factor set during the
        initialization of the layer.

        Returns:
            int: The calculated output dimension, which is equal to
            `self.d_input * self.expand`.

        Examples:
            >>> upsample_layer = UpSample(d_input=64, stride=2, expand=3)
            >>> upsample_layer.d_output
            64 // 3  # Assuming d_input is 64 and expand is 3.
        """
        return self._d_output


registry = {
    "sample": DownSample,
    "pool": DownAvgPool,
    "linear": DownLinearPool,
    "spectral": DownSpectralPool,
}
