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
    Downsample a sequence along the layer dimension and optionally expand along the feature dimension.

    This function performs downsampling on the input sequence by selecting elements at regular intervals
    and optionally expands the feature dimension by repeating values.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim) if not transposed,
                          or (batch_size, feature_dim, seq_len) if transposed.
        stride (int, optional): The stride for downsampling. Defaults to 1.
        expand (int, optional): The expansion factor for the feature dimension. Defaults to 1.
        transposed (bool, optional): Whether the input is in transposed format. Defaults to False.

    Returns:
        torch.Tensor: Downsampled and optionally expanded tensor.

    Raises:
        AssertionError: If the input tensor is not 3-dimensional when stride > 1.

    Examples:
        >>> x = torch.randn(32, 100, 64)  # (batch_size, seq_len, feature_dim)
        >>> y = downsample(x, stride=2, expand=2)
        >>> y.shape
        torch.Size([32, 50, 128])

    Note:
        - If stride > 1, the function only supports 3-dimensional inputs.
        - For higher-dimensional inputs with stride > 1, it is recommended to use average or spectral pooling instead.
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
    Upsample a sequence along the layer dimension and optionally reduce along the feature dimension.

    This function performs upsampling on the input sequence by repeating elements and optionally
    reduces the feature dimension by averaging values.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim) if not transposed,
                          or (batch_size, feature_dim, seq_len) if transposed.
        stride (int, optional): The stride for upsampling. Defaults to 1.
        expand (int, optional): The reduction factor for the feature dimension. Defaults to 1.
        transposed (bool, optional): Whether the input is in transposed format. Defaults to False.

    Returns:
        torch.Tensor: Upsampled and optionally reduced tensor.

    Examples:
        >>> x = torch.randn(32, 50, 128)  # (batch_size, seq_len, feature_dim)
        >>> y = upsample(x, stride=2, expand=2)
        >>> y.shape
        torch.Size([32, 100, 64])

    Note:
        - If expand > 1, the function reduces the feature dimension by taking the mean of groups of 'expand' features.
        - If stride > 1, the function repeats each element 'stride' times along the sequence dimension.
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
    A module for downsampling sequences.

    This class implements downsampling on input sequences by selecting elements at regular intervals
    and optionally expanding the feature dimension. It inherits from SequenceModule and can be used
    in sequential models.

    Attributes:
        d_input (int): The input feature dimension.
        stride (int): The stride for downsampling.
        expand (int): The expansion factor for the feature dimension.
        transposed (bool): Whether the input is in transposed format.

    Note:
        The `step` method is not implemented for stride > 1 or expand > 1.
    """

    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        """
            Forward pass for the DownSample module.

        This method applies downsampling to the input tensor using the specified stride and expand parameters.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim) if not transposed,
                              or (batch_size, feature_dim, seq_len) if transposed.

        Returns:
            torch.Tensor: Downsampled tensor with shape depending on the stride and expand parameters.

        Note:
            This method uses the `downsample` function internally, with `transposed=False` hardcoded.
            The actual transposition behavior is controlled by the `self.transposed` attribute.
        """
        return downsample(x, self.stride, self.expand, False, self.transposed)

    def step(self, x, state, **kwargs):
        """
            Perform a single step of the DownSample module.

        This method is intended for use in sequential or recurrent scenarios where the input is processed
        one step at a time.

        Args:
            x (torch.Tensor): Input tensor for the current step.
            state: The current state of the module (unused in this implementation).
            **kwargs: Additional keyword arguments (unused in this implementation).

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output tensor for the current step (same as input).
                - state: The updated state (unchanged in this implementation).

        Raises:
            NotImplementedError: If stride > 1 or expand > 1, as these cases are not supported for step-wise processing.

        Note:
            This method currently only supports stride=1 and expand=1. For other values, it raises a NotImplementedError.
        """
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        """
            int: The output feature dimension of the DownSample module.

        This property calculates and returns the output feature dimension based on the input dimension
        and the expansion factor.

        Returns:
            int: The output feature dimension, which is the product of the input dimension and the expand factor.

        Note:
            This property is useful for determining the output shape of the module, especially when
            chaining multiple modules together in a sequential model.
        """
        return self.d_input * self.expand


class DownAvgPool(SequenceModule):
    """
    A module for downsampling sequences using average pooling.

    This class implements downsampling on input sequences by applying average pooling
    and optionally expanding the feature dimension. It inherits from SequenceModule and
    can be used in sequential models.

    Attributes:
        d_input (int): The input feature dimension.
        stride (int): The stride for downsampling.
        expand (int): The expansion factor for the feature dimension.
        transposed (bool): Whether the input is in transposed format.

    Note:
        - This module supports multi-dimensional inputs (e.g., 1D, 2D, or higher).
        - The `step` method is not implemented for stride > 1 or expand > 1.
        - Average pooling is applied along the sequence dimension(s) when stride > 1.
        - Feature expansion is performed by repeating when expand > 1.
    """

    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        """
            Forward pass for the DownAvgPool module.

        This method applies average pooling for downsampling and optionally expands the feature dimension
        of the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., feature_dim) if not transposed,
                              or (batch_size, feature_dim, ...) if transposed.

        Returns:
            torch.Tensor: Downsampled and optionally expanded tensor.

        Note:
            - If not transposed, the input is rearranged to (batch_size, feature_dim, ...) before processing.
            - Average pooling is applied using F.avg_pool1d for 3D inputs, F.avg_pool2d for 4D inputs,
              and a custom reduction for higher dimensions.
            - If expand > 1, the feature dimension is expanded by repeating.
            - The output is rearranged back to the original format if not transposed.
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
            Perform a single step of the DownAvgPool module.

        This method is intended for use in sequential or recurrent scenarios where the input is processed
        one step at a time.

        Args:
            x (torch.Tensor): Input tensor for the current step.
            state: The current state of the module (unused in this implementation).
            **kwargs: Additional keyword arguments (unused in this implementation).

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output tensor for the current step (same as input).
                - state: The updated state (unchanged in this implementation).

        Raises:
            NotImplementedError: If stride > 1 or expand > 1, as these cases are not supported for step-wise processing.

        Note:
            This method currently only supports stride=1 and expand=1. For other values, it raises a NotImplementedError.
        """
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        """
            int: The output feature dimension of the DownAvgPool module.

        This property calculates and returns the output feature dimension based on the input dimension
        and the expansion factor.

        Returns:
            int: The output feature dimension, which is the product of the input dimension and the expand factor.

        Note:
            This property is useful for determining the output shape of the module, especially when
            chaining multiple modules together in a sequential model. The output dimension is expanded
            if the `expand` attribute is greater than 1.
        """
        return self.d_input * self.expand


class DownSpectralPool(SequenceModule):
    """
    A module for downsampling sequences using spectral pooling.

    This class implements downsampling on input sequences by applying spectral pooling
    in the frequency domain and optionally expanding the feature dimension. It inherits
    from SequenceModule and can be used in sequential models.

    Spectral pooling is performed by truncating high-frequency components in the Fourier domain,
    which can preserve more information compared to traditional spatial pooling methods.

    Attributes:
        d_input (int): The input feature dimension.
        stride (int): The stride for downsampling.
        expand (int): The expansion factor for the feature dimension.
        transposed (bool): Whether the input is in transposed format.

    Note:
        - This module supports multi-dimensional inputs.
        - The `step` method is not implemented for stride > 1 or expand > 1.
        - Spectral pooling is applied along all spatial dimensions when stride > 1.
        - Feature expansion is performed by repeating when expand > 1.
    """

    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        """
            Forward pass for the DownSpectralPool module.

        This method applies spectral pooling for downsampling and optionally expands the feature dimension
        of the input tensor. The spectral pooling is performed in the frequency domain using FFT.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L..., D) where B is batch size, L... are spatial dimensions,
                              and D is the feature dimension.

        Returns:
            torch.Tensor: Downsampled and optionally expanded tensor.

        Raises:
            AssertionError: If any spatial dimension length is not divisible by the stride.

        Note:
            - If not transposed, the input is rearranged to (B, D, L...) before processing.
            - The method performs the following steps:
              1. Applies FFT to the input tensor.
              2. Truncates high-frequency components based on the stride.
              3. Applies inverse FFT to return to the spatial domain.
              4. Expands the feature dimension if expand > 1.
            - The output is rearranged back to the original format if not transposed.
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
            Perform a single step of the DownSpectralPool module.

        This method is intended for use in sequential or recurrent scenarios where the input is processed
        one step at a time.

        Args:
            x (torch.Tensor): Input tensor for the current step.
            state: The current state of the module (unused in this implementation).
            **kwargs: Additional keyword arguments (unused in this implementation).

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output tensor for the current step (same as input).
                - state: The updated state (unchanged in this implementation).

        Raises:
            NotImplementedError: If stride > 1 or expand > 1, as these cases are not supported for step-wise processing.

        Note:
            This method currently only supports stride=1 and expand=1. For other values, it raises a NotImplementedError.
            Spectral pooling is not performed in the step method, as it requires processing the entire sequence.
        """
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        """
            int: The output feature dimension of the DownSpectralPool module.

        This property calculates and returns the output feature dimension based on the input dimension
        and the expansion factor.

        Returns:
            int: The output feature dimension, which is the product of the input dimension and the expand factor.

        Note:
            This property is useful for determining the output shape of the module, especially when
            chaining multiple modules together in a sequential model. The output dimension is expanded
            if the `expand` attribute is greater than 1, while the spatial dimensions are reduced based
            on the `stride` attribute.
        """
        return self.d_input * self.expand


class UpSample(nn.Module):
    """
    A module for upsampling sequences.

    This class implements upsampling on input sequences by repeating elements along the sequence dimension
    and optionally reducing the feature dimension. It inherits from nn.Module and can be used in
    neural network architectures.

    Attributes:
        d_input (int): The input feature dimension.
        stride (int): The stride for upsampling (number of times each element is repeated).
        expand (int): The reduction factor for the feature dimension.
        transposed (bool): Whether the input is in transposed format.

    Note:
        - The actual upsampling is performed in the forward method using the `upsample` function.
        - The `step` method is not implemented for stride > 1 or expand > 1.
        - The output dimension is calculated as d_input // expand, effectively reducing the feature dimension.
    """

    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        """
            Forward pass for the UpSample module.

        This method applies upsampling to the input tensor using the specified stride and expand parameters.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim) if not transposed,
                              or (batch_size, feature_dim, seq_len) if transposed.

        Returns:
            torch.Tensor: Upsampled tensor with shape depending on the stride and expand parameters.

        Note:
            This method uses the `upsample` function internally. The upsampling process involves:
            1. Repeating elements along the sequence dimension based on the `stride` parameter.
            2. Reducing the feature dimension based on the `expand` parameter (if > 1).
            The exact behavior depends on whether the input is in transposed format or not.
        """
        return upsample(x, self.stride, self.expand, self.transposed)

    @property
    def d_output(self):
        """
            int: The output feature dimension of the UpSample module.

        This property calculates and returns the output feature dimension based on the input dimension
        and the expand factor.

        Returns:
            int: The output feature dimension, which is the input dimension divided by the expand factor.

        Note:
            This property is useful for determining the output shape of the module, especially when
            chaining multiple modules together in a sequential model. The output dimension is reduced
            if the `expand` attribute is greater than 1, reflecting the feature dimension reduction
            that occurs during upsampling.
        """
        return self.d_input // self.expand

    def step(self, x, state, **kwargs):
        """
            Perform a single step of the UpSample module.

        This method is intended for use in sequential or recurrent scenarios where the input is processed
        one step at a time.

        Args:
            x (torch.Tensor): Input tensor for the current step.
            state: The current state of the module (unused in this implementation).
            **kwargs: Additional keyword arguments (unused in this implementation).

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output tensor for the current step (same as input).
                - state: The updated state (unchanged in this implementation).

        Raises:
            NotImplementedError: If stride > 1 or expand > 1, as these cases are not supported for step-wise processing.

        Note:
            This method currently only supports stride=1 and expand=1. For other values, it raises a NotImplementedError.
            Upsampling is not performed in the step method, as it requires processing multiple time steps at once.
        """
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state


""" Pooling functions with trainable parameters """
# For the flexible backbone SequenceModel


class DownLinearPool(SequenceModule):
    """
    A module for downsampling sequences using linear pooling.

    This class implements downsampling on input sequences by applying a linear transformation
    to groups of elements and optionally expanding the feature dimension. It inherits from
    SequenceModule and can be used in sequential models.

    Attributes:
        d_input (int): The input feature dimension.
        stride (int): The stride for downsampling (number of elements to group).
        expand (int): The expansion factor for the feature dimension.
        transposed (bool): Whether the input is in transposed format.
        linear (LinearActivation): The linear transformation applied to grouped elements.

    Note:
        - The linear transformation is applied to groups of `stride` elements along the sequence dimension.
        - The output feature dimension is expanded by a factor of `expand`.
        - The `step` method is not implemented for stride > 1 or expand > 1.
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
            Forward pass for the DownLinearPool module.

        This method applies linear pooling for downsampling and optionally expands the feature dimension
        of the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim) if not transposed,
                              or (batch_size, feature_dim, seq_len) if transposed.

        Returns:
            torch.Tensor: Downsampled and optionally expanded tensor.

        Note:
            - If transposed, the input is rearranged to group `stride` elements along the last dimension.
            - If not transposed, the input is rearranged to group `stride` elements along the second-to-last dimension.
            - The linear transformation is applied to the grouped elements, effectively downsampling the sequence.
            - The output tensor's shape depends on the stride, expand, and transposed parameters.
        """
        if self.transposed:
            x = rearrange(x, "... h (l s) -> ... (h s) l", s=self.stride)
        else:
            x = rearrange(x, "... (l s) h -> ... l (h s)", s=self.stride)
        x = self.linear(x)
        return x

    def step(self, x, state, **kwargs):
        """
            Perform a single step of the DownLinearPool module.

        This method is intended for use in sequential or recurrent scenarios where the input is processed
        one step at a time.

        Args:
            x (torch.Tensor): Input tensor for the current step.
            state: The current state of the module (unused in this implementation).
            **kwargs: Additional keyword arguments (unused in this implementation).

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output tensor for the current step (same as input).
                - state: The updated state (unchanged in this implementation).

        Raises:
            NotImplementedError: If stride > 1 or expand > 1, as these cases are not supported for step-wise processing.

        Note:
            This method currently only supports stride=1 and expand=1. For other values, it raises a NotImplementedError.
            Linear pooling is not performed in the step method, as it requires processing multiple time steps at once.
        """
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        """
            int: The output feature dimension of the DownLinearPool module.

        This property calculates and returns the output feature dimension based on the input dimension
        and the expansion factor.

        Returns:
            int: The output feature dimension, which is the product of the input dimension and the expand factor.

        Note:
            This property is useful for determining the output shape of the module, especially when
            chaining multiple modules together in a sequential model. The output dimension is expanded
            by the `expand` factor, reflecting the feature dimension expansion that occurs during
            the linear pooling operation.
        """
        return self.d_input * self.expand


""" Pooling functions with trainable parameters """


class DownPool2d(SequenceModule):
    """
    A module for downsampling 2D sequences using a combination of linear transformation and average pooling.

    This class implements downsampling on 2D input sequences by first applying a linear transformation
    to change the feature dimension, followed by average pooling to reduce spatial dimensions. It inherits
    from SequenceModule and can be used in sequential models dealing with 2D data.

    Attributes:
        linear (LinearActivation): Linear transformation applied to the input before pooling.
        pool (nn.AvgPool2d): Average pooling layer for spatial downsampling.

    Note:
        - The linear transformation allows for changing the feature dimension before pooling.
        - The average pooling operation reduces the spatial dimensions based on the given stride.
        - This module is particularly useful for processing 2D data such as images or feature maps.
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
            Forward pass for the DownPool2d module.

        This method applies the linear transformation followed by average pooling to downsample the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width) if transposed,
                              or (batch_size, height, width, channels) if not transposed.

        Returns:
            torch.Tensor: Downsampled tensor after linear transformation and pooling.

        Note:
            - If transposed, the pooling operation is applied directly after the linear transformation.
            - The implementation for non-transposed input is not provided in the given code snippet.
            - The output tensor's spatial dimensions will be reduced based on the pooling parameters.
        """
        if self.transposed:
            x = self.pool(x)


# DownLinearPool is used by the registry (for isotropic backbone)
# DownPool is essentially the same as DownLinearPool. These should be consolidated
class DownPool(SequenceModule):
    """
    A flexible module for downsampling sequences using linear transformation.

    This class implements downsampling on input sequences by applying a linear transformation
    to groups of elements. It allows for both downsampling and feature dimension adjustment.
    It inherits from SequenceModule and can be used in sequential models.

    Attributes:
        d_output (int): The output feature dimension.
        stride (int): The stride for downsampling (number of elements to group).
        transposed (bool): Whether the input is in transposed format.
        linear (LinearActivation): The linear transformation applied to grouped elements.

    Note:
        - The linear transformation is applied to groups of `stride` elements along the sequence dimension.
        - This module can both downsample the sequence and adjust the feature dimension.
        - Supports optional weight normalization and activation function in the linear transformation.
        - The `step` method provides functionality for recurrent processing.
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
            Forward pass for the DownPool module.

        This method applies downsampling and linear transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim) if not transposed,
                              or (batch_size, feature_dim, seq_len) if transposed.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Downsampled and transformed tensor.
                - None: Placeholder for consistency with other modules that might return additional information.

        Note:
            - If transposed, the input is rearranged to group `stride` elements along the second dimension.
            - If not transposed, the input is rearranged to group `stride` elements along the second-to-last dimension.
            - The linear transformation is applied to the grouped elements, effectively downsampling the sequence.
            - The output tensor's shape depends on the stride, d_output, and transposed parameters.
        """
        if self.transposed:
            x = rearrange(x, "... h (l s) -> ... (h s) l", s=self.stride)
        else:
            x = rearrange(x, "... (l s) h -> ... l (h s)", s=self.stride)
        x = self.linear(x)
        return x, None

    def step(self, x, state, **kwargs):
        """
            Perform a single step of the DownPool module for recurrent processing.

        This method allows the DownPool module to be used in a recurrent manner, processing
        one time step at a time and maintaining an internal state.

        Args:
            x (torch.Tensor or None): Input tensor for the current step, or None if no input is available.
            state (list): Current state of the module, containing previously seen inputs.
            **kwargs: Additional keyword arguments (unused in this implementation).

        Returns:
            tuple: A tuple containing:
                - torch.Tensor or None: Output tensor if a full group is processed, otherwise None.
                - list: Updated state of the module.

        Note:
            - The method accumulates inputs in the state until it has gathered 'stride' elements.
            - When 'stride' elements are gathered, it applies the linear transformation and resets the state.
            - If x is None, it returns None and the current state.
            - This method enables the module to work with variable-length sequences in a recurrent setting.
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
            Initialize the default state for the DownPool module.

        This method creates the initial state for the module when used in a recurrent setting.

        Args:
            *batch_shape: Variable length argument to specify the batch dimensions.
            device (torch.device, optional): The device on which to create the state. Defaults to None.

        Returns:
            list: An empty list representing the initial state of the module.

        Note:
            - The default state is an empty list, which will be filled with input tensors during the step method.
            - This method is typically called once before starting the recurrent processing of a sequence.
            - The batch_shape and device arguments are included for compatibility with other modules
              but are not used in this implementation.
        """
        return []


class UpPool(SequenceModule):
    """
    A module for upsampling sequences using linear transformation.

    This class implements upsampling on input sequences by applying a linear transformation
    followed by reshaping to increase the sequence length. It inherits from SequenceModule
    and can be used in sequential models.

    Attributes:
        d_input (int): The input feature dimension.
        _d_output (int): The output feature dimension.
        stride (int): The upsampling factor (increase in sequence length).
        transposed (bool): Whether the input is in transposed format.
        linear (LinearActivation): The linear transformation applied before upsampling.

    Note:
        - The linear transformation expands the feature dimension by a factor of `stride`.
        - The output is then reshaped to increase the sequence length by `stride`.
        - Supports optional weight normalization and activation function in the linear transformation.
        - Includes a step method for recurrent processing and a default_state method for initialization.
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
            Forward pass for the UpPool module.

        This method applies upsampling to the input tensor using linear transformation and reshaping.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim) if not transposed,
                              or (batch_size, feature_dim, seq_len) if transposed.
            skip (torch.Tensor, optional): Skip connection input to be added after upsampling. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Upsampled tensor, potentially with skip connection added.
                - None: Placeholder for consistency with other modules that might return additional information.

        Note:
            - The input is first passed through a linear transformation to expand the feature dimension.
            - The transformed tensor is then reshaped to increase the sequence length by the stride factor.
            - A shift operation is applied to ensure causality in the upsampled sequence.
            - If a skip connection is provided, it's added to the upsampled tensor.
            - The output tensor's shape depends on the stride, d_output, and transposed parameters.
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
            Perform a single step of the UpPool module for recurrent processing.

        This method allows the UpPool module to be used in a recurrent manner, processing
        one time step at a time and maintaining an internal state.

        Args:
            x (torch.Tensor or None): Input tensor for the current step, or None if no input is available.
            state (list): Current state of the module, containing future outputs.
            **kwargs: Additional keyword arguments (unused in this implementation).

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Output tensor for the current step.
                - list: Updated state of the module.

        Note:
            - The method returns the next output from the state and updates the state.
            - If the state is empty and x is provided, it applies the linear transformation and generates new outputs.
            - This method enables the module to work with variable-length sequences in a recurrent setting.
            - It maintains causality by only using current and past inputs to generate outputs.
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
            Initialize the default state for the UpPool module.

        This method creates the initial state for the module when used in a recurrent setting.

        Args:
            *batch_shape: Variable length argument to specify the batch dimensions.
            device (torch.device, optional): The device on which to create the state. Defaults to None.

        Returns:
            list: A list of torch.Tensor objects representing the initial state of the module.

        Note:
            - The default state is a list of zero tensors, with shape (batch_shape, d_output, stride).
            - Each tensor in the list represents a future output in the upsampled sequence.
            - The state is initialized with 'stride' number of zero tensors.
            - This method is typically called once before starting the recurrent processing of a sequence.
            - The state allows the module to maintain the expanded sequence length during step-wise processing.
        """
        state = torch.zeros(
            batch_shape + (self.d_output, self.stride), device=device
        )  # (batch, h, s)
        state = list(torch.unbind(state, dim=-1))  # List of (..., H)
        return state

    @property
    def d_output(self):
        """
            int: The output feature dimension of the UpPool module.

        This property returns the output feature dimension of the module.

        Returns:
            int: The output feature dimension.

        Note:
            - This property returns the value of the `_d_output` attribute, which is set during initialization.
            - The output dimension is typically different from the input dimension due to the upsampling process.
            - This property is useful for determining the output shape of the module, especially when
              chaining multiple modules together in a sequential model.
        """
        return self._d_output


registry = {
    "sample": DownSample,
    "pool": DownAvgPool,
    "linear": DownLinearPool,
    "spectral": DownSpectralPool,
}
