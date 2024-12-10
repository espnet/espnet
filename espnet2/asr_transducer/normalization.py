"""Normalization modules for Transducer."""

from typing import Dict, Optional, Tuple

import torch


def get_normalization(
    normalization_type: str,
    eps: Optional[float] = None,
    partial: Optional[float] = None,
) -> Tuple[torch.nn.Module, Dict]:
    """
    Normalization modules for Transducer.

    This module provides various normalization techniques used in 
    transducer models. It includes basic normalization, layer normalization, 
    RMS normalization, and scale normalization. The `get_normalization` 
    function retrieves the appropriate normalization module based on the 
    specified type.

    Functions:
        get_normalization: Get normalization module and arguments given parameters.

    Classes:
        BasicNorm: Basic normalization module.
        RMSNorm: RMS normalization module.
        ScaleNorm: Scale normalization module.
    """
    norm = {
        "basic_norm": (
            BasicNorm,
            {"eps": eps if eps is not None else 0.25},
        ),
        "layer_norm": (torch.nn.LayerNorm, {"eps": eps if eps is not None else 1e-12}),
        "rms_norm": (
            RMSNorm,
            {
                "eps": eps if eps is not None else 1e-05,
                "partial": partial if partial is not None else -1.0,
            },
        ),
        "scale_norm": (
            ScaleNorm,
            {"eps": eps if eps is not None else 1e-05},
        ),
    }

    return norm[normalization_type]


class BasicNorm(torch.nn.Module):
    """
    BasicNorm module for performing basic normalization.

    This module applies basic normalization to input tensors, which involves
    scaling the input based on the mean of the squares of its elements for
    each feature. It is useful for stabilizing training in neural networks.

    Reference:
        https://github.com/k2-fsa/icefall/pull/288

    Args:
        normalized_shape (int): Expected size of the input features.
        eps (float, optional): Value added to the denominator for numerical
            stability. Default is 0.25.

    Examples:
        >>> import torch
        >>> basic_norm = BasicNorm(normalized_shape=10, eps=0.1)
        >>> input_tensor = torch.randn(2, 5, 10)  # (B, T, D_hidden)
        >>> output_tensor = basic_norm(input_tensor)
        >>> output_tensor.shape
        torch.Size([2, 5, 10])
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 0.25,
    ) -> None:
        """Construct a BasicNorm object."""
        super().__init__()

        self.eps = torch.nn.Parameter(torch.tensor(eps).log().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute basic normalization.

        This method applies basic normalization to the input tensor `x` by 
        computing the scaling factor based on the mean of the squares of the 
        input elements and applying it to the input tensor.

        Args:
            x: Input sequences. Shape (B, T, D_hidden), where:
            - B is the batch size,
            - T is the sequence length,
            - D_hidden is the dimensionality of the hidden states.

        Returns:
            Tensor: Output sequences after applying basic normalization. Shape 
            (B, T, D_hidden).

        Examples:
            >>> norm = BasicNorm(normalized_shape=128)
            >>> input_tensor = torch.randn(32, 10, 128)  # Batch of 32, seq length 10
            >>> output_tensor = norm(input_tensor)
            >>> output_tensor.shape
            torch.Size([32, 10, 128])
        
        Note:
            The normalization is performed by scaling the input tensor based on 
            the computed scales which are derived from the mean of the squares 
            of the input tensor.
        """
        scales = (torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps.exp()) ** -0.5

        return x * scales


class RMSNorm(torch.nn.Module):
    """
    RMSNorm module definition.

    This class implements the Root Mean Square Layer Normalization (RMSNorm),
    which normalizes the input using the root mean square of the input values
    along the specified dimensions. RMSNorm is beneficial for stabilizing 
    training and improving model performance.

    Reference:
        - https://arxiv.org/pdf/1910.07467.pdf

    Args:
        normalized_shape (int): The expected size of the input tensor for
            normalization.
        eps (float, optional): A small value added to the denominator for 
            numerical stability. Default is 1e-5.
        partial (float, optional): A value defining the part of the input used
            for RMS statistics. It should be in the range (0, 1). Default is 0.0,
            which means RMS statistics will be computed over the entire input.

    Attributes:
        normalized_shape (int): The shape of the input tensor.
        partial (bool): A boolean indicating whether to use partial normalization.
        p (float): The fraction of the input to be used for RMS statistics.
        eps (float): The small value for numerical stability.
        scale (torch.nn.Parameter): Learnable scale parameter.

    Examples:
        >>> rms_norm = RMSNorm(normalized_shape=256, eps=1e-5, partial=0.5)
        >>> input_tensor = torch.randn(32, 10, 256)  # Batch size of 32
        >>> output_tensor = rms_norm(input_tensor)
        >>> output_tensor.shape
        torch.Size([32, 10, 256])

    Returns:
        torch.Tensor: The normalized output tensor with the same shape as the
        input tensor.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        partial: float = 0.0,
    ) -> None:
        """Construct a RMSNorm object."""
        super().__init__()

        self.normalized_shape = normalized_shape

        self.partial = True if 0 < partial < 1 else False
        self.p = partial
        self.eps = eps

        self.scale = torch.nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm module definition.

        This module applies Root Mean Square (RMS) normalization to the input tensor.
        RMS normalization helps in stabilizing the training of deep neural networks
        by ensuring that the input to each layer has a consistent scale.

        Reference: https://arxiv.org/pdf/1910.07467.pdf

        Args:
            normalized_shape: Expected size of the input tensor.
            eps: Value added to the denominator for numerical stability. Default is 1e-5.
            partial: Value defining the part of the input used for RMS stats. If this 
                value is between 0 and 1, only a portion of the input is used to 
                compute the RMS statistics. Default is 0.0, which means full input is 
                used.

        Attributes:
            normalized_shape: The expected size of the input tensor.
            partial: A boolean indicating whether partial RMS statistics should be used.
            p: The proportion of the input to use for RMS stats if partial is enabled.
            eps: The epsilon value for numerical stability.
            scale: A learnable parameter for scaling the normalized output.

        Examples:
            >>> rms_norm = RMSNorm(normalized_shape=64, eps=1e-5, partial=0.5)
            >>> input_tensor = torch.randn(32, 10, 64)  # (Batch, Time, Features)
            >>> output_tensor = rms_norm(input_tensor)
            >>> output_tensor.shape
            torch.Size([32, 10, 64])  # Output shape matches input shape

        Returns:
            x: Output sequences after RMS normalization. Shape is (B, T, D_hidden).
        """
        if self.partial:
            partial_size = int(self.normalized_shape * self.p)
            partial_x, _ = torch.split(
                x, [partial_size, self.normalized_shape - partial_size], dim=-1
            )

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        else:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.normalized_shape

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x = self.scale * (x / (rms_x + self.eps))

        return x


class ScaleNorm(torch.nn.Module):
    """
    ScaleNorm module definition.

    This module performs scale normalization on the input tensor. The scaling
    is based on the L2 norm of the input, ensuring that the output has a 
    consistent scale. This can be useful in various neural network architectures
    where normalization of activations is required.

    Reference:
        https://arxiv.org/pdf/1910.05895.pdf

    Args:
        normalized_shape: Expected size of the input tensor for normalization.
        eps: A small value added to the denominator for numerical stability
            (default: 1e-5).

    Examples:
        >>> scale_norm = ScaleNorm(normalized_shape=128)
        >>> input_tensor = torch.randn(32, 10, 128)  # (B, T, D_hidden)
        >>> output_tensor = scale_norm(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([32, 10, 128])

    Returns:
        Output sequences with the same shape as the input, scaled based on
        the computed norm.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        """Construct a ScaleNorm object."""
        super().__init__()

        self.eps = eps
        self.scale = torch.nn.Parameter(torch.tensor(normalized_shape**0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ScaleNorm module definition.

        Reference: https://arxiv.org/pdf/1910.05895.pdf

        Args:
            normalized_shape: Expected size of the input tensor.
            eps: Value added to the denominator for numerical stability (default: 1e-5).

        Attributes:
            scale: A learnable parameter representing the scaling factor.

        Methods:
            forward(x: torch.Tensor) -> torch.Tensor:
                Computes the scale normalization for the input tensor.

        Examples:
            >>> import torch
            >>> scale_norm = ScaleNorm(normalized_shape=10)
            >>> input_tensor = torch.randn(5, 2, 10)  # Batch size of 5, seq length of 2
            >>> output_tensor = scale_norm(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([5, 2, 10])
        """
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)

        return x * norm
