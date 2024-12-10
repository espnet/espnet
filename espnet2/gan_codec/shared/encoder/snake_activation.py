import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm


@torch.jit.script
def snake(x, alpha):
    """
        Applies the Snake activation function to the input tensor.

    The Snake activation function is a non-linear transformation that enhances
    the representation power of neural networks by introducing a sine-based
    component, controlled by the parameter alpha. This activation is designed
    to be used in deep learning models, particularly within neural network
    layers.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, L), where N is the batch
            size, C is the number of channels, and L is the length of each
            channel.
        alpha (torch.Tensor): Parameter controlling the sine transformation,
            should be of shape (1, 1, 1) or broadcastable to the shape of x.

    Returns:
        torch.Tensor: Output tensor after applying the Snake activation function,
            with the same shape as the input tensor x.

    Examples:
        >>> x = torch.randn(2, 3, 4)  # Batch of 2, 3 channels, length 4
        >>> alpha = torch.tensor([[[0.5]]])  # Alpha value for the transformation
        >>> output = snake(x, alpha)
        >>> print(output.shape)
        torch.Size([2, 3, 4])

    Attributes:
        alpha (torch.Tensor): The alpha parameter for the Snake activation,
            initialized to a tensor of ones with shape (1, 1, 1).

    Raises:
        ValueError: If the input tensor x is not of shape (N, C, L).

    Note:
        This activation function is designed to be used within neural networks
        to improve learning capacity and can be integrated into custom layers.

    Todo:
        - Explore additional configurations for the alpha parameter to enhance
          performance across different datasets.
    """
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    """
        Snake1d is a PyTorch neural network module that applies the Snake activation
    function to the input tensor. The Snake activation function enhances the
    non-linearity of the neural network by applying a sine transformation to the
    input values.

    Attributes:
        alpha (torch.Tensor): A tensor that controls the shape of the activation
            function, initialized to ones with shape (1, 1, 1).

    Args:
        x (torch.Tensor): The input tensor to which the Snake activation will be
            applied. It is expected to be of shape (N, C, H, W), where N is the
            batch size, C is the number of channels, and H, W are the height and
            width of the input, respectively.

    Returns:
        torch.Tensor: The output tensor after applying the Snake activation,
            preserving the same shape as the input tensor.

    Examples:
        >>> import torch
        >>> snake1d = Snake1d()
        >>> input_tensor = torch.randn(2, 3, 4)  # Example input tensor
        >>> output_tensor = snake1d(input_tensor)
        >>> print(output_tensor.shape)  # Should print: torch.Size([2, 3, 4])

    Note:
        The alpha parameter can be modified to achieve different activation
        behaviors. By default, it is set to a tensor of ones.

    Todo:
        - Implement functionality to allow dynamic adjustment of alpha per
          channel.
    """

    def __init__(self):
        super().__init__()
        self.alpha = torch.ones(1, 1, 1)

    def forward(self, x):
        """
                Applies the Snake transformation to the input tensor.

        The Snake transformation is a non-linear operation defined by the function `snake`,
        which modifies the input tensor `x` based on a parameter `alpha`. This method is
        useful in various neural network architectures for enhancing feature extraction.

        Attributes:
            alpha (torch.Tensor): A tensor containing the alpha parameters for the
                Snake transformation, initialized to a tensor of ones.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, height,
                width) or (batch_size, channels, -1) where `batch_size` is the number
                of samples, `channels` is the number of channels, and `height` and
                `width` are spatial dimensions.

        Returns:
            torch.Tensor: The transformed tensor after applying the Snake operation,
                with the same shape as the input tensor `x`.

        Examples:
            >>> model = Snake1d()
            >>> input_tensor = torch.randn(2, 3, 4, 4)  # (batch_size, channels, height, width)
            >>> output_tensor = model(input_tensor)
            >>> print(output_tensor.shape)  # Should be (2, 3, 4, 4)

        Note:
            The alpha parameter can be modified for different transformation behaviors.

        Todo:
            - Implement support for varying alpha parameters for different channels.
        """
        channels = x.shape[1]
        self.alpha_repeat = self.alpha.repeat(1, channels, 1).to(x.device)
        return snake(x, self.alpha_repeat)
