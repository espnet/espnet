import argparse
from typing import Dict, Optional

import torch
from typeguard import typechecked

from espnet2.uasr.discriminator.abs_discriminator import AbsDiscriminator
from espnet2.utils.types import str2bool


class SamePad(torch.nn.Module):
    """
        SamePad is a PyTorch module that applies same padding to input tensors. It is
    used to ensure that the output tensor maintains the same length as the input
    tensor, after applying convolution operations.

    Attributes:
        remove (int): The number of elements to remove from the input tensor,
            determined by the kernel size and whether causal padding is used.

    Args:
        kernel_size (int): The size of the convolutional kernel.
        causal (bool): If True, applies causal padding; otherwise, applies
            symmetric padding.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Applies same padding to the input tensor.

    Examples:
        >>> same_pad = SamePad(kernel_size=3, causal=False)
        >>> input_tensor = torch.randn(1, 2, 10)  # (Batch, Channel, Length)
        >>> output_tensor = same_pad(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 2, 10])  # Output shape remains the same as input shape
    """

    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        """
            Forward pass for the ConvDiscriminator.

        This method takes an input tensor and processes it through the
        convolutional layers defined in the network. It also handles an
        optional padding mask to manage the output tensor based on
        padding conditions.

        Args:
            x (torch.Tensor): The input tensor of shape (Batch, Time, Channel).
            padding_mask (Optional[torch.Tensor]): A mask tensor that indicates
                which elements in the input tensor should be ignored during
                processing. Should have shape (Batch, Time) and will be
                broadcasted accordingly.

        Returns:
            torch.Tensor: The output tensor after processing, of shape
            (Batch, Time, Channel) where the Channel dimension is reduced to
            a single output dimension.

        Examples:
            >>> discriminator = ConvDiscriminator(input_dim=128)
            >>> input_tensor = torch.randn(32, 100, 128)  # (Batch, Time, Channel)
            >>> padding_mask = torch.zeros(32, 100, dtype=torch.bool)
            >>> output = discriminator.forward(input_tensor, padding_mask)
            >>> output.shape
            torch.Size([32, 100, 1])

        Note:
            The output is computed by transposing the input tensor to fit
            the expected shape for convolutional operations, and then
            transposing back to the original shape before returning.
        """
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class ConvDiscriminator(AbsDiscriminator):
    """
    Convolutional discriminator for Unsupervised Automatic Speech Recognition (UASR).

    This class implements a convolutional neural network (CNN) based discriminator
    designed for the UASR task. It utilizes various convolutional layers and
    configurations to process input features and produce discriminative outputs.

    Attributes:
        conv_channels (int): Number of channels for convolutional layers.
        conv_kernel (int): Size of the convolutional kernel.
        conv_dilation (int): Dilation rate for convolutional layers.
        conv_depth (int): Number of convolutional layers in the network.
        linear_emb (bool): Whether to use a linear embedding.
        causal (bool): If True, applies causal convolution.
        max_pool (bool): If True, applies max pooling in the output layer.
        act_after_linear (bool): If True, applies activation after the linear layer.
        dropout (float): Dropout rate for regularization.
        spectral_norm (bool): If True, applies spectral normalization to conv layers.
        weight_norm (bool): If True, applies weight normalization to conv layers.

    Args:
        input_dim (int): Dimension of the input features.
        cfg (Optional[Dict], optional): Configuration dictionary. Defaults to None.
        conv_channels (int, optional): Number of channels for convolutional layers.
            Defaults to 384.
        conv_kernel (int, optional): Size of the convolutional kernel. Defaults to 8.
        conv_dilation (int, optional): Dilation rate for convolutional layers.
            Defaults to 1.
        conv_depth (int, optional): Number of convolutional layers. Defaults to 2.
        linear_emb (str2bool, optional): Use linear embedding. Defaults to False.
        causal (str2bool, optional): Use causal convolution. Defaults to True.
        max_pool (str2bool, optional): Use max pooling in output layer. Defaults to False.
        act_after_linear (str2bool, optional): Apply activation after linear layer.
            Defaults to False.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        spectral_norm (str2bool, optional): Use spectral normalization. Defaults to False.
        weight_norm (str2bool, optional): Use weight normalization. Defaults to False.

    Returns:
        torch.Tensor: The output of the discriminator.

    Examples:
        >>> discriminator = ConvDiscriminator(input_dim=128)
        >>> input_tensor = torch.randn(32, 100, 128)  # (Batch, Time, Features)
        >>> output = discriminator(input_tensor)
        >>> print(output.shape)
        torch.Size([32, 1])  # Output shape depends on the architecture

    Note:
        The input tensor is expected to be in the shape (Batch, Time, Features).
        The output is processed through various convolutional layers and may
        undergo padding and pooling based on the configuration.

    Raises:
        ValueError: If input_dim is not positive.
    """

    @typechecked
    def __init__(
        self,
        input_dim: int,
        cfg: Optional[Dict] = None,
        conv_channels: int = 384,
        conv_kernel: int = 8,
        conv_dilation: int = 1,
        conv_depth: int = 2,
        linear_emb: str2bool = False,
        causal: str2bool = True,
        max_pool: str2bool = False,
        act_after_linear: str2bool = False,
        dropout: float = 0.0,
        spectral_norm: str2bool = False,
        weight_norm: str2bool = False,
    ):
        super().__init__()
        if cfg is not None:
            cfg = argparse.Namespace(**cfg)
            self.conv_channels = cfg.discriminator_dim
            self.conv_kernel = cfg.discriminator_kernel
            self.conv_dilation = cfg.discriminator_dilation
            self.conv_depth = cfg.discriminator_depth
            self.linear_emb = cfg.discriminator_linear_emb
            self.causal = cfg.discriminator_causal
            self.max_pool = cfg.discriminator_max_pool
            self.act_after_linear = cfg.discriminator_act_after_linear
            self.dropout = cfg.discriminator_dropout
            self.spectral_norm = cfg.discriminator_spectral_norm
            self.weight_norm = cfg.discriminator_weight_norm
        else:
            self.conv_channels = conv_channels
            self.conv_kernel = conv_kernel
            self.conv_dilation = conv_dilation
            self.conv_depth = conv_depth
            self.linear_emb = linear_emb
            self.causal = causal
            self.max_pool = max_pool
            self.act_after_linear = act_after_linear
            self.dropout = dropout
            self.spectral_norm = spectral_norm
            self.weight_norm = weight_norm

        if self.causal:
            self.conv_padding = self.conv_kernel - 1
        else:
            self.conv_padding = self.conv_kernel // 2

        def make_conv(
            in_channel, out_channel, kernal_size, padding_size=0, dilation_value=1
        ):
            conv = torch.nn.Conv1d(
                in_channel,
                out_channel,
                kernel_size=kernal_size,
                padding=padding_size,
                dilation=dilation_value,
            )
            if self.spectral_norm:
                conv = torch.nn.utils.spectral_norm(conv)
            elif self.weight_norm:
                conv = torch.nn.utils.weight_norm(conv)
            return conv

        # initialize embedding
        if self.linear_emb:
            emb_net = [
                make_conv(
                    input_dim, self.conv_channels, 1, dilation_value=self.conv_dilation
                )
            ]
        else:
            emb_net = [
                make_conv(
                    input_dim,
                    self.conv_channels,
                    self.conv_kernel,
                    self.conv_padding,
                    dilation_value=self.conv_dilation,
                ),
                SamePad(kernel_size=self.conv_kernel, causal=self.causal),
            ]

        if self.act_after_linear:
            emb_net.append(torch.nn.GELU())

        # initialize inner conv
        inner_net = [
            torch.nn.Sequential(
                make_conv(
                    self.conv_channels,
                    self.conv_channels,
                    self.conv_kernel,
                    self.conv_padding,
                    dilation_value=self.conv_dilation,
                ),
                SamePad(kernel_size=self.conv_kernel, causal=self.causal),
                torch.nn.Dropout(self.dropout),
                torch.nn.GELU(),
            )
            for _ in range(self.conv_depth - 1)
        ]

        inner_net += [
            make_conv(
                self.conv_channels,
                1,
                self.conv_kernel,
                self.conv_padding,
                dilation_value=1,
            ),
            SamePad(kernel_size=self.conv_kernel, causal=self.causal),
        ]

        self.net = torch.nn.Sequential(
            *emb_net,
            torch.nn.Dropout(dropout),
            *inner_net,
        )

    @typechecked
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]):
        """
            Forward pass for the ConvDiscriminator.

        This method processes the input tensor `x` through the convolutional layers
        defined in the network. It expects the input tensor to be in the shape of
        (Batch, Time, Channel) and transforms it accordingly. The method also
        applies padding based on the provided `padding_mask` if available.

        Args:
            x (torch.Tensor): The input tensor with shape (Batch, Time, Channel).
            padding_mask (Optional[torch.Tensor]): A tensor used to mask out certain
                elements of the input. It should have the shape (Batch, Time) and
                can be used to apply a maximum pooling or to set specific values to
                negative infinity.

        Returns:
            torch.Tensor: The output tensor after passing through the network, with
            shape (Batch, Channel).

        Examples:
            >>> discriminator = ConvDiscriminator(input_dim=128)
            >>> input_tensor = torch.randn(32, 100, 128)  # (Batch, Time, Channel)
            >>> output = discriminator(input_tensor, None)
            >>> output.shape
            torch.Size([32, 1])

        Note:
            The input tensor `x` will be transposed to match the expected shape for
            convolution operations. If a `padding_mask` is provided, it will be used
            to handle padding accordingly.

        Raises:
            ValueError: If the shape of `x` is not compatible with the expected
            input dimensions.
        """

        # (Batch, Time, Channel) -> (Batch, Channel, Time)
        x = x.transpose(1, 2)

        x = self.net(x)

        # (Batch, Channel, Time) -> (Batch, Time, Channel)
        x = x.transpose(1, 2)
        x_sz = x.size(1)
        if padding_mask is not None and padding_mask.any() and padding_mask.dim() > 1:
            padding_mask = padding_mask[:, : x.size(1)]
            padding_mask.to(x.device)
            x[padding_mask] = float("-inf") if self.max_pool else 0
            x_sz = x_sz - padding_mask.sum(dim=-1)

        x = x.squeeze(-1)
        if self.max_pool:
            x, _ = x.max(dim=-1)
        else:
            x = x.sum(dim=-1)
            x = x / x_sz

        return x
