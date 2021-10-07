"""N-d Convolution module definition for custom architecture."""

from typing import Tuple
from typing import Union

import torch


class ConvNd(torch.nn.Module):
    """N-D convolution module for custom encoder.

    Args:
        conv_dim: Dimension of the convolution.
        idim: Input dimension.
        odim: Output dimension.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        dilation: Spacing between the kernel points.
        groups: Number of blocked connections from input channels to output channels.
        bias: Whether to add a learnable bias to the output.
        batch_norm: Whether to use batch normalization after convolution.
        relu: Whether to use a ReLU activation after convolution.
        dropout_rate: Dropout rate.

    """

    def __init__(
        self,
        conv_dim: int,
        idim: int,
        odim: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        dilation: [int, Tuple] = 1,
        groups: [int, Tuple] = 1,
        bias: bool = True,
        batch_norm: bool = False,
        relu: bool = True,
        dropout_rate: float = 0.0,
    ):
        """Construct a N-d convolution module object."""
        super().__init__()

        if 0 <= conv_dim <= 2:
            conv_class = getattr(torch.nn, "Conv" + str(conv_dim) + "d")
        else:
            raise ValueError("ConvNd only support 1D and 2D convolution.")

        self.conv = conv_class(
            idim,
            odim,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.dropout = torch.nn.Dropout(p=dropout_rate)

        if relu:
            self.relu_func = torch.nn.ReLU()

        if batch_norm:
            self.bn = getattr(torch.nn, "BatchNorm" + str(conv_dim) + "d")(odim)

        self.relu = relu
        self.batch_norm = batch_norm

        if conv_dim == 2:
            kernel_size_t = kernel_size if type(kernel_size) is int else kernel_size[1]
            stride_t = stride if type(stride) is int else stride[1]
            dilation_t = dilation if type(dilation) is int else dilation[1]

            self._pad = (kernel_size_t - 1) * dilation_t
            self.stride = stride_t

            self.out = torch.nn.Linear(
                odim
                * (
                    (kernel_size if type(kernel_size) is int else kernel_size[0] - 1)
                    * dilation
                    if type(dilation) is int
                    else dilation[0]
                ),
                odim,
            )
        else:
            self._pad = (kernel_size - 1) * dilation
            self.stride = stride

        self.conv_dim = conv_dim

    def forward(
        self,
        sequence: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Forward N-d convolution module object.

        Args:
            sequence: Input sequences. (B, T, D_in)
            mask: Mask of input sequences. (B, 1, T)

        Returns:
            sequence: Output sequences. (B, sub(T), D_out)
            mask: Mask of output sequences. (B, 1, sub(T))

        """
        if isinstance(sequence, tuple):
            sequence, pos_embed = sequence[0], sequence[1]
        else:
            sequence, pos_embed = sequence, None

        sequence = sequence.transpose(1, 2)

        if self.conv_dim > 1:
            sequence = sequence.unsqueeze(2)
            b, c, h, t = sequence.size()

        sequence = self.conv(sequence)

        if self.batch_norm:
            sequence = self.bn(sequence)

        sequence = self.dropout(sequence)

        if self.relu:
            sequence = self.relu_func(sequence)

        if self.conv_dim > 1:
            sequence = self.out(
                sequence.transpose(1, 3).contiguous().view(b, t, (h * c))
            )
        else:
            sequence = sequence.transpose(1, 2)

        mask = self.create_new_mask(mask)

        if pos_embed is not None:
            pos_embed = self.create_new_pos_embed(
                pos_embed, pos_embed.size(1) == (2 * sequence.size(2)) - 1
            )

            return (sequence, pos_embed), mask

        return sequence, mask

    def create_new_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create new mask.

        Args:
            mask: Mask of input sequences. (B, 1, T)

        Returns:
            mask: Mask of output sequences. (B, 1, sub(T))

        """
        if mask is None:
            return mask

        if self._pad != 0:
            mask = mask[:, :, : -self._pad]

        mask = mask[:, :, :: self.stride]

        return mask

    def create_new_pos_embed(
        self, pos_embed: torch.Tensor, is_bidir: bool
    ) -> torch.Tensor:
        """Create new positional embedding vector.

        Args:
            pos_embed: Input sequences positional embedding.
                       (B, T, D_att) or (B, 2 * (T - 1), D_att)
            is_bidir: Whether positional embedding is bidirectional.

        Return:
            pos_embed: Output sequences positional embedding.
                       (B, sub(T), D_att) or (B, 2 * (sub(T) - 1), D_att)

        """
        if is_bidir:
            pos_embed_positive = pos_embed[:, : pos_embed.size(1) // 2 + 1, :]
            pos_embed_negative = pos_embed[:, pos_embed.size(1) // 2 :, :]

            if self._pad != 0:
                pos_embed_positive = pos_embed_positive[:, : -self._pad, :]
                pos_embed_negative = pos_embed_negative[:, : -self._pad, :]

            pos_embed_positive = pos_embed_positive[:, :: self.stride, :]
            pos_embed_negative = pos_embed_negative[:, :: self.stride, :]

            pos_embed = torch.cat(
                [pos_embed_positive, pos_embed_negative[:, 1:, :]], dim=1
            )
        else:
            if self._pad != 0:
                pos_embed = pos_embed[:, : -self._pad, :]

            pos_embed = pos_embed[:, :: self.stride, :]

        return pos_embed
