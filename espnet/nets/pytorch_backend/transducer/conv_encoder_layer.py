"""N-D convolution module definition for custom encoder."""

from typing import Tuple
from typing import Union

import torch


class ConvEncoderLayer(torch.nn.Module):
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
        """Construct a ConvEncoderLayer module object."""
        super().__init__()

        if 0 <= conv_dim <= 2:
            conv_class = getattr(torch.nn, "Conv" + str(conv_dim) + "d")
        else:
            raise ValueError("ConvEncoderLayer only support 1D and 2D convolution.")

        self.is_2d = conv_dim == 2

        self.conv = conv_class(
            1 if self.is_2d else idim,
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
            kernel_size_t = kernel_size if type(kernel_size) is int else kernel_size[0]
            stride_t = stride if type(stride) is int else stride[0]
            dilation_t = dilation if type(dilation) is int else dilation[0]

            kernel_size_f = kernel_size if type(kernel_size) is int else kernel_size[1]
            stride_f = stride if type(stride) is int else stride[1]
            dilation_f = dilation if type(dilation) is int else dilation[1]

            self._pad = dilation_t * (kernel_size_t - 1)
            self.stride = stride_t

            f_odim = ((idim - dilation_f * (kernel_size_f - 1) - 1) // stride_f) + 1

            self.out = torch.nn.Linear((odim * f_odim), odim)
        else:
            self._pad = dilation * (kernel_size - 1)
            self.stride = stride

        self.out_pos = torch.nn.Linear(idim, odim)

    def forward(
        self,
        sequence: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Forward ConvEncoderLayer module object.

        Args:
            sequence: Input sequences.
                      (B, T, D_in) or ((B, T, D_in), (
            mask: Mask of input sequences. (B, 1, T)

        Returns:
            sequence: Output sequences. (B, sub(T), D_out)
            mask: Mask of output sequences. (B, 1, sub(T))

        """
        if isinstance(sequence, tuple):
            sequence, pos_embed = sequence[0], sequence[1]
        else:
            sequence, pos_embed = sequence, None

        if self.is_2d:
            sequence = sequence.unsqueeze(1)
        else:
            sequence = sequence.transpose(1, 2)

        if pos_embed is not None and pos_embed.size(1) == (2 * sequence.size(2)) - 1:
            bidir_pos_embed = True
        else:
            bidir_pos_embed = False

        sequence = self.conv(sequence)

        if self.batch_norm:
            sequence = self.bn(sequence)

        sequence = self.dropout(sequence)

        if self.relu:
            sequence = self.relu_func(sequence)

        sequence = sequence.transpose(1, 2)

        if self.is_2d:
            b, t, c, f = sequence.size()

            sequence = self.out(sequence.contiguous().view(b, t, (c * f)))

        mask = self.create_new_mask(mask)

        if pos_embed is not None:
            pos_embed = self.create_new_pos_embed(
                pos_embed,
                bidir_pos_embed,
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

        return self.out_pos(pos_embed)
