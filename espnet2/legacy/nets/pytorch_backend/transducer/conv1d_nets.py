"""Convolution networks definition for custom archictecture."""

from typing import Optional, Tuple, Union

import torch


class Conv1d(torch.nn.Module):
    """1D convolution module for custom encoder.

    Args:
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
        idim: int,
        odim: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        dilation: Union[int, Tuple] = 1,
        groups: Union[int, Tuple] = 1,
        bias: bool = True,
        batch_norm: bool = False,
        relu: bool = True,
        dropout_rate: float = 0.0,
    ):
        """Construct a Conv1d module object."""
        super().__init__()

        self.conv = torch.nn.Conv1d(
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
            self.bn = torch.nn.BatchNorm1d(odim)

        self.relu = relu
        self.batch_norm = batch_norm

        self.padding = dilation * (kernel_size - 1)
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
                      (B, T, D_in)
                        or (B, T, D_in),  (B, 2 * (T - 1), D_att)
            mask: Mask of input sequences. (B, 1, T)

        Returns:
            sequence: Output sequences.
                      (B, sub(T), D_out)
                        or (B, sub(T), D_out),  (B, 2 * (sub(T) - 1), D_att)
            mask: Mask of output sequences. (B, 1, sub(T))

        """
        if isinstance(sequence, tuple):
            sequence, pos_embed = sequence[0], sequence[1]
        else:
            sequence, pos_embed = sequence, None

        sequence = sequence.transpose(1, 2)
        sequence = self.conv(sequence)

        if self.batch_norm:
            sequence = self.bn(sequence)

        sequence = self.dropout(sequence)

        if self.relu:
            sequence = self.relu_func(sequence)

        sequence = sequence.transpose(1, 2)

        mask = self.create_new_mask(mask)

        if pos_embed is not None:
            pos_embed = self.create_new_pos_embed(pos_embed)

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

        if self.padding != 0:
            mask = mask[:, :, : -self.padding]

        mask = mask[:, :, :: self.stride]

        return mask

    def create_new_pos_embed(self, pos_embed: torch.Tensor) -> torch.Tensor:
        """Create new positional embedding vector.

        Args:
            pos_embed: Input sequences positional embedding.
                       (B, 2 * (T - 1), D_att)

        Return:
            pos_embed: Output sequences positional embedding.
                       (B, 2 * (sub(T) - 1), D_att)

        """
        pos_embed_positive = pos_embed[:, : pos_embed.size(1) // 2 + 1, :]
        pos_embed_negative = pos_embed[:, pos_embed.size(1) // 2 :, :]

        if self.padding != 0:
            pos_embed_positive = pos_embed_positive[:, : -self.padding, :]
            pos_embed_negative = pos_embed_negative[:, : -self.padding, :]

        pos_embed_positive = pos_embed_positive[:, :: self.stride, :]
        pos_embed_negative = pos_embed_negative[:, :: self.stride, :]

        pos_embed = torch.cat([pos_embed_positive, pos_embed_negative[:, 1:, :]], dim=1)

        return self.out_pos(pos_embed)


class CausalConv1d(torch.nn.Module):
    """1D causal convolution module for custom decoder.

    Args:
        idim: Input dimension.
        odim: Output dimension.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        dilation: Spacing between the kernel points.
        groups: Number of blocked connections from input channels to output channels.
        bias: Whether to add a learnable bias to the output.
        batch_norm: Whether to apply batch normalization.
        relu: Whether to pass final output through ReLU activation.
        dropout_rate: Dropout rate.

    """

    def __init__(
        self,
        idim: int,
        odim: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        batch_norm: bool = False,
        relu: bool = True,
        dropout_rate: float = 0.0,
    ):
        """Construct a CausalConv1d object."""
        super().__init__()

        self.padding = (kernel_size - 1) * dilation

        self.causal_conv1d = torch.nn.Conv1d(
            idim,
            odim,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.dropout = torch.nn.Dropout(p=dropout_rate)

        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(odim)

        if relu:
            self.relu_func = torch.nn.ReLU()

        self.batch_norm = batch_norm
        self.relu = relu

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward CausalConv1d for custom decoder.

        Args:
            sequence: CausalConv1d input sequences. (B, U, D_in)
            mask: Mask of CausalConv1d input sequences. (B, 1, U)


        Returns:
            sequence: CausalConv1d output sequences. (B, sub(U), D_out)
            mask: Mask of CausalConv1d output sequences. (B, 1, sub(U))

        """
        sequence = sequence.transpose(1, 2)
        sequence = self.causal_conv1d(sequence)

        if self.padding != 0:
            sequence = sequence[:, :, : -self.padding]

        if self.batch_norm:
            sequence = self.bn(sequence)

        sequence = self.dropout(sequence)

        if self.relu:
            sequence = self.relu_func(sequence)

        sequence = sequence.transpose(1, 2)

        return sequence, mask
