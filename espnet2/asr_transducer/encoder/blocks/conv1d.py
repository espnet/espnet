"""Conv1d block for Transducer encoder."""

from typing import Tuple
from typing import Union

import torch


class Conv1d(torch.nn.Module):
    """Conv1d module definition.

    Args:
        dim_input: Input dimension.
        dim_output: Output dimension.
        kernel_size: Size of the convolving kernel.
        mask_type: Type of mask for forward computation.
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
        dim_input: int,
        dim_output: int,
        kernel_size: Union[int, Tuple],
        mask_type: str,
        stride: Union[int, Tuple] = 1,
        dilation: Union[int, Tuple] = 1,
        groups: Union[int, Tuple] = 1,
        bias: bool = True,
        batch_norm: bool = False,
        relu: bool = True,
        dropout_rate: float = 0.0,
    ):
        """Construct a Conv1d object."""
        super().__init__()

        self.conv = torch.nn.Conv1d(
            dim_input,
            dim_output,
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
            self.bn = torch.nn.BatchNorm1d(dim_output)

        self.relu = relu
        self.batch_norm = batch_norm

        self.padding = dilation * (kernel_size - 1)
        self.stride = stride

        if mask_type == "rnn":
            self.create_new_mask = self.create_new_rnn_mask
        else:
            self.create_new_mask = self.create_new_conformer_mask

            self.out_pos = torch.nn.Linear(dim_input, dim_output)

    def forward(
        self,
        sequence: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        cache: torch.Tensor = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Encode input sequences.

        Args:
            sequence: Conv1d input sequences.
                      (B, T, D_in) or
                      ((B, T, D_in),  (B, 2 * (T - 1), D_att))
            mask: Mask of input sequences. (B, 1, T)

        Returns:
            sequence: Conv1d output sequences.
                      (B, sub(T), D_out) or
                      ((B, sub(T), D_out),  (B, 2 * (sub(T) - 1), D_att))
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

        if mask is not None:
            mask = self.create_new_mask(mask)

        if pos_embed is not None:
            sequence = (sequence, self.create_new_pos_embed(pos_embed))

        return sequence, mask

    def create_new_conformer_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create new conformer mask for output sequences.

        Args:
            mask: Mask of input sequences. (B, 1, T)

        Returns:
            mask: Mask of output sequences. (B, 1, sub(T))

        """
        if self.padding != 0:
            mask = mask[:, :, : -self.padding]

        return mask[:, :, :: self.stride]

    def create_new_rnn_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create new RNN mask for output sequences.

        Args:
            mask: Mask of input sequences. (B,)

        Returns:
            mask: Mask of output sequences. (B,)

        """
        if self.padding != 0:
            mask = mask - self.padding

        mask = mask // self.stride

        return mask

    def create_new_pos_embed(self, pos_embed: torch.Tensor) -> torch.Tensor:
        """Create new positional embedding vector.

        Args:
            pos_embed: Input sequences positional embedding.
                       (B, 2 * (T - 1), D_att)

        Returns:
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
