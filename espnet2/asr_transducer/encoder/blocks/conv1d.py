"""Conv1d block for Transducer encoder."""

from typing import Optional, Tuple, Union

import torch


class Conv1d(torch.nn.Module):
    """Conv1d module definition.

    Args:
        input_size: Input dimension.
        output_size: Output dimension.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        dilation: Spacing between the kernel points.
        groups: Number of blocked connections from input channels to output channels.
        bias: Whether to add a learnable bias to the output.
        batch_norm: Whether to use batch normalization after convolution.
        relu: Whether to use a ReLU activation after convolution.
        causal: Whether to use causal convolution.
        dropout_rate: Dropout rate.

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        dilation: Union[int, Tuple] = 1,
        groups: Union[int, Tuple] = 1,
        bias: bool = True,
        batch_norm: bool = False,
        relu: bool = True,
        causal: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        """Construct a Conv1d object."""
        super().__init__()

        if causal:
            self.lorder = kernel_size - 1
            padding = 0
        else:
            self.lorder = 0
            padding = dilation * (kernel_size - 1)
        # padding = dilation * (kernel_size - 1)

        self.conv = torch.nn.Conv1d(
            input_size,
            output_size,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.dropout = torch.nn.Dropout(p=dropout_rate)

        if relu:
            self.relu_func = torch.nn.ReLU()

        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(output_size)

        self.relu = relu
        self.batch_norm = batch_norm

        self.causal = causal

        self.padding = padding
        self.stride = stride

        self.out_pos = torch.nn.Linear(input_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequences.

        Args:
            sequence: Conv1d input sequences. (B, T, D_in)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_att)

        Returns:
            x: Conv1d output sequences. (B, sub(T), D_out)

        """
        x = x.transpose(1, 2)

        x = self.conv(x)

        if self.batch_norm:
            x = self.bn(x)

        x = self.dropout(x)

        if self.relu:
            x = self.relu_func(x)

        x = x.transpose(1, 2)

        if self.causal:
            return x, pos_enc, mask

        return x

    def create_new_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create new mask for output sequences.

        Args:
            mask: Mask of input sequences. (B, 1, T)

        Returns:
            mask: Mask of output sequences. (B, 1, sub(T))

        """
        if self.padding != 0:
            mask = mask[:, :, : -self.padding]

        return mask[:, :, :: self.stride]

    def create_new_pos_enc(self, pos_enc: torch.Tensor) -> torch.Tensor:
        """Create new positional embedding vector.

        Args:
            pos_enc: Input sequences positional embedding.
                     (B, 2 * (T - 1), D_att)

        Returns:
            pos_enc: Output sequences positional embedding.
                     (B, 2 * (sub(T) - 1), D_att)

        """
        pos_enc_positive = pos_enc[:, : pos_enc.size(1) // 2 + 1, :]
        pos_enc_negative = pos_enc[:, pos_enc.size(1) // 2 :, :]

        if self.padding != 0:
            pos_enc_positive = pos_enc_positive[:, : -self.padding, :]
            pos_enc_negative = pos_enc_negative[:, : -self.padding, :]

        pos_enc_positive = pos_enc_positive[:, :: self.stride, :]
        pos_enc_negative = pos_enc_negative[:, :: self.stride, :]

        pos_enc = torch.cat([pos_enc_positive, pos_enc_negative[:, 1:, :]], dim=1)

        return self.out_pos(pos_enc)
