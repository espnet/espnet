"""CausalConv1d module definition for custom decoder."""

from typing import Optional
from typing import Tuple

import torch


class CausalConv1d(torch.nn.Module):
    """CausalConv1d module for custom decoder.

    Args:
        idim: Input dimension.
        odim: Output dimension.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        dilation: Spacing between the kernel points.
        groups: Number of blocked connections from input channels to output channels.
        bias: Whether to add a learnable bias to the output.

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
    ):
        """Construct a CausalConv1d object."""
        super().__init__()

        self._pad = (kernel_size - 1) * dilation

        self.causal_conv1d = torch.nn.Conv1d(
            idim,
            odim,
            kernel_size=kernel_size,
            stride=stride,
            padding=self._pad,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

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
        sequence = sequence.permute(0, 2, 1)
        sequence = self.causal_conv1d(sequence)

        if self._pad != 0:
            sequence = sequence[:, :, : -self._pad]

        sequence = sequence.permute(0, 2, 1)

        return sequence, mask
