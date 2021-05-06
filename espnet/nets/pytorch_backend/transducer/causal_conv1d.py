"""CausalConv1d module definition for custom decoder."""

import torch


class CausalConv1d(torch.nn.Module):
    """CausalConv1d module for custom decoder.

    Args:
        idim (int): dimension of inputs
        odim (int): dimension of outputs
        kernel_size (int): size of convolving kernel
        stride (int): stride of the convolution
        dilation (int): spacing between the kernel points
        groups (int): number of blocked connections from ichannels to ochannels
        bias (bool): whether to add a learnable bias to the output

    """

    def __init__(
        self, idim, odim, kernel_size, stride=1, dilation=1, groups=1, bias=True
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

    def forward(self, x, x_mask, cache=None):
        """CausalConv1d forward for x.

        Args:
            x (torch.Tensor): input torch (B, U, idim)
            x_mask (torch.Tensor): (B, 1, U)

        Returns:
            x (torch.Tensor): input torch (B, sub(U), attention_dim)
            x_mask (torch.Tensor): (B, 1, sub(U))

        """
        x = x.permute(0, 2, 1)
        x = self.causal_conv1d(x)

        if self._pad != 0:
            x = x[:, :, : -self._pad]

        x = x.permute(0, 2, 1)

        return x, x_mask
