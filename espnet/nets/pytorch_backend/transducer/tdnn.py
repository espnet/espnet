"""TDNN modules definition for transformer encoder."""

import torch
import torch.nn.functional as F


class TDNN(torch.nn.Module):
    """TDNN implementation based on Peddinti et al. implementation.

    Reference: https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

    This implementation exploits Conv1d's dilation argument.
    Thus, temporal context is defined as a value encapsulating following constraints:
        - Context must be symmetric
        - Spacing between each element of the context must be equal.

    Unfold usage is based on cvqluu's implementation (https://github.com/cvqluu/TDNN)

    Args:
        idim (int): dimension of inputs
        odim (int): dimension of outputs
        ctx_size (int): size of context window
        stride (int): stride of the sliding blocks
        dilation (int): parameter to control the stride of
                        elements within the neighborhood
        batch_norm (bool): whether to use batch normalization
        relu (bool): whether to use non-linearity layer (ReLU)

    """

    def __init__(
        self, idim, odim, ctx_size=5, dilation=1, stride=1, batch_norm=True, relu=True
    ):
        """Construct a TDNN object."""
        super(TDNN, self).__init__()

        self.idim = idim
        self.odim = odim

        self.ctx_size = ctx_size
        self.stride = stride
        self.dilation = dilation

        self.batch_norm = batch_norm
        self.relu = relu

        self.kernel = torch.nn.Linear((idim * ctx_size), odim)

        if self.batch_norm:
            self.bn = torch.nn.BatchNorm1d(odim)

        if self.relu:
            self.rel = torch.nn.ReLU()

    def forward(self, xs, masks):
        """Forward TDNN.

        Args:
            xs (torch.Tensor): input tensor (B, seq_len, idim)
            masks (torch.Tensor): input mask (B, 1, seq_len)

        Returns:
            xs (torch.Tensor): output tensor (B, new_seq_len, odim)
            masks (torch.Tensor): output mask (B, 1, new_seq_len)

        """
        xs = F.unfold(
            xs.unsqueeze(1),
            (self.ctx_size, self.idim),
            stride=(self.stride, self.idim),
            dilation=(self.dilation, 1),
        )

        xs = xs.transpose(1, 2)
        xs = self.kernel(xs)

        if self.batch_norm:
            xs = xs.transpose(1, 2)
            xs = self.bn(xs)
            xs = xs.transpose(1, 2)

        if self.relu:
            xs = self.rel(xs)

        if masks is not None:
            sub = (self.ctx_size - 1) * self.dilation

            if sub != 0:
                masks = masks[:, :, : -((self.ctx_size - 1) * self.dilation)]
            masks = masks[:, :, :: self.stride]

        return xs, masks
