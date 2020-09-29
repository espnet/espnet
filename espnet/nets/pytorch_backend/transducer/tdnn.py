"""TDNN modules definition for transformer encoder."""

import torch
import torch.nn.functional as F


class TDNN(torch.nn.Module):
    """TDNN implementation based on Peddinti et al. implementation.

    Reference: https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

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
        self,
        idim,
        odim,
        ctx_size=5,
        dilation=1,
        stride=1,
        batch_norm=True,
        relu=True,
        dropout_rate=0.0,
    ):
        """Construct a TDNN object."""
        super().__init__()

        self.idim = idim
        self.odim = odim

        self.ctx_size = ctx_size
        self.stride = stride
        self.dilation = dilation

        self.batch_norm = batch_norm
        self.relu = relu

        self.tdnn = torch.nn.Conv1d(
            idim, odim, ctx_size, stride=stride, dilation=dilation
        )

        if self.batch_norm:
            self.bn = torch.nn.BatchNorm1d(odim)

        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, xs, masks):
        """Forward TDNN.

        Args:
            xs (torch.Tensor): input tensor (B, seq_len, idim)
            masks (torch.Tensor): input mask (B, 1, seq_len)

        Returns:
            xs (torch.Tensor): output tensor (B, new_seq_len, odim)
            masks (torch.Tensor): output mask (B, 1, new_seq_len)

        """
        xs = xs.transpose(1, 2).contiguous()
        xs = self.tdnn(xs)

        if self.batch_norm:
            xs = self.bn(xs)

        if self.relu:
            xs = F.relu(xs)

        xs = self.dropout(xs.transpose(1, 2).contiguous())

        if masks is not None:
            sub = (self.ctx_size - 1) * self.dilation

            if sub != 0:
                masks = masks[:, :, :-sub]

            masks = masks[:, :, :: self.stride]

        return xs, masks
