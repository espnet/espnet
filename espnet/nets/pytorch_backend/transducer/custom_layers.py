"""Convolution-related modules definition for transformer-transducer."""

import torch
import torch.nn.functional as F


class VGG2L(torch.nn.Module):
    """VGG2L module for transformer-transducer encoder."""

    def __init__(self, idim, odim):
        """Construct a VGG2L object.

        Args:
            idim (int): dimension of inputs
            odim (int): dimension of outputs

        """
        super(VGG2L, self).__init__()

        self.vgg2l = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((3, 2)),
            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
        )

        self.output = torch.nn.Linear(128 * ((idim // 2) // 2), odim)

    def forward(self, x, x_mask):
        """VGG2L forward for x.

        Args:
            x (torch.Tensor): input torch (B, T, idim)
            x_mask (torch.Tensor): (B, 1, T)

        Returns:
            x (torch.Tensor): input torch (B, sub(T), attention_dim)
            x_mask (torch.Tensor): (B, 1, sub(T))

        """
        x = x.unsqueeze(1)
        x = self.vgg2l(x)

        b, c, t, f = x.size()

        x = self.output(x.transpose(1, 2).contiguous().view(b, t, c * f))

        if x_mask is None:
            return x, None
        else:
            x_mask = self.create_new_mask(x_mask, x)

            return x, x_mask

    def create_new_mask(self, x_mask, x):
        """Create a subsampled version of x_mask.

        Args:
            x_mask (torch.Tensor): (B, 1, T)
            x (torch.Tensor): (B, sub(T), attention_dim)

        Returns:
            x_mask (torch.Tensor): (B, 1, sub(T))

        """
        x_t1 = x_mask.size(2) - (x_mask.size(2) % 3)
        x_mask = x_mask[:, :, :x_t1][:, :, ::3]

        x_t2 = x_mask.size(2) - (x_mask.size(2) % 2)
        x_mask = x_mask[:, :, :x_t2][:, :, ::2]

        return x_mask


class CausalConv1d(torch.nn.Module):
    """CausalConv1d module for transformer-transducer encoder."""

    def __init__(
        self, idim, odim, kernel_size, stride=1, dilation=1, groups=1, bias=True
    ):
        """Construct a CausalConv1d object.

        Args:
            idim (int): dimension of inputs
            odim (int): dimension of outputs
            kernel_size (int): size of convolving kernel
            stride (int): stride of the convolution
            dilation (int): spacing between the kernel points
            groups (int): number of blocked connections from ichannels to ochannels
            bias (bool): whether to add a learnable bias to the output

        """
        super(CausalConv1d, self).__init__()

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
