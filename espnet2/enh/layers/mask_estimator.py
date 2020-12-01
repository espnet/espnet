from typing import Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet.nets.pytorch_backend.rnn.encoders import RNNP


class MaskEstimator(torch.nn.Module):
    def __init__(
        self, type, idim, layers, units, projs, dropout, nmask=1, nonlinear="sigmoid"
    ):
        super().__init__()
        subsample = np.ones(layers + 1, dtype=np.int)

        typ = type.lstrip("vgg").rstrip("p")
        if type[-1] == "p":
            self.brnn = RNNP(idim, layers, units, projs, subsample, dropout, typ=typ)
        else:
            self.brnn = RNN(idim, layers, units, projs, dropout, typ=typ)

        self.type = type
        self.nmask = nmask
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(projs, idim) for _ in range(nmask)]
        )

        if nonlinear not in ("sigmoid", "relu", "tanh", "crelu"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = nonlinear

    def forward(
        self, xs: ComplexTensor, ilens: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.LongTensor]:
        """Mask estimator forward function.

        Args:
            xs: (B, F, C, T)
            ilens: (B,)
        Returns:
            hs (torch.Tensor): The hidden vector (B, F, C, T)
            masks: A tuple of the masks. (B, F, C, T)
            ilens: (B,)
        """
        assert xs.size(0) == ilens.size(0), (xs.size(0), ilens.size(0))
        _, _, C, input_length = xs.size()
        # (B, F, C, T) -> (B, C, T, F)
        xs = xs.permute(0, 2, 3, 1)

        # Calculate amplitude: (B, C, T, F) -> (B, C, T, F)
        xs = (xs.real ** 2 + xs.imag ** 2) ** 0.5
        # xs: (B, C, T, F) -> xs: (B * C, T, F)
        xs = xs.contiguous().view(-1, xs.size(-2), xs.size(-1))
        # ilens: (B,) -> ilens_: (B * C)
        ilens_ = ilens[:, None].expand(-1, C).contiguous().view(-1)

        # xs: (B * C, T, F) -> xs: (B * C, T, D)
        xs, _, _ = self.brnn(xs, ilens_)
        # xs: (B * C, T, D) -> xs: (B, C, T, D)
        xs = xs.view(-1, C, xs.size(-2), xs.size(-1))

        masks = []
        for linear in self.linears:
            # xs: (B, C, T, D) -> mask:(B, C, T, F)
            mask = linear(xs)

            if self.nonlinear == "sigmoid":
                mask = torch.sigmoid(mask)
            elif self.nonlinear == "relu":
                mask = torch.relu(mask)
            elif self.nonlinear == "tanh":
                mask = torch.tanh(mask)
            elif self.nonlinear == "crelu":
                mask = torch.clamp(mask, min=0, max=1)
            # Zero padding
            mask.masked_fill(make_pad_mask(ilens, mask, length_dim=2), 0)

            # (B, C, T, F) -> (B, F, C, T)
            mask = mask.permute(0, 3, 1, 2)

            # Take cares of multi gpu cases: If input_length > max(ilens)
            if mask.size(-1) < input_length:
                mask = F.pad(mask, [0, input_length - mask.size(-1)], value=0)
            masks.append(mask)

        return tuple(masks), ilens


##########################################
# Below are for TCN-based mask estimator #
##########################################
# modified from https://gitlab.uni-oldenburg.de/hura4843/deep-mfmvdr/-/blob/master/deep_mfmvdr/building_blocks.py (# noqa: E501)
class cLN(torch.nn.Module):
    """Cumulative layer normalization."""

    def __init__(self, dimension, eps=1e-8, trainable=True):
        super().__init__()

        self.eps = eps
        self.gain = torch.nn.Parameter(
            torch.ones((1, dimension, 1), requires_grad=trainable)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros((1, dimension, 1), requires_grad=trainable)
        )

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step

        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = torch.arange(
            channel,
            channel * (time_step + 1),
            channel,
            dtype=input.dtype,
            device=input.device,
        )
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(
            2
        )  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(
            x.type()
        )


class DepthConv1d(torch.nn.Module):
    """Depthwise separable convolution."""

    def __init__(
        self,
        input_channel,
        hidden_channel,
        kernel,
        padding,
        dilation=1,
        skip=True,
        causal=False,
        no_residual=False,
    ):
        super().__init__()

        self.causal = causal
        # for the last layer, residual will not be used in TCN when self.skip=True
        # This flag is for avoiding unused parameters in the model.
        self.no_residual = no_residual
        self.skip = skip

        self.conv1d = torch.nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv1d = torch.nn.Conv1d(
            hidden_channel,
            hidden_channel,
            kernel,
            dilation=dilation,
            groups=hidden_channel,
            padding=self.padding,
        )
        if not self.skip or not self.no_residual:
            self.res_out = torch.nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = torch.nn.PReLU()
        self.nonlinearity2 = torch.nn.PReLU()
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:
            self.reg1 = torch.nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = torch.nn.GroupNorm(1, hidden_channel, eps=1e-08)

        if self.skip:
            self.skip_out = torch.nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        if self.causal:
            output = self.reg2(
                self.nonlinearity2(self.dconv1d(output)[:, :, : -self.padding])
            )
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))

        if self.skip:
            skip = self.skip_out(output)
            if self.no_residual:
                return skip
            else:
                residual = self.res_out(output)
                return residual, skip
        else:
            residual = self.res_out(output)
            return residual


class TCNEstimator(torch.nn.Module):
    """Small modification of TCN for spectrum-based parameter estimation
    based on TCN implementation in https://github.com/naplab/Conv-TasNet.
    """  # noqa: H405, D205, D400

    def __init__(
        self,
        input_dim,
        output_dim,
        BN_dim,
        hidden_dim,
        layer=8,
        stack=3,
        kernel=3,
        skip=True,
        causal=True,
        dilated=True,
    ):
        super().__init__()

        # input is a sequence of features of shape (B, N, L)
        # here, N/2 refers to the number of frequencies, stacked real and imaginary components (# noqa: E501)

        # normalization
        if causal:
            self.LN = cLN(input_dim, eps=1e-8)
        else:
            self.LN = torch.nn.GroupNorm(1, input_dim, eps=1e-8)

        self.BN = torch.nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated

        self.TCN = torch.nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                no_residual = s == stack - 1 and i == layer - 1
                if self.dilated:
                    self.TCN.append(
                        DepthConv1d(
                            BN_dim,
                            hidden_dim,
                            kernel,
                            dilation=2 ** i,
                            padding=2 ** i,
                            skip=skip,
                            causal=causal,
                            no_residual=no_residual,
                        )
                    )
                else:
                    self.TCN.append(
                        DepthConv1d(
                            BN_dim,
                            hidden_dim,
                            kernel,
                            dilation=1,
                            padding=1,
                            skip=skip,
                            causal=causal,
                            no_residual=no_residual,
                        )
                    )
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2 ** i
                    else:
                        self.receptive_field += kernel - 1

        self.output = torch.nn.Conv1d(BN_dim, output_dim, 1)

        self.skip = skip

    def forward(self, input):
        # input shape: (B, N, L)
        # normalization
        output = self.BN(self.LN(input))

        # pass to TCN
        length = len(self.TCN)
        if self.skip:
            skip_connection = 0.0
            for i in range(length):
                if i == length - 1:
                    skip = self.TCN[i](output)
                else:
                    residual, skip = self.TCN[i](output)
                    output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(length):
                residual = self.TCN[i](output)
                output = output + residual

        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)

        return output
