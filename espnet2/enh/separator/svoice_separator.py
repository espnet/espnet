import math
from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.enh.layers.dpmulcat import DPMulCat
from espnet2.enh.layers.dprnn import merge_feature, split_feature
from espnet2.enh.separator.abs_separator import AbsSeparator


def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.

        Adds potentially overlapping frames of a signal with shape
        `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
        The resulting tensor has shape `[..., output_size]` where
            output_size = (frames - 1) * frame_step + frame_length

        Args:
            signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown,
                and rank must be at least 2.
            frame_step: An integer denoting overlap offsets.
                Must be less than or equal to frame_length.

        Returns:
            A Tensor with shape [..., output_size] containing the
                overlap-added frames of signal's inner-most two dimensions.
            output_size = (frames - 1) * frame_step + frame_length

        Based on

    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    # gcd=Greatest Common Divisor
    subframe_length = math.gcd(frame_length, frame_step)
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(
        0, subframes_per_frame, subframe_step
    )
    frame = frame.clone().detach().long().to(signal.device)
    # frame = signal.new_tensor(frame).clone().long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class Encoder(nn.Module):
    def __init__(self, enc_kernel_size: int, enc_feat_dim: int):
        super().__init__()
        # setting 50% overlap
        self.conv = nn.Conv1d(
            1,
            enc_feat_dim,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            bias=False,
        )
        self.nonlinear = nn.ReLU()

    def forward(self, mixture):
        mixture = torch.unsqueeze(mixture, 1)
        mixture_w = self.nonlinear(self.conv(mixture))
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, est_source):
        est_source = torch.transpose(est_source, 2, 3)
        est_source = nn.AvgPool2d((1, self.kernel_size))(est_source)
        est_source = overlap_and_add(est_source, self.kernel_size // 2)

        return est_source


class SVoiceSeparator(AbsSeparator):
    """SVoice model for speech separation.

    Reference:
        Voice Separation with an Unknown Number of Multiple Speakers;
        E. Nachmani et al., 2020;
        https://arxiv.org/abs/2003.01531

    Args:
        enc_dim: int, dimension of the encoder module's output. (Default: 128)
        kernel_size: int, the kernel size of Conv1D layer in both encoder and
            decoder modules. (Default: 8)
        hidden_size: int, dimension of the hidden state in RNN layers. (Default: 128)
        num_spk: int, the number of speakers in the output. (Default: 2)
        num_layers: int, number of stacked MulCat blocks. (Default: 4)
        segment_size: dual-path segment size. (Default: 20)
        bidirectional: bool, whether the RNN layers are bidirectional. (Default: True)
        input_normalize: bool, whether to apply GroupNorm on the input Tensor.
            (Default: False)
    """

    def __init__(
        self,
        input_dim: int,
        enc_dim: int,
        kernel_size: int,
        hidden_size: int,
        num_spk: int = 2,
        num_layers: int = 4,
        segment_size: int = 20,
        bidirectional: bool = True,
        input_normalize: bool = False,
    ):
        super().__init__()

        self._num_spk = num_spk
        self.enc_dim = enc_dim
        self.segment_size = segment_size
        # model sub-networks
        self.encoder = Encoder(kernel_size, enc_dim)
        self.decoder = Decoder(kernel_size)
        self.rnn_model = DPMulCat(
            input_size=enc_dim,
            hidden_size=hidden_size,
            output_size=enc_dim,
            num_spk=num_spk,
            num_layers=num_layers,
            bidirectional=bidirectional,
            input_normalize=input_normalize,
        )

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """
        # fix time dimension, might change due to convolution operations
        T_mix = input.size(-1)

        mixture_w = self.encoder(input)

        enc_segments, enc_rest = split_feature(mixture_w, self.segment_size)
        # separate
        output_all = self.rnn_model(enc_segments)

        # generate wav after each RNN block and optimize the loss
        outputs = []
        for ii in range(len(output_all)):
            output_ii = merge_feature(output_all[ii], enc_rest)
            output_ii = output_ii.view(
                input.shape[0], self._num_spk, self.enc_dim, mixture_w.shape[2]
            )
            output_ii = self.decoder(output_ii)
            T_est = output_ii.size(-1)
            output_ii = F.pad(output_ii, (0, T_mix - T_est))
            output_ii = list(output_ii.unbind(dim=1))
            if self.training:
                outputs.append(output_ii)
            else:
                outputs = output_ii

        others = {}
        return outputs, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
