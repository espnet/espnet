from collections import OrderedDict
import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from espnet2.enh.abs_enh import AbsEnhancement

EPS = torch.finfo(torch.get_default_dtype()).eps


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

    Based on https://github.com/tensorflow/tensorflow/blob/
    r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    # subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)
    subframe_signal = signal.reshape(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(
        0, subframes_per_frame, subframe_step
    )
    frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


def remove_pad(inputs, inputs_lengths):
    """Remove the padding part for all inputs.

    Args:
        inputs: torch.Tensor, [B, C, T] or [B, T], B is batch size
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.dim()
    if dim == 3:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 3:  # [B, C, T]
            results.append(input[:, :length].view(C, -1).cpu().numpy())
        elif dim == 2:  # [B, T]
            results.append(input[:length].view(-1).cpu().numpy())
    return results


def choose_norm(norm_type, channel_size):
    """The input of normalization will be (M, C, K).

    M: batch size
    C: channel size
    K: sequence length
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    elif norm_type == "GroupNorm":
        return nn.GroupNorm(1, channel_size, eps=1e-8)
    elif norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)
    else:
        raise ValueError("Unsupported normalization type")


# TODO(Jing): Use nn.LayerNorm to impl cLN to speed up
class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)."""

    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """Channel-wise layer normalization forwad.

        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)."""

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """Global layer normalization forward.

        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO(Jing): in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (
            (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        )
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer."""

    def __init__(self, W=2, N=64):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.W, self.N = W, N
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=W, stride=W // 2, bias=False)

    def forward(self, mixture):
        """Encoder forward.

        Args:
            mixture: [B, T], B is batch size, T is #samples
        Returns:
            mixture_w: [B, N, L], where L = (T-W)/(W/2)+1 = 2T/W-1
            L is the number of time steps
        """
        mixture = torch.unsqueeze(mixture, 1)  # [B, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [B, N, L]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, E, W):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.E, self.W = E, W
        # Components
        self.basis_signals = nn.Linear(E, W, bias=False)

    def forward(self, mixture_w, est_mask):
        """Decoder forward.

        Args:
            mixture_w: [B, E, L]
            est_mask: [B, C, E, L]
        Returns:
            est_source: [B, C, T]
        """
        # D = W * M
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # [B, C, E, L]
        source_w = torch.transpose(source_w, 2, 3)  # [B, C, L, E]
        # S = DV
        est_source = self.basis_signals(source_w)  # [B, C, L, W]
        est_source = overlap_and_add(est_source, self.W // 2)  # B x C x T
        return est_source


class SingleRNN(nn.Module):
    """Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(
        self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False
    ):
        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            1,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        # input = input.to(device)
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(
            rnn_output.contiguous().view(-1, rnn_output.shape[2])
        ).view(output.shape)
        return rnn_output


# dual-path RNN
class DPRNN(nn.Module):
    """Deep dual-path RNN.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        output_size,
        dropout=0,
        num_layers=1,
        bidirectional=True,
    ):
        super(DPRNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # dual-path RNN
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_rnn.append(
                SingleRNN(
                    rnn_type, input_size, hidden_size, dropout, bidirectional=True
                )
            )  # intra-segment RNN is always noncausal
            self.col_rnn.append(
                SingleRNN(
                    rnn_type,
                    input_size,
                    hidden_size,
                    dropout,
                    bidirectional=bidirectional,
                )
            )
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            # default is to use noncausal LayerNorm for inter-chunk RNN.
            # For causal setting change it to causal normalization accordingly.
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size, 1))

    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        # input = input.to(device)
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            row_input = (
                output.permute(0, 3, 2, 1)
                .contiguous()
                .view(batch_size * dim2, dim1, -1)
            )  # B*dim2, dim1, N
            row_output = self.row_rnn[i](row_input)  # B*dim2, dim1, H
            row_output = (
                row_output.view(batch_size, dim2, dim1, -1)
                .permute(0, 3, 2, 1)
                .contiguous()
            )  # B, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output

            col_input = (
                output.permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size * dim1, dim2, -1)
            )  # B*dim1, dim2, N
            col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, H
            col_output = (
                col_output.view(batch_size, dim1, dim2, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )  # B, N, dim1, dim2
            col_output = self.col_norm[i](col_output)
            output = output + col_output

        output = self.output(output)  # B, output_size, dim1, dim2

        return output


# base module for deep DPRNN
class DPRNN_base(nn.Module):
    def __init__(
        self,
        input_dim,
        feature_dim,
        hidden_dim,
        num_spk=1,
        layer=4,
        segment_size=100,
        bidirectional=True,
        rnn_type="LSTM",
    ):
        super(DPRNN_base, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk

        self.eps = 1e-8

        # bottleneck
        self.BN = nn.Conv1d(
            self.input_dim, self.feature_dim, 1, bias=False
        )  # raw DPRNN: 64-->64, Asteroid DPRNN: 64-->128

        # DPRNN model
        self.DPRNN = DPRNN(
            rnn_type,
            self.feature_dim,
            self.hidden_dim,
            self.feature_dim * self.num_spk,
            num_layers=layer,
            bidirectional=bidirectional,
        )

    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(
            input.type()
        )
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = (
            input[:, :, :-segment_stride]
            .contiguous()
            .view(batch_size, dim, -1, segment_size)
        )
        segments2 = (
            input[:, :, segment_stride:]
            .contiguous()
            .view(batch_size, dim, -1, segment_size)
        )
        segments = (
            torch.cat([segments1, segments2], 3)
            .view(batch_size, dim, -1, segment_size)
            .transpose(2, 3)
        )

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = (
            input.transpose(2, 3)
            .contiguous()
            .view(batch_size, dim, -1, segment_size * 2)
        )  # B, N, K, L

        input1 = (
            input[:, :, :, :segment_size]
            .contiguous()
            .view(batch_size, dim, -1)[:, :, segment_stride:]
        )
        input2 = (
            input[:, :, :, segment_size:]
            .contiguous()
            .view(batch_size, dim, -1)[:, :, :-segment_stride]
        )

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()  # B, N, T

    def forward(self, input):
        pass


# pure dprnn for single separation
class SEP_module(DPRNN_base):
    def __init__(self, *args, **kwargs):
        super(SEP_module, self).__init__(*args, **kwargs)

        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, 1), nn.Sigmoid()
        )

    def forward(self, input, voiceP=None):
        # input = input.to(device)
        # input: (B, E, T)
        batch_size, E, seq_length = input.shape

        enc_feature = self.BN(input)  # (B, E, L)-->(B, N, L)
        # N = enc_feature.shape[1]

        # split the encoder output into overlapped, longer segments
        enc_segments, enc_rest = self.split_feature(
            enc_feature, self.segment_size
        )  # B, N, L, K: L is the segment_size
        # pass to DPRNN
        output = self.DPRNN(enc_segments)  # B, topk*N, L, K
        output = output.view(
            batch_size * self.num_spk, self.feature_dim, self.segment_size, -1
        )  # B*topk, N, L, K

        # overlap-and-add of the outputs
        output = self.merge_feature(output, enc_rest)  # B*topk, N, L
        # output -- > B,topk,N,L
        # output = output.view(batch_size, self.num_spk,self.feature_dim,seq_length)

        # END cat mode
        # gated output layer for filter generation
        bf_filter = self.output(output) * self.output_gate(output)  # B*nspk, K, L
        # self.feature_dim)  # B, nspk, L, N
        return bf_filter


# DPRNN for beamforming filter estimation
class BF_module(DPRNN_base):
    def __init__(self, *args, **kwargs):
        super(BF_module, self).__init__(*args, **kwargs)

        # gated output layer
        # self.output = nn.Sequential(
        #     nn.Conv1d(self.feature_dim, self.feature_dim, 1), nn.Tanh() )
        # self.output_gate = nn.Sequential(
        #     nn.Conv1d(self.feature_dim, self.feature_dim, 1), nn.Sigmoid() )
        self.output = nn.Sequential(
            nn.Conv1d(self.feature_dim + 512, self.feature_dim, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(self.feature_dim + 512, self.feature_dim, 1), nn.Sigmoid()
        )

    def forward(self, input, voiceP):
        # input = input.to(device)
        # input: (B, E, T)
        # voiceP: (BS,topk,D)
        batch_size, E, seq_length = input.shape
        _, topk, D = voiceP.size()
        assert batch_size == _
        self.num_spk = topk

        enc_feature = self.BN(input)  # (B, E, L)-->(B, N, L)
        N = enc_feature.shape[1]
        # split the encoder output into overlapped, longer segments
        enc_segments, enc_rest = self.split_feature(
            enc_feature, self.segment_size
        )  # B, N, L, K: L is the segment_size
        # pass to DPRNN
        output = self.DPRNN(enc_segments)
        output = output.view(
            batch_size, self.feature_dim, self.segment_size, -1
        )  # B, N, L, K

        # print('output shape',output.shape)
        # overlap-and-add of the outputs
        output = self.merge_feature(output, enc_rest)  # B, N, L
        output = output.unsqueeze(1).expand(-1, topk, -1, -1)  # B,topk,N,L
        voiceP = voiceP.unsqueeze(-1).expand(
            batch_size, topk, D, seq_length
        )  # B,topk,D,L

        # END cat mode
        output = torch.cat((output, voiceP), dim=2).view(
            -1, N + D, seq_length
        )  # [B*topk,(N+D),L]
        del voiceP

        # gated output layer for filter generation
        bf_filter = self.output(output) * self.output_gate(output)  # B*nspk, K, L
        # self.feature_dim)  # B, nspk, L, N
        # bf_filter = output

        return bf_filter


# base module for FaSNet
class FaSNet_base(AbsEnhancement):
    def __init__(
        self,
        fs=8000,
        enc_dim=64,
        feature_dim=64,
        hidden_dim=128,
        layer=6,
        segment_size=250,
        nspk=2,
        win_len=2,
    ):
        """Fasnet base.

        Reference:
            "Dual-path RNN: efficient long sequence modeling for
            time-domain single-channel speech separation", Yi Luo, Zhuo Chen
            and Takuya Yoshioka. https://arxiv.org/abs/1910.06379

        Based on https://github.com/kaituoxu/Conv-TasNet and
              https://github.com/yluo42/TAC
        """
        super(FaSNet_base, self).__init__()

        # parameters
        self.fs = fs
        self.window = win_len
        self.stride = self.window // 2

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim  # 64
        self.hidden_dim = hidden_dim  # 128
        self.segment_size = segment_size  # 250

        self.layer = layer  # 6
        self.num_spk = nspk  # 2
        self.eps = 1e-8

        # waveform encoder
        # self.encoder = nn.Conv1d(1, self.enc_dim, self.feature_dim, bias=False)
        self.encoder = Encoder(win_len, enc_dim)  # [B T]-->[B N L]
        # Notice: the norm is groupNorm in raw drpnn, but gLN in Asteroid.
        # self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8) # [B N L]-->[B N L]
        # self.enc_LN= choose_norm('GroupNorm', self.enc_dim)
        self.enc_LN = choose_norm("gLN", self.enc_dim)
        self.separator = SEP_module(
            input_dim=self.enc_dim,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_spk=self.num_spk,
            layer=self.layer,
            segment_size=self.segment_size,
        )
        # [B, N, L] -> [B, E, L]
        self.mask_conv1x1 = nn.Conv1d(self.feature_dim, self.enc_dim, 1, bias=False)
        # self.mask_conv1x1 = nn.Conv1d(self.feature_dim + 512,
        #                               self.enc_dim, 1, bias=False)
        # self.mask_conv1x1 = nn.Conv1d(self.feature_dim + 64,
        #                               self.enc_dim, 1, bias=False)
        self.decoder = Decoder(enc_dim, win_len)
        self.forward_rawwav = self.forward
        self.stft = None
        self.loss_type = "si_snr"

    def pad_input(self, input, window):
        """Zero-padding input according to window/stride size."""
        batch_size, nsample = input.shape
        stride = window // 2

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input, voiceP=None):
        """FaSNet_base forward.

        Args:
            input: torch.Tensor(batch, T)
            voiceP: (batch, topk, spk_emb) reserved target-spk query
        Returns:
            est_source: List[torch.Tensor(batch, T)]
            voiceP
            masks: OrderedDict
        """
        # pass to a DPRNN
        # input = input.to(device)
        B, _ = input.size()
        # input, rest = self.pad_input(input, self.window)
        mixture_w = self.encoder(input)  # B, E, L

        score_ = self.enc_LN(mixture_w)  # B, E, L
        score_ = self.separator(score_, voiceP)  # B*nspk, N, L
        # score_ = score_.view(B*self.num_spk, -1, self.feature_dim).\
        #     transpose(1, 2).contiguous()  # B*nspk, N, T

        # score_ = voiceP.transpose(1,2) * score_ # bs*steps*d * bs*spk*d*steps

        score = self.mask_conv1x1(score_)  # [B*nspk, N, L] -> [B*nspk, E, L]
        score = score.view(
            B, self.num_spk, self.enc_dim, -1
        )  # [B*nspk, E, L] -> [B, nspk, E, L]
        est_mask = F.relu(score)

        # est_mask = voiceP.unsqueeze(1).transpose(2,3) * \
        #            est_mask # bs*steps*d * bs*spk*d*steps

        est_source = self.decoder(
            mixture_w, est_mask
        )  # [B, E, L] + [B, nspk, E, L]--> [B, nspk, T]
        T_origin = input.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))  # M,C,T

        # if rest > 0:
        #     est_source = est_source[:, :, :-rest]
        est_source = [es for es in est_source.transpose(0, 1)]  # List(B,T)
        masks = OrderedDict(
            zip(["spk{}".format(i + 1) for i in range(self.num_spk)], est_source)
        )

        return est_source, voiceP, masks

    def forward_rawwav(self, mixture, ilens=None):
        return self.forward(mixture, ilens)


if __name__ == "__main__":
    mixture = torch.randn(3, 2001)  # bs, samples
    print("input shape", mixture.shape)
    voiceP = torch.randn(3, 2, 256)  # bs, topk , emb , reserved spk-target embedding

    print("\n")
    net = FaSNet_base(feature_dim=128, nspk=2)
    output = net(mixture, voiceP)[0]
    print("1st spk output shape", output[0].shape)
    assert output[0].shape == mixture.shape

    print("\n")
    net = FaSNet_base(segment_size=100, nspk=2, win_len=16)
    output = net(mixture)[0]
    print("1st spk output shape", output[0].shape)
    assert output[0].shape == mixture.shape
