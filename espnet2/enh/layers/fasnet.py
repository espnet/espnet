# The implementation of FaSNet in
# Y. Luo, et al.  “FaSNet: Low-Latency Adaptive Beamforming
# for Multi-Microphone Audio Processing”
# The implementation is based on:
# https://github.com/yluo42/TAC
# Licensed under CC BY-NC-SA 3.0 US.
#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.enh.layers import dprnn


# DPRNN for beamforming filter estimation
class BF_module(nn.Module):
    def __init__(
        self,
        input_dim,
        feature_dim,
        hidden_dim,
        output_dim,
        num_spk=2,
        layer=4,
        segment_size=100,
        bidirectional=True,
        dropout=0.0,
        fasnet_type="ifasnet",
    ):
        super().__init__()

        assert fasnet_type in [
            "fasnet",
            "ifasnet",
        ], "fasnet_type should be fasnet or ifasnet"

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk

        self.dprnn_model = dprnn.DPRNN_TAC(
            "lstm",
            self.feature_dim,
            self.hidden_dim,
            self.feature_dim * self.num_spk,
            num_layers=layer,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.eps = 1e-8

        self.fasnet_type = fasnet_type

        if fasnet_type == "ifasnet":
            # output layer in ifasnet
            self.output = nn.Conv1d(self.feature_dim, self.output_dim, 1)
        elif fasnet_type == "fasnet":
            # gated output layer in ifasnet
            self.output = nn.Sequential(
                nn.Conv1d(self.feature_dim, self.output_dim, 1), nn.Tanh()
            )
            self.output_gate = nn.Sequential(
                nn.Conv1d(self.feature_dim, self.output_dim, 1), nn.Sigmoid()
            )

        self.num_spk = num_spk
        self.BN = nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False)

    def forward(self, input, num_mic):
        # input: (B, ch, N, T)
        batch_size, ch, N, seq_length = input.shape

        input = input.view(batch_size * ch, N, seq_length)  # B*ch, N, T
        enc_feature = self.BN(input)

        # split the encoder output into overlapped, longer segments
        enc_segments, enc_rest = dprnn.split_feature(
            enc_feature, self.segment_size
        )  # B*ch, N, L, K

        enc_segments = enc_segments.view(
            batch_size, ch, -1, enc_segments.shape[2], enc_segments.shape[3]
        )  # B, ch, N, L, K
        output = self.dprnn_model(enc_segments, num_mic).view(
            batch_size * ch * self.num_spk,
            self.feature_dim,
            self.segment_size,
            -1,
        )  # B*ch*nspk, N, L, K
        # overlap-and-add of the outputs
        output = dprnn.merge_feature(output, enc_rest)  # B*ch*nspk, N, T

        if self.fasnet_type == "fasnet":
            # gated output layer for filter generation
            bf_filter = self.output(output) * self.output_gate(
                output
            )  # B*ch*nspk, K, T
            bf_filter = (
                bf_filter.transpose(1, 2)
                .contiguous()
                .view(batch_size, ch, self.num_spk, -1, self.output_dim)
            )  # B, ch, nspk, L, N

        elif self.fasnet_type == "ifasnet":
            # output layer
            bf_filter = self.output(output)  # B*ch*nspk, K, T
            bf_filter = bf_filter.view(
                batch_size, ch, self.num_spk, self.output_dim, -1
            )  # B, ch, nspk, K, L

        return bf_filter


# base module for FaSNet
class FaSNet_base(nn.Module):
    def __init__(
        self,
        enc_dim,
        feature_dim,
        hidden_dim,
        layer,
        segment_size=24,
        nspk=2,
        win_len=16,
        context_len=16,
        dropout=0.0,
        sr=16000,
    ):
        super(FaSNet_base, self).__init__()

        # parameters
        self.win_len = win_len
        self.window = max(int(sr * win_len / 1000), 2)
        self.stride = self.window // 2
        self.sr = sr
        self.context_len = context_len
        self.dropout = dropout

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size

        self.layer = layer
        self.num_spk = nspk
        self.eps = 1e-8

    def pad_input(self, input, window):
        """Zero-padding input according to window/stride size."""

        batch_size, nmic, nsample = input.shape

        stride = self.stride

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, nmic, rest).type(input.type())
            input = torch.cat([input, pad], 2)
        pad_aux = torch.zeros(batch_size, nmic, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def seg_signal_context(self, x, window, context):
        """Segmenting the signal into chunks with specific context.

        input:
            x: size (B, ch, T)
            window: int
            context: int
        """

        # pad input accordingly
        # first pad according to window size
        input, rest = self.pad_input(x, window)
        batch_size, nmic, nsample = input.shape
        stride = window // 2

        # pad another context size
        pad_context = torch.zeros(batch_size, nmic, context).type(input.type())
        input = torch.cat([pad_context, input, pad_context], 2)  # B, ch, L

        # calculate index for each chunk
        nchunk = 2 * nsample // window - 1
        begin_idx = np.arange(nchunk) * stride
        begin_idx = (
            torch.from_numpy(begin_idx).type(input.type()).long().view(1, 1, -1)
        )  # 1, 1, nchunk
        begin_idx = begin_idx.expand(batch_size, nmic, nchunk)  # B, ch, nchunk
        # select entries from index
        chunks = [
            torch.gather(input, 2, begin_idx + i).unsqueeze(3)
            for i in range(2 * context + window)
        ]  # B, ch, nchunk, 1
        chunks = torch.cat(chunks, 3)  # B, ch, nchunk, chunk_size

        # center frame
        center_frame = chunks[:, :, :, context : context + window]

        return center_frame, chunks, rest

    def signal_context(self, x, context):
        """signal context function

        Segmenting the signal into chunks with specific context.
        input:
            x: size (B, dim, nframe)
            context: int
        """

        batch_size, dim, nframe = x.shape

        zero_pad = torch.zeros(batch_size, dim, context).type(x.type())
        pad_past = []
        pad_future = []
        for i in range(context):
            pad_past.append(
                torch.cat([zero_pad[:, :, i:], x[:, :, : -context + i]], 2).unsqueeze(2)
            )
            pad_future.append(
                torch.cat([x[:, :, i + 1 :], zero_pad[:, :, : i + 1]], 2).unsqueeze(2)
            )

        pad_past = torch.cat(pad_past, 2)  # B, D, C, L
        pad_future = torch.cat(pad_future, 2)  # B, D, C, L
        all_context = torch.cat(
            [pad_past, x.unsqueeze(2), pad_future], 2
        )  # B, D, 2*C+1, L

        return all_context

    def seq_cos_sim(self, ref, target):
        """Cosine similarity between some reference mics and some target mics

        ref: shape (nmic1, L, seg1)
        target: shape (nmic2, L, seg2)
        """

        assert ref.size(1) == target.size(1), "Inputs should have same length."
        assert ref.size(2) >= target.size(
            2
        ), "Reference input should be no smaller than the target input."

        seq_length = ref.size(1)

        larger_ch = ref.size(0)
        if target.size(0) > ref.size(0):
            ref = ref.expand(
                target.size(0), ref.size(1), ref.size(2)
            ).contiguous()  # nmic2, L, seg1
            larger_ch = target.size(0)
        elif target.size(0) < ref.size(0):
            target = target.expand(
                ref.size(0), target.size(1), target.size(2)
            ).contiguous()  # nmic1, L, seg2

        # L2 norms
        ref_norm = F.conv1d(
            ref.view(1, -1, ref.size(2)).pow(2),
            torch.ones(ref.size(0) * ref.size(1), 1, target.size(2)).type(ref.type()),
            groups=larger_ch * seq_length,
        )  # 1, larger_ch*L, seg1-seg2+1
        ref_norm = ref_norm.sqrt() + self.eps
        target_norm = (
            target.norm(2, dim=2).view(1, -1, 1) + self.eps
        )  # 1, larger_ch*L, 1
        # cosine similarity
        cos_sim = F.conv1d(
            ref.view(1, -1, ref.size(2)),
            target.view(-1, 1, target.size(2)),
            groups=larger_ch * seq_length,
        )  # 1, larger_ch*L, seg1-seg2+1
        cos_sim = cos_sim / (ref_norm * target_norm)

        return cos_sim.view(larger_ch, seq_length, -1)

    def forward(self, input, num_mic):
        """abstract forward function

        input: shape (batch, max_num_ch, T)
        num_mic: shape (batch, ), the number of channels for each input.
                 Zero for fixed geometry configuration.
        """
        pass


# single-stage FaSNet + TAC
class FaSNet_TAC(FaSNet_base):
    def __init__(self, *args, **kwargs):
        super(FaSNet_TAC, self).__init__(*args, **kwargs)

        self.context = int(self.sr * self.context_len / 1000)
        self.filter_dim = self.context * 2 + 1

        # DPRNN + TAC for estimation
        self.all_BF = BF_module(
            self.filter_dim + self.enc_dim,
            self.feature_dim,
            self.hidden_dim,
            self.filter_dim,
            self.num_spk,
            self.layer,
            self.segment_size,
            dropout=self.dropout,
            fasnet_type="fasnet",
        )

        # waveform encoder
        self.encoder = nn.Conv1d(
            1, self.enc_dim, self.context * 2 + self.window, bias=False
        )
        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8)

    def forward(self, input, num_mic):
        batch_size = input.size(0)
        nmic = input.size(1)

        # split input into chunks
        all_seg, all_mic_context, rest = self.seg_signal_context(
            input, self.window, self.context
        )  # B, nmic, L, win/chunk
        seq_length = all_seg.size(2)

        # embeddings for all channels
        enc_output = (
            self.encoder(all_mic_context.view(-1, 1, self.context * 2 + self.window))
            .view(batch_size * nmic, seq_length, self.enc_dim)
            .transpose(1, 2)
            .contiguous()
        )  # B*nmic, N, L
        enc_output = self.enc_LN(enc_output).view(
            batch_size, nmic, self.enc_dim, seq_length
        )  # B, nmic, N, L

        # calculate the cosine similarities for ref channel's center
        # frame with all channels' context

        ref_seg = all_seg[:, 0].contiguous().view(1, -1, self.window)  # 1, B*L, win
        all_context = (
            all_mic_context.transpose(0, 1)
            .contiguous()
            .view(nmic, -1, self.context * 2 + self.window)
        )  # 1, B*L, 3*win
        all_cos_sim = self.seq_cos_sim(all_context, ref_seg)  # nmic, B*L, 2*win+1
        all_cos_sim = (
            all_cos_sim.view(nmic, batch_size, seq_length, self.filter_dim)
            .permute(1, 0, 3, 2)
            .contiguous()
        )  # B, nmic, 2*win+1, L

        input_feature = torch.cat([enc_output, all_cos_sim], 2)  # B, nmic, N+2*win+1, L

        # pass to DPRNN
        all_filter = self.all_BF(input_feature, num_mic)  # B, ch, nspk, L, 2*win+1

        # convolve with all mic's context
        mic_context = torch.cat(
            [
                all_mic_context.view(
                    batch_size * nmic, 1, seq_length, self.context * 2 + self.window
                )
            ]
            * self.num_spk,
            1,
        )  # B*nmic, nspk, L, 3*win
        all_bf_output = F.conv1d(
            mic_context.view(1, -1, self.context * 2 + self.window),
            all_filter.view(-1, 1, self.filter_dim),
            groups=batch_size * nmic * self.num_spk * seq_length,
        )  # 1, B*nmic*nspk*L, win
        all_bf_output = all_bf_output.view(
            batch_size, nmic, self.num_spk, seq_length, self.window
        )  # B, nmic, nspk, L, win

        # reshape to utterance
        bf_signal = all_bf_output.view(
            batch_size * nmic * self.num_spk, -1, self.window * 2
        )
        bf_signal1 = (
            bf_signal[:, :, : self.window]
            .contiguous()
            .view(batch_size * nmic * self.num_spk, 1, -1)[:, :, self.stride :]
        )
        bf_signal2 = (
            bf_signal[:, :, self.window :]
            .contiguous()
            .view(batch_size * nmic * self.num_spk, 1, -1)[:, :, : -self.stride]
        )
        bf_signal = bf_signal1 + bf_signal2  # B*nmic*nspk, 1, T
        if rest > 0:
            bf_signal = bf_signal[:, :, :-rest]

        bf_signal = bf_signal.view(
            batch_size, nmic, self.num_spk, -1
        )  # B, nmic, nspk, T
        # consider only the valid channels
        if num_mic.max() == 0:
            bf_signal = bf_signal.mean(1)  # B, nspk, T
        else:
            bf_signal = [
                bf_signal[b, : num_mic[b]].mean(0).unsqueeze(0)
                for b in range(batch_size)
            ]  # nspk, T
            bf_signal = torch.cat(bf_signal, 0)  # B, nspk, T

        return bf_signal


def test_model(model):
    x = torch.rand(2, 4, 32000)  # (batch, num_mic, length)
    num_mic = (
        torch.from_numpy(np.array([3, 2]))
        .view(
            -1,
        )
        .type(x.type())
    )  # ad-hoc array
    none_mic = torch.zeros(1).type(x.type())  # fixed-array
    y1 = model(x, num_mic.long())
    y2 = model(x, none_mic.long())
    print(y1.shape, y2.shape)  # (batch, nspk, length)


if __name__ == "__main__":
    model_TAC = FaSNet_TAC(
        enc_dim=64,
        feature_dim=64,
        hidden_dim=128,
        layer=4,
        segment_size=50,
        nspk=2,
        win_len=4,
        context_len=16,
        sr=16000,
    )

    test_model(model_TAC)
