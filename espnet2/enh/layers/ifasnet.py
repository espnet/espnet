# The implementation of iFaSNet in
# Luo. et al. "Implicit Filter-and-sum Network for
# Multi-channel Speech Separation"
#
# The implementation is based on:
# https://github.com/yluo42/TAC
# Licensed under CC BY-NC-SA 3.0 US.
#

from pyexpat import model
import torch
import torch.nn as nn

from espnet2.enh.layers import dprnn


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
    ):
        super().__init__()

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
        )
        self.eps = 1e-8

        # output layer
        self.output = nn.Conv1d(self.feature_dim, self.output_dim, 1)
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
        sr=16000,
    ):
        super(FaSNet_base, self).__init__()

        # parameters
        self.window = max(int(sr * win_len / 1000), 2)
        self.stride = self.window // 2
        self.context = context_len * 2 // win_len

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size

        self.layer = layer
        self.num_spk = nspk
        self.eps = 1e-8

        # waveform encoder/decoder
        self.encoder = nn.Conv1d(
            1, self.enc_dim, self.window, stride=self.stride, bias=False
        )
        self.decoder = nn.ConvTranspose1d(
            self.enc_dim, 1, self.window, stride=self.stride, bias=False
        )
        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=self.eps)

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nmic, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, nmic, rest).type(input.type())
            input = torch.cat([input, pad], 2)
        pad_aux = torch.zeros(batch_size, nmic, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def signal_context(self, x, context):
        """
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

    def forward(self, input, num_mic):
        """
        input: shape (batch, max_num_ch, T)
        num_mic: shape (batch, ), the number of channels for each input. Zero for fixed geometry configuration.
        """
        pass


# implicit FaSNet (iFaSNet)
class iFaSNet(FaSNet_base):
    def __init__(self, *args, **kwargs):
        super(iFaSNet, self).__init__(*args, **kwargs)

        # context compression
        self.summ_BN = nn.Linear(self.enc_dim, self.feature_dim)
        self.summ_RNN = dprnn.SingleRNN(
            "LSTM", self.feature_dim, self.hidden_dim, bidirectional=True
        )
        self.summ_LN = nn.GroupNorm(1, self.feature_dim, eps=self.eps)
        self.summ_output = nn.Linear(self.feature_dim, self.enc_dim)

        self.separator = BF_module(
            self.enc_dim + (self.context * 2 + 1) ** 2,
            self.feature_dim,
            self.hidden_dim,
            self.enc_dim,
            self.num_spk,
            self.layer,
            self.segment_size,
        )

        # context decompression
        self.gen_BN = nn.Conv1d(self.enc_dim * 2, self.feature_dim, 1)
        self.gen_RNN = dprnn.SingleRNN(
            "LSTM", self.feature_dim, self.hidden_dim, bidirectional=True
        )
        self.gen_LN = nn.GroupNorm(1, self.feature_dim, eps=self.eps)
        self.gen_output = nn.Conv1d(self.feature_dim, self.enc_dim, 1)

    def forward(self, input, num_mic):

        batch_size = input.size(0)
        nmic = input.size(1)

        # pad input accordingly
        input, rest = self.pad_input(input, self.window, self.stride)

        # encoder on all channels
        enc_output = self.encoder(input.view(batch_size * nmic, 1, -1))  # B*nmic, N, L
        seq_length = enc_output.shape[-1]

        # calculate the context of the encoder output
        # consider both past and future
        enc_context = self.signal_context(
            enc_output, self.context
        )  # B*nmic, N, 2C+1, L
        enc_context = enc_context.view(
            batch_size, nmic, self.enc_dim, -1, seq_length
        )  # B, nmic, N, 2C+1, L

        # NCC feature
        ref_enc = enc_context[:, 0].contiguous()  # B, N, 2C+1, L
        ref_enc = (
            ref_enc.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size * seq_length, self.enc_dim, -1)
        )  # B*L, N, 2C+1
        enc_context_copy = (
            enc_context.permute(0, 4, 1, 3, 2)
            .contiguous()
            .view(batch_size * seq_length, nmic, -1, self.enc_dim)
        )  # B*L, nmic, 2C+1, N
        NCC = torch.cat(
            [enc_context_copy[:, i].bmm(ref_enc).unsqueeze(1) for i in range(nmic)], 1
        )  # B*L, nmic, 2C+1, 2C+1
        ref_norm = (
            ref_enc.pow(2).sum(1).unsqueeze(1) + self.eps
        ).sqrt()  # B*L, 1, 2C+1
        enc_norm = (
            enc_context_copy.pow(2).sum(3).unsqueeze(3) + self.eps
        ).sqrt()  # B*L, nmic, 2C+1, 1
        NCC = NCC / (ref_norm.unsqueeze(1) * enc_norm)  # B*L, nmic, 2C+1, 2C+1
        NCC = torch.cat(
            [NCC[:, :, i] for i in range(NCC.shape[2])], 2
        )  # B*L, nmic, (2C+1)^2
        NCC = (
            NCC.view(batch_size, seq_length, nmic, -1).permute(0, 2, 3, 1).contiguous()
        )  # B, nmic, (2C+1)^2, L

        # context compression
        norm_output = self.enc_LN(enc_output)  # B*nmic, N, L
        norm_context = self.signal_context(
            norm_output, self.context
        )  # B*nmic, N, 2C+1, L
        norm_context = (
            norm_context.permute(0, 3, 2, 1)
            .contiguous()
            .view(-1, self.context * 2 + 1, self.enc_dim)
        )
        norm_context_BN = self.summ_BN(norm_context.view(-1, self.enc_dim)).view(
            -1, self.context * 2 + 1, self.feature_dim
        )
        embedding = (
            self.summ_RNN(norm_context_BN).transpose(1, 2).contiguous()
        )  # B*nmic*L, N, 2C+1
        embedding = norm_context_BN.transpose(1, 2).contiguous() + self.summ_LN(
            embedding
        )  # B*nmic*L, N, 2C+1
        embedding = self.summ_output(embedding.mean(2)).view(
            batch_size, nmic, seq_length, self.enc_dim
        )  # B, nmic, L, N
        embedding = embedding.transpose(2, 3).contiguous()  # B, nmic, N, L

        input_feature = torch.cat([embedding, NCC], 2)  # B, nmic, N+(2C+1)^2, L

        # pass to DPRNN-TAC
        embedding = self.separator(input_feature, num_mic)[
            :, 0
        ].contiguous()  # B, nspk, N, L

        # concatenate with encoder outputs and generate masks
        # context decompression
        norm_context = norm_context.view(
            batch_size, nmic, seq_length, -1, self.enc_dim
        )  # B, nmic, L, 2C+1, N
        norm_context = norm_context.permute(0, 1, 4, 3, 2)[
            :, :1
        ].contiguous()  # B, 1, N, 2C+1, L

        embedding = torch.cat(
            [embedding.unsqueeze(3)] * (self.context * 2 + 1), 3
        )  # B, nspk, N, 2C+1, L
        norm_context = torch.cat(
            [norm_context] * self.num_spk, 1
        )  # B, nspk, N, 2C+1, L
        embedding = (
            torch.cat([norm_context, embedding], 2).permute(0, 1, 4, 2, 3).contiguous()
        )  # B, nspk, L, 2N, 2C+1
        all_filter = self.gen_BN(
            embedding.view(-1, self.enc_dim * 2, self.context * 2 + 1)
        )  # B*nspk*L, N, 2C+1
        all_filter = all_filter + self.gen_LN(
            self.gen_RNN(all_filter.transpose(1, 2)).transpose(1, 2)
        )  # B*nspk*L, N, 2C+1
        all_filter = self.gen_output(all_filter)  # B*nspk*L, N, 2C+1
        all_filter = all_filter.view(
            batch_size, self.num_spk, seq_length, self.enc_dim, -1
        )  # B, nspk, L, N+1, 2C+1
        all_filter = all_filter.permute(
            0, 1, 3, 4, 2
        ).contiguous()  # B, nspk, N, 2C+1, L

        # apply to with ref mic's encoder context
        output = (enc_context[:, :1] * all_filter).mean(3)  # B, nspk, N, L

        # decode
        bf_signal = self.decoder(
            output.view(batch_size * self.num_spk, self.enc_dim, -1)
        )  # B*nspk, 1, T

        if rest > 0:
            bf_signal = bf_signal[:, :, self.stride : -rest - self.stride]

        bf_signal = bf_signal.view(batch_size, self.num_spk, -1)  # B, nspk, T

        return bf_signal


def test_model(model):
    import numpy as np

    x = torch.rand(3, 4, 32000)  # (batch, num_mic, length)
    num_mic = (
        torch.from_numpy(np.array([3, 3, 2]))
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
    model_iFaSNet = iFaSNet(
        enc_dim=64,
        feature_dim=64,
        hidden_dim=128,
        layer=6,
        segment_size=24,
        nspk=2,
        win_len=16,
        context_len=16,
        sr=16000,
    )

    test_model(model_iFaSNet)
