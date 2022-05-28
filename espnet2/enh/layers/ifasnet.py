# The implementation of iFaSNet in
# Luo. et al. "Implicit Filter-and-sum Network for
# Multi-channel Speech Separation"
#
# The implementation is based on:
# https://github.com/yluo42/TAC
# Licensed under CC BY-NC-SA 3.0 US.
#

import torch
import torch.nn as nn

from espnet2.enh.layers import dprnn
from espnet2.enh.layers.fasnet import BF_module, FaSNet_base


# implicit FaSNet (iFaSNet)
class iFaSNet(FaSNet_base):
    def __init__(self, *args, **kwargs):
        super(iFaSNet, self).__init__(*args, **kwargs)

        self.context = self.context_len * 2 // self.win_len
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
            dropout=self.dropout,
            fasnet_type="ifasnet",
        )

        # waveform encoder/decoder
        self.encoder = nn.Conv1d(
            1, self.enc_dim, self.window, stride=self.stride, bias=False
        )
        self.decoder = nn.ConvTranspose1d(
            self.enc_dim, 1, self.window, stride=self.stride, bias=False
        )
        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=self.eps)

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
        input, rest = self.pad_input(input, self.window)

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
