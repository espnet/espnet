# The implementation of DPRNN in
# Luo. et al. "Dual-path rnn: efficient long sequence modeling
# for time-domain single-channel speech separation."
#
# The code is based on:
# https://github.com/yluo42/TAC/blob/master/utility/models.py
#


import torch
from torch.autograd import Variable
import torch.nn as nn


EPS = torch.finfo(torch.get_default_dtype()).eps


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
        super().__init__()

        rnn_type = rnn_type.upper()

        assert rnn_type in [
            "RNN",
            "LSTM",
            "GRU",
        ], f"Only support 'RNN', 'LSTM' and 'GRU', current type: {rnn_type}"

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        # input = input.to(device)
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.dropout(rnn_output)
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
        bidirectional: bool, whether the RNN layers are bidirectional. Default is True.
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
        super().__init__()

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


def _pad_segment(input, segment_size):
    # input is the features: (B, N, T)
    batch_size, dim, seq_len = input.shape
    segment_stride = segment_size // 2

    rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
    if rest > 0:
        pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
        input = torch.cat([input, pad], 2)

    pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
    input = torch.cat([pad_aux, input, pad_aux], 2)

    return input, rest


def split_feature(input, segment_size):
    # split the feature into chunks of segment size
    # input is the features: (B, N, T)

    input, rest = _pad_segment(input, segment_size)
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


def merge_feature(input, rest):
    # merge the splitted features into full utterance
    # input is the features: (B, N, L, K)

    batch_size, dim, segment_size, _ = input.shape
    segment_stride = segment_size // 2
    input = (
        input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)
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
