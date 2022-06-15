# The implementation of DPTNet proposed in
# J. Chen, Q. Mao, and D. Liu, “Dual-path transformer network:
# Direct context-aware modeling for end-to-end monaural speech
# separation,” in Proc. ISCA Interspeech, 2020, pp. 2642–2646.
#
# Ported from https://github.com/ujscjj/DPTNet

import torch.nn as nn

from espnet2.enh.layers.tcn import choose_norm
from espnet.nets.pytorch_backend.nets_utils import get_activation


class SingleTransformer(nn.Module):
    """Container module of the (improved) Transformer proposed in [1].

    Reference:
        Dual-path transformer network: Direct context-aware modeling for end-to-end
        monaural speech separation; Chen et al, Interspeech 2020.

    Args:
        rnn_type (str): select from 'RNN', 'LSTM' and 'GRU'.
        input_size (int): Dimension of the input feature.
        att_heads (int): Number of attention heads.
        hidden_size (int): Dimension of the hidden state.
        dropout (float): Dropout ratio. Default is 0.
        activation (str): activation function applied at the output of RNN.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        norm (str, optional): Type of normalization to use.
    """

    def __init__(
        self,
        rnn_type,
        input_size,
        att_heads,
        hidden_size,
        dropout=0.0,
        activation="relu",
        bidirectional=True,
        norm="gLN",
    ):
        super().__init__()

        rnn_type = rnn_type.upper()
        assert rnn_type in [
            "RNN",
            "LSTM",
            "GRU",
        ], f"Only support 'RNN', 'LSTM' and 'GRU', current type: {rnn_type}"
        self.rnn_type = rnn_type

        self.att_heads = att_heads
        self.self_attn = nn.MultiheadAttention(input_size, att_heads, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm_attn = choose_norm(norm, input_size)

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        activation = get_activation(activation)
        hdim = 2 * hidden_size if bidirectional else hidden_size
        self.feed_forward = nn.Sequential(
            activation, nn.Dropout(p=dropout), nn.Linear(hdim, input_size)
        )

        self.norm_ff = choose_norm(norm, input_size)

    def forward(self, x, attn_mask=None):
        # (batch, seq, input_size) -> (seq, batch, input_size)
        src = x.permute(1, 0, 2)
        # (seq, batch, input_size) -> (batch, seq, input_size)
        out = self.self_attn(src, src, src, attn_mask=attn_mask)[0].permute(1, 0, 2)
        out = self.dropout(out) + x
        # ... -> (batch, input_size, seq) -> ...
        out = self.norm_attn(out.transpose(-1, -2)).transpose(-1, -2)

        out2 = self.feed_forward(self.rnn(out)[0])
        out2 = self.dropout(out2) + out
        return self.norm_ff(out2.transpose(-1, -2)).transpose(-1, -2)


class DPTNet(nn.Module):
    """Dual-path transformer network.

    args:
        rnn_type (str): select from 'RNN', 'LSTM' and 'GRU'.
        input_size (int): dimension of the input feature.
            Input size must be a multiple of `att_heads`.
        hidden_size (int): dimension of the hidden state.
        output_size (int): dimension of the output size.
        att_heads (int): number of attention heads.
        dropout (float): dropout ratio. Default is 0.
        activation (str): activation function applied at the output of RNN.
        num_layers (int): number of stacked RNN layers. Default is 1.
        bidirectional (bool): whether the RNN layers are bidirectional. Default is True.
        norm_type (str): type of normalization to use after each inter- or
            intra-chunk Transformer block.
    """

    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        output_size,
        att_heads=4,
        dropout=0,
        activation="relu",
        num_layers=1,
        bidirectional=True,
        norm_type="gLN",
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # dual-path transformer
        self.row_transformer = nn.ModuleList()
        self.col_transformer = nn.ModuleList()
        for i in range(num_layers):
            self.row_transformer.append(
                SingleTransformer(
                    rnn_type,
                    input_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    bidirectional=True,
                    norm=norm_type,
                )
            )  # intra-segment RNN is always noncausal
            self.col_transformer.append(
                SingleTransformer(
                    rnn_type,
                    input_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    bidirectional=bidirectional,
                    norm=norm_type,
                )
            )

        # output layer
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size, 1))

    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply Transformer on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        # input = input.to(device)
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_transformer)):
            output = self.intra_chunk_process(output, i)
            output = self.inter_chunk_process(output, i)

        output = self.output(output)  # B, output_size, dim1, dim2

        return output

    def intra_chunk_process(self, x, layer_index):
        batch, N, chunk_size, n_chunks = x.size()
        x = x.transpose(1, -1).reshape(batch * n_chunks, chunk_size, N)
        x = self.row_transformer[layer_index](x)
        x = x.reshape(batch, n_chunks, chunk_size, N).permute(0, 3, 2, 1)
        return x

    def inter_chunk_process(self, x, layer_index):
        batch, N, chunk_size, n_chunks = x.size()
        x = x.permute(0, 2, 3, 1).reshape(batch * chunk_size, n_chunks, N)
        x = self.col_transformer[layer_index](x)
        x = x.reshape(batch, chunk_size, n_chunks, N).permute(0, 3, 1, 2)
        return x
