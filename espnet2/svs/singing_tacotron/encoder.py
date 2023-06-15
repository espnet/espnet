#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
# Copyright 2023 Renmin University of China (Yuning Wu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Singing Tacotron encoder related modules."""

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def encoder_init(m):
    """Initialize encoder parameters."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))


class Encoder(torch.nn.Module):
    """Encoder module of Spectrogram prediction network.

    This is a module of encoder of Spectrogram prediction network in Singing Tacotron,
    which described in `Singing-Tacotron: Global Duration Control Attention and Dynamic
    Filter for End-to-end Singing Voice Synthesis`_. This is the encoder which converts
    either a sequence of characters or acoustic features into the sequence of
    hidden states.

    .. _`Singing-Tacotron: Global Duration Control Attention and Dynamic
    Filter for End-to-end Singing Voice Synthesis`:
       https://arxiv.org/abs/2202.07907

    """

    def __init__(
        self,
        idim,
        input_layer="embed",
        embed_dim=512,
        elayers=1,
        eunits=512,
        econv_layers=3,
        econv_chans=512,
        econv_filts=5,
        use_batch_norm=True,
        use_residual=False,
        dropout_rate=0.5,
        padding_idx=0,
    ):
        """Initialize Singing Tacotron encoder module.

        Args:
            idim (int) Dimension of the inputs.
            input_layer (str): Input layer type.
            embed_dim (int, optional) Dimension of character embedding.
            elayers (int, optional) The number of encoder blstm layers.
            eunits (int, optional) The number of encoder blstm units.
            econv_layers (int, optional) The number of encoder conv layers.
            econv_filts (int, optional) The number of encoder conv filter size.
            econv_chans (int, optional) The number of encoder conv filter channels.
            use_batch_norm (bool, optional) Whether to use batch normalization.
            use_residual (bool, optional) Whether to use residual connection.
            dropout_rate (float, optional) Dropout rate.

        """
        super(Encoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.use_residual = use_residual

        # define network layer modules
        if input_layer == "linear":
            self.embed = torch.nn.Linear(idim, econv_chans)
        elif input_layer == "embed":
            self.embed = torch.nn.Embedding(idim, embed_dim, padding_idx=padding_idx)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        if econv_layers > 0:
            self.convs = torch.nn.ModuleList()
            for layer in range(econv_layers):
                ichans = (
                    embed_dim if layer == 0 and input_layer == "embed" else econv_chans
                )
                if use_batch_norm:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.BatchNorm1d(econv_chans),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
                else:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
        else:
            self.convs = None
        if elayers > 0:
            iunits = econv_chans if econv_layers != 0 else embed_dim
            self.blstm = torch.nn.LSTM(
                iunits, eunits // 2, elayers, batch_first=True, bidirectional=True
            )
        else:
            self.blstm = None

        # initialize
        self.apply(encoder_init)

    def forward(self, xs, ilens=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of the padded sequence. Either character ids (B, Tmax)
                or acoustic feature (B, Tmax, idim * encoder_reduction_factor). Padded
                value should be 0.
            ilens (LongTensor): Batch of lengths of each input batch (B,).

        Returns:
            Tensor: Batch of the sequences of encoder states(B, Tmax, eunits).
            LongTensor: Batch of lengths of each sequence (B,)

        """
        xs = xs.transpose(1, 2)
        if self.convs is not None:
            for i in range(len(self.convs)):
                if self.use_residual:
                    xs = xs + self.convs[i](xs)
                else:
                    xs = self.convs[i](xs)
        if self.blstm is None:
            return xs.transpose(1, 2)
        if not isinstance(ilens, torch.Tensor):
            ilens = torch.tensor(ilens)
        xs = pack_padded_sequence(
            xs.transpose(1, 2), ilens.cpu(), batch_first=True, enforce_sorted=False
        )
        self.blstm.flatten_parameters()
        xs, _ = self.blstm(xs)  # (B, Tmax, C)
        xs, hlens = pad_packed_sequence(xs, batch_first=True)

        return xs, hlens

    def inference(self, x, ilens):
        """Inference.

        Args:
            x (Tensor): The sequeunce of character ids (T,)
                    or acoustic feature (T, idim * encoder_reduction_factor).

        Returns:
            Tensor: The sequences of encoder states(T, eunits).

        """

        xs = x

        return self.forward(xs, ilens)[0][0]


class Duration_Encoder(torch.nn.Module):
    """Duration_Encoder module of Spectrogram prediction network.

    This is a module of encoder of Spectrogram prediction network in Singing-Tacotron,
    This is the encoder which converts the sequence
    of durations and tempo features into a transition token.

    .. _`SINGING-TACOTRON: GLOBAL DURATION CONTROL ATTENTION AND DYNAMIC FILTER FOR
    END-TO-END SINGING VOICE SYNTHESIS`:
       https://arxiv.org/abs/2202.07907

    """

    def __init__(
        self,
        idim,
        embed_dim=512,
        dropout_rate=0.5,
        padding_idx=0,
    ):
        """Initialize Singing-Tacotron encoder module.

        Args:
            idim (int) Dimension of the inputs.
            embed_dim (int, optional) Dimension of character embedding.
            dropout_rate (float, optional) Dropout rate.

        """
        super(Duration_Encoder, self).__init__()
        # store the hyperparameters
        self.idim = idim

        # define network layer modules
        self.dense24 = torch.nn.Linear(idim, 24)
        self.convs = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                24,
                32,
                3,
                stride=1,
                bias=False,
                padding=2 // 2,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                32,
                32,
                3,
                stride=1,
                bias=False,
                padding=2 // 2,
            ),
            torch.nn.ReLU(),
        )
        self.dense1 = torch.nn.Linear(32, 1)
        self.nntanh = torch.nn.Tanh()

        # initialize
        self.apply(encoder_init)

    def forward(self, xs):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of the duration sequence.(B, Tmax, feature_len)

        Returns:
            Tensor: Batch of the sequences of transition token (B, Tmax, 1).
            LongTensor: Batch of lengths of each sequence (B,)

        """
        xs = self.dense24(xs).transpose(1, 2)
        xs = self.convs(xs).transpose(1, 2)
        xs = self.dense1(xs)
        xs = self.nntanh(xs)
        xs = (xs + 1) / 2
        return xs

    def inference(self, x):
        """Inference."""

        xs = x

        return self.forward(xs)
