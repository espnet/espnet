#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class CBHG(torch.nn.Module):
    """CBHG module to convert log Mel-filterbanks to linear spectrogram.

    This is a module of CBHG introduced in `Tacotron: Towards End-to-End Speech Synthesis`_.
    The CBHG converts the sequence of log Mel-filterbanks into linear spectrogram.

    Args:
        idim (int): Dimension of the inputs.
        odim (int): Dimension of the outputs.
        conv_bank_layers (int, optional): The number of convolution bank layers.
        conv_bank_chans (int, optional): The number of channels in convolution bank.
        conv_proj_filts (int, optional): Kernel size of convolutional projection layer.
        conv_proj_chans (int, optional): The number of channels in convolutional projection layer.
        highway_layers (int, optional): The number of highway network layers.
        highway_units (int, optional): The number of highway network units.
        gru_units (int, optional): The number of GRU units (for both directions).

    .. _`Tacotron: Towards End-to-End Speech Synthesis`: https://arxiv.org/abs/1703.10135

    """

    def __init__(self,
                 idim,
                 odim,
                 conv_bank_layers=8,
                 conv_bank_chans=128,
                 conv_proj_filts=3,
                 conv_proj_chans=256,
                 highway_layers=4,
                 highway_units=128,
                 gru_units=256):
        super(CBHG, self).__init__()
        self.idim = idim
        self.odim = odim
        self.conv_bank_layers = conv_bank_layers
        self.conv_bank_chans = conv_bank_chans
        self.conv_proj_filts = conv_proj_filts
        self.conv_proj_chans = conv_proj_chans
        self.highway_layers = highway_layers
        self.highway_units = highway_units
        self.gru_units = gru_units

        # define 1d convolution bank
        self.conv_bank = torch.nn.ModuleList()
        for k in range(1, self.conv_bank_layers + 1):
            if k % 2 != 0:
                padding = (k - 1) // 2
            else:
                padding = ((k - 1) // 2, (k - 1) // 2 + 1)
            self.conv_bank += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(padding, 0.0),
                torch.nn.Conv1d(idim, self.conv_bank_chans, k, stride=1,
                                padding=0, bias=True),
                torch.nn.BatchNorm1d(self.conv_bank_chans),
                torch.nn.ReLU())]

        # define max pooling (need padding for one-side to keep same length)
        self.max_pool = torch.nn.Sequential(
            torch.nn.ConstantPad1d((0, 1), 0.0),
            torch.nn.MaxPool1d(2, stride=1))

        # define 1d convolution projection
        self.projections = torch.nn.Sequential(
            torch.nn.Conv1d(self.conv_bank_chans * self.conv_bank_layers, self.conv_proj_chans,
                            self.conv_proj_filts, stride=1,
                            padding=(self.conv_proj_filts - 1) // 2, bias=True),
            torch.nn.BatchNorm1d(self.conv_proj_chans),
            torch.nn.ReLU(),
            torch.nn.Conv1d(self.conv_proj_chans, self.idim,
                            self.conv_proj_filts, stride=1,
                            padding=(self.conv_proj_filts - 1) // 2, bias=True),
            torch.nn.BatchNorm1d(self.idim),
        )

        # define highway network
        self.highways = torch.nn.ModuleList()
        self.highways += [torch.nn.Linear(idim, self.highway_units)]
        for _ in range(self.highway_layers):
            self.highways += [HighwayNet(self.highway_units)]

        # define bidirectional GRU
        self.gru = torch.nn.GRU(self.highway_units, gru_units // 2, num_layers=1,
                                batch_first=True, bidirectional=True)

        # define final projection
        self.output = torch.nn.Linear(gru_units, odim, bias=True)

    def forward(self, xs, ilens):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of the padded sequences of inputs (B, Tmax, idim).
            ilens (LongTensor): Batch of lengths of each input sequence (B,).

        Return:
            Tensor: Batch of the padded sequence of outputs (B, Tmax, odim).
            LongTensor: Batch of lengths of each output sequence (B,).

        """
        xs = xs.transpose(1, 2)  # (B, idim, Tmax)
        convs = []
        for k in range(self.conv_bank_layers):
            convs += [self.conv_bank[k](xs)]
        convs = torch.cat(convs, dim=1)  # (B, #CH * #BANK, Tmax)
        convs = self.max_pool(convs)
        convs = self.projections(convs).transpose(1, 2)  # (B, Tmax, idim)
        xs = xs.transpose(1, 2) + convs
        # + 1 for dimension adjustment layer
        for l in range(self.highway_layers + 1):
            xs = self.highways[l](xs)

        # sort by length
        xs, ilens, sort_idx = self._sort_by_length(xs, ilens)

        # total_length needs for DataParallel
        # (see https://github.com/pytorch/pytorch/pull/6327)
        total_length = xs.size(1)
        xs = pack_padded_sequence(xs, ilens, batch_first=True)
        self.gru.flatten_parameters()
        xs, _ = self.gru(xs)
        xs, ilens = pad_packed_sequence(xs, batch_first=True, total_length=total_length)

        # revert sorting by length
        xs, ilens = self._revert_sort_by_length(xs, ilens, sort_idx)

        xs = self.output(xs)  # (B, Tmax, odim)

        return xs, ilens

    def inference(self, x):
        """Inference.

        Args:
            x (Tensor): The sequences of inputs (T, idim).

        Return:
            Tensor: The sequence of outputs (T, odim).

        """
        assert len(x.size()) == 2
        xs = x.unsqueeze(0)
        ilens = x.new([x.size(0)]).long()

        return self.forward(xs, ilens)[0][0]

    def _sort_by_length(self, xs, ilens):
        sort_ilens, sort_idx = ilens.sort(0, descending=True)
        return xs[sort_idx], ilens[sort_idx], sort_idx

    def _revert_sort_by_length(self, xs, ilens, sort_idx):
        _, revert_idx = sort_idx.sort(0)
        return xs[revert_idx], ilens[revert_idx]


class HighwayNet(torch.nn.Module):
    """Highway Network module.

    This is a module of Highway Network introduced in `Highway Networks`_.

    Args:
        idim (int): Dimension of the inputs.

    .. _`Highway Networks`: https://arxiv.org/abs/1505.00387

    """

    def __init__(self, idim):
        super(HighwayNet, self).__init__()
        self.idim = idim
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(idim, idim),
            torch.nn.ReLU())
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(idim, idim),
            torch.nn.Sigmoid())

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of inputs (B, *, idim).

        Returns:
            Tensor: Batch of outputs, which are the same shape as inputs (B, *, idim).

        """
        proj = self.projection(x)
        gate = self.gate(x)
        return proj * gate + x * (1.0 - gate)
