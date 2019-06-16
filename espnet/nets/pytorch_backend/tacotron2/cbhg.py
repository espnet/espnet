#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class CBHG(torch.nn.Module):
    """CBHG module to convert log mel-fbank to linear spectrogram

    Reference:
        Tacotron: Towards End-to-End Speech Synthesis
        (https://arxiv.org/abs/1703.10135)

    :param int idim: dimension of the inputs
    :param int odim: dimension of the outputs
    :param int conv_bank_layers: the number of convolution bank layers
    :param int conv_bank_chans: the number of channels in convolution bank
    :param int conv_proj_filts: kernel size of convolutional projection layer
    :param int conv_proj_chans: the number of channels in convolutional projection layer
    :param int highway_layers: the number of highway network layers
    :param int highway_units: the number of highway network units
    :param int gru_units: the number of GRU units (for both directions)
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
        """CBHG module forward

        :param torch.Tensor xs: batch of the sequences of inputs (B, Tmax, idim)
        :param torch.Tensor ilens: list of lengths of each input batch (B)
        :return: batch of sequences of padded outputs (B, Tmax, eunits)
        :rtype: torch.Tensor
        :return: batch of lengths of each encoder states (B)
        :rtype: list
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
        """CBHG module inference

        :param torch.Tensor x: input (T, idim)
        :return: the sequence encoder states (T, odim)
        :rtype: torch.Tensor
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
    """Highway Network

    :param int idim: dimension of the inputs
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
        """Highway Network forward

        :param torch.Tensor x: batch of inputs (B, *, idim)
        :return: batch of outputs (B, *, idim)
        :rtype: torch.Tensor
        """
        proj = self.projection(x)
        gate = self.gate(x)
        return proj * gate + x * (1.0 - gate)
