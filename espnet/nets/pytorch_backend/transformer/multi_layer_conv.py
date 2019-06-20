#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch


class MultiLayeredConv1d(torch.nn.Module):
    """Multi-layered conv1d for Transformer block

    Reference:
        FastSpeech: Fast, Robust and Controllable Text to Speech
        (https://arxiv.org/pdf/1905.09263.pdf)

    :param int in_chans: number of input channels
    :param int hidden_chans: number of hidden channels
    :param int kernel_size: kernel size of conv1d
    :param float dropout_rate: dropout rate
    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = torch.nn.Conv1d(in_chans, hidden_chans, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = torch.nn.Conv1d(hidden_chans, in_chans, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)
