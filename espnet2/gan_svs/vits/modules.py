#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch


class Projection(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_mask):
        # print("xmask",x_mask.shape)
        # print("self.proj(x)",self.proj(x).shape)
        stats = self.proj(x) * x_mask
        m_p, logs_p = torch.split(stats, self.out_channels, dim=1)
        return m_p, logs_p


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)
