#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list


class LengthRegulator(torch.nn.Module):
    """Length regulator module

    Reference:
        FastSpeech: Fast, Robust and Controllable Text to Speech
        (https://arxiv.org/pdf/1905.09263.pdf)

    :param float pad_value: value used for padding
    """

    def __init__(self, pad_value=0.0):
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(self, xs, ds, ilens, alpha=1.0):
        """Apply length regulator

        :param torch.Tensor xs: input tensor with the shape (B, Tmax, D)
        :param torch.Tensor ds: duration of each components of each sequence (B, T)
        :param torch.Tensor ilens: batch of input lengths (B,)
        :param float alpha: alpha value to control speed of speech
        :return torch.Tensor: length regularized input tensor (B, T*, D)
        """
        assert alpha > 0
        if alpha != 1.0:
            ds = torch.round(ds.float() * alpha).long()
        xs = [x[:ilen] for x, ilen in zip(xs, ilens)]
        ds = [d[:ilen] for d, ilen in zip(ds, ilens)]
        xs = [self._repeat_one_sequence(x, d) for x, d in zip(xs, ds)]

        return pad_list(xs, self.pad_value)

    def _repeat_one_sequence(self, x, d):
        """Repeat each frame according to duration

        >>> x = torch.tensor([[1], [2], [3]])
        tensor([[1],
                [2],
                [3]])
        >>> d = torch.tensor([1, 2, 3])
        tensor([1, 2, 3])
        >>> self._repeat_one_sequence(x, d)
        tensor([[1],
                [2],
                [2],
                [3],
                [3],
                [3]])

        :param torch.Tensor x: input tensor with the shape (T, D)
        :param torch.Tensor d: duration of each frame of input tensor (T,)
        :return torch.Tensor: length regularized input tensor (T*, D)
        """
        return torch.cat([x_.repeat(int(d_), 1) for x_, d_ in zip(x, d) if d_ != 0], dim=0)
