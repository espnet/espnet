#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list


class LengthRegulator(torch.nn.Module):
    """Length regulator module for feed-forward Transformer.

    This is a module of length regulator described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The length regulator expands char or phoneme-level embedding features to frame-level by repeating each
    feature based on the corresponding predicted durations.

    Args:
        pad_value (float, optional): Value used for padding.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, pad_value=0.0):
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(self, xs, ds, ilens, alpha=1.0):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            ds (LongTensor): Batch of durations of each frame (B, T).
            ilens (LongTensor): Batch of input lengths (B,).
            alpha (float, optional): Alpha value to control speed of speech.

        Returns:
            Tensor: replicated input tensor based on durations (B, T*, D).

        """
        assert alpha > 0
        if alpha != 1.0:
            ds = torch.round(ds.float() * alpha).long()
        xs = [x[:ilen] for x, ilen in zip(xs, ilens)]
        ds = [d[:ilen] for d, ilen in zip(ds, ilens)]
        xs = [self._repeat_one_sequence(x, d) for x, d in zip(xs, ds)]

        return pad_list(xs, self.pad_value)

    def _repeat_one_sequence(self, x, d):
        """Repeat each frame according to duration.

        Examples:
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

        """
        if d.sum() == 0:
            logging.warn("all of the predicted durations are 0. fill 0 with 1.")
            d = d.fill_(1)
        return torch.cat([x_.repeat(int(d_), 1) for x_, d_ in zip(x, d) if d_ != 0], dim=0)
