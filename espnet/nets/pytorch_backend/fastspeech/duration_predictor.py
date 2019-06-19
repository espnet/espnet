#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class DurationPredictor(torch.nn.Module):
    """Duration predictor module

    Reference:
        FastSpeech: Fast, Robust and Controllable Text to Speech
        (https://arxiv.org/pdf/1905.09263.pdf)

    :param int idim: input dimension
    :param int n_layers: number of convolutional layers
    :param int n_chans: number of channels of convolutional layers
    :param int kernel_size: kernel size of convolutional layers
    :param float dropout_rate: dropout rate
    :param float offset: offset value to avoid nan in log domain
    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0):
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, 1)

    def _forward(self, xs, x_masks=None, is_inferece=False):
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)  # (B, Tmax)

        if is_inferece:
            # NOTE: calculate in linear domain
            xs = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long()  # avoid negative value

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        return xs

    def forward(self, xs, x_masks=None):
        """Calculate duration predictor forward propagation

        :param torch.Tensor xs: input tensor (B, Tmax, idim)
        :param torch.Tensor x_masks: mask for removing padded part (B, Tmax)
        :return torch.Tensor: predicted duration tensor in log domain (B, Tmax)
        """
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):
        """Inference duration

        :param torch.Tensor xs: input tensor with tha shape (B, Tmax, idim)
        :param torch.Tensor x_masks: mask for removing padded part (B, Tmax)
        :return torch.Tensor: predicted duration in linear domain with the shape (B, Tmax)
        """
        return self._forward(xs, x_masks, True)


class DurationPredictorLoss(torch.nn.Module):
    """Duration predictor loss module

    :param float offset: offset value to avoid nan in log domain
    """

    def __init__(self, offset=1.0):
        super(DurationPredictorLoss, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.offset = offset

    def forward(self, outputs, targets):
        """Calculate loss value

        :param torch.Tensor outputs: prediction duration in log domain (B, T)
        :param torch.Tensor targets: groundtruth duration in linear domain (B, T)
        """
        # NOTE: outputs is in log domain while targets in linear
        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets)

        return loss
