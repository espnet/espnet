#! /usr/bin/python
# -*- encoding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.spk.loss.abs_loss import AbsLoss


class MSE(AbsLoss):
    """Mean Squared Error loss for MOS prediction.

    args:
        nout: dimensionality of input embeddings
    """

    def __init__(self, nout, **kwargs):
        super().__init__(nout)
        self.in_feats = nout
        self.weight = nn.Parameter(torch.FloatTensor(1, nout), requires_grad=True)
        self.mse = nn.MSELoss(reduction="mean")
        nn.init.xavier_normal_(self.weight, gain=1)

    def forward(self, x, label=None):
        """Calculate MSE loss between predicted and true MOS scores.

        args:
            x: input embeddings (batch_size, nout)
            label: MOS scores (batch_size,)
        """
        if len(label.size()) == 2:
            label = label.squeeze(1)  # Handle (batch_size, 1) labels

        assert x.size()[0] == label.size()[0], "Batch size mismatch"
        assert x.size()[1] == self.in_feats, "Feature dimension mismatch"

        predicted_mos = F.linear(F.normalize(x), F.normalize(self.weight))
        predicted_mos = predicted_mos.squeeze(1)  # (batch_size, 1) -> (batch_size,)

        # calculate loss
        loss = self.mse(predicted_mos, label)
        return loss

    def clamping_activation(self, x):
        "clamping activation based on RELU6"
        return F.relu6(x)
