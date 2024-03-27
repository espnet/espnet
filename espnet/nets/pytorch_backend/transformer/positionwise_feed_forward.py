#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""

import torch


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self,
        idim,
        hidden_units,
        dropout_rate,
        activation=torch.nn.ReLU(),
        activation_ckpt=False,
    ):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation
        self.activation_ckpt = activation_ckpt

    def forward(self, x):
        """Forward function."""
        if self.activation_ckpt:
            x = torch.utils.checkpoint.checkpoint(self.w_1, x, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(
                self.activation, x, use_reentrant=False
            )
        else:
            x = self.w_1(x)
            x = self.activation(x)
        return self.w_2(self.dropout(x))
