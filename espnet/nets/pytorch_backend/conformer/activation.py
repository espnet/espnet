#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Activation funcitons for Conformer."""

import torch
import torch.nn as nn


class Swish(nn.Module):
    """Construct an Swish object."""
    def forward(self, x):
        return x * torch.sigmoid(x)


def get_activation(act):
    """Return activation function."""
    activation_funcs = {
        "hardtanh": nn.Hardtanh,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "swish": Swish,
    }

    return activation_funcs[act]()
