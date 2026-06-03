#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch

from espnet2.legacy.nets.pytorch_backend.fastspeech.length_regulator import (
    LengthRegulator,
)
from espnet2.legacy.nets.pytorch_backend.nets_utils import pad_list


def test_length_regulator():
    # prepare inputs
    idim = 5
    ilens = [10, 5, 3]
    xs = pad_list([torch.randn((ilen, idim)) for ilen in ilens], 0.0)
    ds = pad_list([torch.arange(ilen) for ilen in ilens], 0)

    # test with non-zero durations
    length_regulator = LengthRegulator()
    xs_expand = length_regulator(xs, ds)
    assert int(xs_expand.shape[1]) == int(ds.sum(dim=-1).max())

    # test with duration including zero
    ds[:, 2] = 0
    xs_expand = length_regulator(xs, ds)
    assert int(xs_expand.shape[1]) == int(ds.sum(dim=-1).max())
