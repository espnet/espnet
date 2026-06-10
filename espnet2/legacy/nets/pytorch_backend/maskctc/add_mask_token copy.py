#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Waseda University (Yosuke Higuchi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Token masking module for Masked LM."""

import numpy
import torch

from espnet2.legacy.nets.pytorch_backend.nets_utils import pad_list


def mask_uniform(ys_pad, mask_token, eos, ignore_id):
    """Replace random tokens with <mask> label and add <eos> label.

    The number of <mask> is chosen from a uniform distribution
    between one and the target sequence's length.
    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int mask_token: index of <mask>
    :param int eos: index of <eos>
    :param int ignore_id: index of padding
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    """

    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_out = [y.new(y.size()).fill_(ignore_id) for y in ys]
    ys_in = [y.clone() for y in ys]

    for i in range(len(ys)):
        num_samples = numpy.random.randint(1, len(ys[i]) + 1)
        idx = numpy.random.choice(len(ys[i]), num_samples)
        ys_in[i][idx] = mask_token
        ys_out[i][idx] = ys[i][idx]

    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def apply_mask(ys_pad, mask_token, eos, ignore_id, num_hypotheses=0):
    """Replace random tokens with <mask> label and add <eos> label."""
    assert num_hypotheses > 0, "num_hypotheses must be > 0"
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [y.clone() for y in ys]
    ys_out = [y.clone() for y in ys]
    for i in range(len(ys)):
        num_samples = numpy.random.randint(1, len(ys[i]) + 1)
        idx = numpy.random.choice(len(ys[i]), num_samples)
        ys_in[i][idx] = mask_token
        ys_out[i][idx] = ys[i][idx]

    ys_in_pad = pad_list(ys_in, eos)
    ys_out_pad = pad_list(ys_out, ignore_id)
    mask_idx = ys_in_pad == mask_token
    length = ys_in_pad.size(1)
    ys_in_pad = ys_in_pad.unsqueeze(1).repeat(1, num_hypotheses, 1).reshape(-1, length)
    ys_out_pad = (
        ys_out_pad.unsqueeze(1).repeat(1, num_hypotheses, 1).reshape(-1, length)
    )
    mask_idx = mask_idx.unsqueeze(1).repeat(1, num_hypotheses, 1).reshape(-1, length)
    noise = torch.randint(0, eos, ys_in_pad.shape, device=ys_in_pad.device)

    # replace by random
    prob = torch.rand(ys_in_pad.shape).to(ys_pad.device)
    # rep_idx = (prob < 0.5) & mask_idx
    rep_idx = prob < 0.5
    ys_in_pad[rep_idx] = noise[rep_idx]

    # replace by ground truth
    # prob = torch.rand(ys_in_pad.shape).to(ys_pad.device)
    # rep_idx = (prob < 0.2) & mask_idx
    # ys_in_pad[rep_idx] = ys_out_pad[rep_idx]

    return ys_in_pad, ys_out_pad
