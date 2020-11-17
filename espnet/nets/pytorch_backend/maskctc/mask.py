#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Waseda University (Yosuke Higuchi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Attention masking module for Masked LM."""


def square_mask(ys_in_pad, ignore_id):
    """Create attention mask to avoid attending on padding tokens.

    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int ignore_id: index of padding
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor (B, Lmax, Lmax)
    """
    ys_mask = (ys_in_pad != ignore_id).unsqueeze(-2)
    ymax = ys_mask.size(-1)
    ys_mask_tmp = ys_mask.transpose(1, 2).repeat(1, 1, ymax)
    ys_mask = ys_mask.repeat(1, ymax, 1) & ys_mask_tmp

    return ys_mask
