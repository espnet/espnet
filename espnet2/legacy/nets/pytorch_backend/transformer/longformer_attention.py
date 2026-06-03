#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Roshan Sharma (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Longformer based Local Attention Definition."""

from longformer.longformer import LongformerConfig, LongformerSelfAttention
from torch import nn


class LongformerAttention(nn.Module):
    """Longformer based Local Attention Definition."""

    def __init__(self, config: LongformerConfig, layer_id: int):
        """Compute Longformer based Self-Attention.

        Args:
            config : Longformer attention configuration
            layer_id: Integer representing the layer index
        """
        super().__init__()
        self.attention_window = config.attention_window[layer_id]
        self.attention_layer = LongformerSelfAttention(config, layer_id=layer_id)
        self.attention = None

    def forward(self, query, key, value, mask):
        """Compute Longformer Self-Attention with masking.

        Expects `len(hidden_states)` to be multiple of `attention_window`.
        Padding to `attention_window` happens in :meth:`encoder.forward`
        to avoid redoing the padding on each layer.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, 2*time1-1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        attention_mask = mask.int()
        attention_mask[mask == 0] = -1
        attention_mask[mask == 1] = 0
        output, self.attention = self.attention_layer(
            hidden_states=query,
            attention_mask=attention_mask.unsqueeze(1),
            head_mask=None,
            output_attentions=True,
        )
        return output
