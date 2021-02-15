# Copyright 2021 Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Monkey patch wav2vec2.py Transformer Encoder extract_features()."""

import numpy as np
import torch
import torch.nn.functional as F


def patched_extract_features(self, x, padding_mask=None):

    if padding_mask is not None:
        x[padding_mask] = 0

    x_conv = self.pos_conv(x.transpose(1, 2))
    x_conv = x_conv.transpose(1, 2)
    x += x_conv

    if not self.layer_norm_first:
        x = self.layer_norm(x)

    x = F.dropout(x, p=self.dropout, training=self.training)

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    layer_results = []
    if self.finetune_last_n_layers == 0:
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                layer_results.append(x)
    else:
        with torch.no_grad():
            for i, layer in enumerate(self.layers[: -self.finetune_last_n_layers]):
                dropout_probability = np.random.random()
                if not self.training or (dropout_probability > self.layerdrop):
                    x, z = layer(
                        x, self_attn_padding_mask=padding_mask, need_weights=False
                    )
                    layer_results.append(x)
        for i, layer in enumerate(self.layers[-self.finetune_last_n_layers :]):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                layer_results.append(x)

    # T x B x C -> B x T x C
    x = x.transpose(0, 1)

    return x
