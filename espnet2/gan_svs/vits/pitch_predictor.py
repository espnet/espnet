# Copyright 2022 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import math
from typing import Optional

import torch
import torch.nn.functional as F
from espnet.nets.pytorch_backend.conformer.encoder import Encoder


class PitchPredictor(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int = 192,
        attention_dim: int = 192,
        attention_heads: int = 2,
        linear_units: int = 768,
        blocks: int = 6,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 3,
        positional_encoding_layer_type: str = "rel_pos",
        self_attention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        normalize_before: bool = True,
        use_macaron_style: bool = False,
        use_conformer_conv: bool = False,
        conformer_kernel_size: int = 7,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.pitch_net = Encoder(
            idim=-1,
            input_layer=None,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=blocks,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            normalize_before=normalize_before,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            macaron_style=use_macaron_style,
            pos_enc_layer_type=positional_encoding_layer_type,
            selfattention_layer_type=self_attention_layer_type,
            activation_type=activation_type,
            use_cnn_module=use_conformer_conv,
            cnn_module_kernel=conformer_kernel_size,
        )
        self.proj = torch.nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, x, x_mask):
        x = x * x_mask
        x = x.transpose(1, 2)
        pitch_embedding, _ = self.pitch_net(x, x_mask)
        pitch_embedding = pitch_embedding.transpose(1, 2)
        pred_pitch = self.proj(pitch_embedding) * x_mask
        return pred_pitch, pitch_embedding
