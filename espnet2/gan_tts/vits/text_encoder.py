# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Text encoder of VITS.

This code is based on the official implementation:
- https://github.com/jaywalnut310/vits

"""

import math

import torch

from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class TextEncoder(torch.nn.Module):
    """Text encoder module."""

    def __init__(
        self,
        vocabs,
        attention_dim=192,
        attention_heads=2,
        linear_units=768,
        blocks=6,
        positionwise_layer_type="conv1d",
        positionwise_conv_kernel_size=3,
        positional_encoding_layer_type="rel_pos",
        self_attention_layer_type="rel_selfattn",
        activation_type="swish",
        normalize_before=False,
        use_macaron_style=False,
        use_conformer_conv=False,
        conformer_kernel_size=7,
        dropout_rate=0.1,
        positional_dropout_rate=0.0,
        attention_dropout_rate=0.0,
    ):
        """Initialize TextEncoder module.

        Args:
            vocabs (int): Vocabulary size.
            attention_dim (int): Attention dimension.
            attention_heads (int): Number of attention heads.
            linear_units (int): Number of linear units of positionwise layers.
            blocks (int): Number of encoder blocks.
            positionwise_layer_type (str): Positionwise layer type.
            positionwise_conv_kernel_size (int): Positionwise layer's kernel size.
            positional_encoding_layer_type (str): Positional encoding layer type.
            self_attention_layer_type (str): Self-attention layer type.
            activation_type (str): Activation function type.
            normalize_before (bool): Whether to apply LayerNorm before attention.
            use_macaron_style (bool): Whether to use macaron style components.
            use_conformer_conv (bool): Whether to use conformer conv layers.
            conformer_kernel_size (int): Conformer's conv kernel size.
            dropout_rate (float): Dropout rate.
            positional_dropout_rate (float): Dropout rate for positional encoding.
            attention_dropout_rate (float): Dropout rate for attention.

        """
        super().__init__()
        self.attention_dim = attention_dim

        # define modules
        self.input_emb = torch.nn.Embedding(vocabs, attention_dim)
        torch.nn.init.normal_(self.input_emb.weight, 0.0, attention_dim ** -0.5)

        # NOTE(kan-bayashi): We use conformer encoder instead of transformer encoder
        self.encoder = Encoder(
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

        self.output_conv = torch.nn.Conv1d(attention_dim, attention_dim * 2, 1)

    def forward(self, x, x_lengths):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input index tensor (B, T).
            x_lengths (Tensor): Length tensor (B,).

        Returns:
            Tensor: Encoded hidden representation (B, attention_dim, T).
            Tensor: VAE mean tensor (B, attention_dim, T).
            Tensor: VAE scale tensor (B, attention_dim, T).
            Tensor: Mask tensor for input tensor (B, 1, T).

        """
        x = self.input_emb(x) * math.sqrt(self.attention_dim)
        x_mask = (
            make_non_pad_mask(x_lengths)
            .to(
                device=x.device,
                dtype=x.dtype,
            )
            .unsqueeze(1)
        )
        x, _ = self.encoder(x, x_mask)
        x = x.transpose(1, 2)  # (B, attention_dim, T)
        stats = self.output_conv(x) * x_mask
        m, logs = torch.split(stats, self.attention_dim, dim=1)
        return x, m, logs, x_mask
