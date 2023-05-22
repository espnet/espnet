# Copyright 2023 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch

from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class PriorDecoder(torch.nn.Module):
    def __init__(
        self,
        out_channels: int = 192 * 2,
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
        global_channels: int = 0,
    ):
        """
        Initialize prior decoder module.

        Args:
            out_channels (int): Output channels of the prior decoder. Defaults to 384.
            attention_dim (int): Dimension of the attention mechanism. Defaults to 192.
            attention_heads (int): Number of attention heads. Defaults to 2.
            linear_units (int): Number of units in the linear layer. Defaults to 768.
            blocks (int): Number of blocks in the encoder. Defaults to 6.
            positionwise_layer_type (str): Type of the positionwise layer.
                                           Defaults to "conv1d".
            positionwise_conv_kernel_size (int): Kernel size of the positionwise
                                                 convolutional layer. Defaults to 3.
            positional_encoding_layer_type (str): Type of positional encoding layer.
                                                  Defaults to "rel_pos".
            self_attention_layer_type (str): Type of self-attention layer.
                                             Defaults to "rel_selfattn".
            activation_type (str): Type of activation. Defaults to "swish".
            normalize_before (bool): Flag for normalization. Defaults to True.
            use_macaron_style (bool): Flag for macaron style. Defaults to False.
            use_conformer_conv (bool): Flag for using conformer convolution.
                                                 Defaults to False.
            conformer_kernel_size (int): Kernel size for conformer convolution.
                                                   Defaults to 7.
            dropout_rate (float): Dropout rate. Defaults to 0.1.
            positional_dropout_rate (float): Dropout rate for positional encoding.
                                                       Defaults to 0.0.
            attention_dropout_rate (float): Dropout rate for attention.
                                                      Defaults to 0.0.
            global_channels (int): Number of global channels. Defaults to 0.
        """
        super().__init__()

        self.prenet = torch.nn.Conv1d(attention_dim + 2, attention_dim, 3, padding=1)
        self.decoder = Encoder(
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
        self.proj = torch.nn.Conv1d(attention_dim, out_channels, 1)

        if global_channels > 0:
            self.conv = torch.nn.Conv1d(global_channels, attention_dim, 1)

    def forward(self, x, x_lengths, g=None):
        """
        Forward pass of the PriorDecoder module.

        Args:
            x (Tensor): Input tensor (B, attention_dim + 2, T).
            x_lengths (Tensor): Length tensor (B,).
            g (Tensor): Tensor for multi-singer. (B, global_channels, 1)

        Returns:
            Tensor: Output tensor (B, out_channels, T).
            Tensor: Output mask tensor (B, 1, T).
        """

        x_mask = (
            make_non_pad_mask(x_lengths)
            .to(
                device=x.device,
                dtype=x.dtype,
            )
            .unsqueeze(1)
        )
        x = self.prenet(x) * x_mask

        # multi-singer
        if g is not None:
            g = torch.detach(g)
            x = x + self.conv(g)

        x = x * x_mask
        x = x.transpose(1, 2)
        x, _ = self.decoder(x, x_mask)
        x = x.transpose(1, 2)

        bn = self.proj(x) * x_mask

        return bn, x_mask
