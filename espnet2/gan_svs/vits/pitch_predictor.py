# Copyright 2022 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch

from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class Decoder(torch.nn.Module):
    """Pitch or Mel decoder module in VISinger 2."""

    def __init__(
        self,
        out_channels: int = 192,
        attention_dim: int = 192,
        attention_heads: int = 2,
        linear_units: int = 768,
        blocks: int = 6,
        pw_layer_type: str = "conv1d",
        pw_conv_kernel_size: int = 3,
        pos_enc_layer_type: str = "rel_pos",
        self_attention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        normalize_before: bool = True,
        use_macaron_style: bool = False,
        use_conformer_conv: bool = False,
        conformer_kernel_size: int = 7,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        global_channels: int = -1,
    ):
        """
        Args:
            out_channels (int): The output dimension of the module.
            attention_dim (int): The dimension of the attention mechanism.
            attention_heads (int): The number of attention heads.
            linear_units (int): The number of units in the linear layer.
            blocks (int): The number of encoder blocks.
            pw_layer_type (str): The type of position-wise layer to use.
            pw_conv_kernel_size (int): The kernel size of the position-wise
                                       convolutional layer.
            pos_enc_layer_type (str): The type of positional encoding layer to use.
            self_attention_layer_type (str): The type of self-attention layer to use.
            activation_type (str): The type of activation function to use.
            normalize_before (bool): Whether to normalize the data before the
                                     position-wise layer or after.
            use_macaron_style (bool): Whether to use the macaron style or not.
            use_conformer_conv (bool): Whether to use Conformer style conv or not.
            conformer_kernel_size (int): The kernel size of the conformer
                                         convolutional layer.
            dropout_rate (float): The dropout rate to use.
            positional_dropout_rate (float): The positional dropout rate to use.
            attention_dropout_rate (float): The attention dropout rate to use.
            global_channels (int): The number of channels to use for global
                                   conditioning.
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
            positionwise_layer_type=pw_layer_type,
            positionwise_conv_kernel_size=pw_conv_kernel_size,
            macaron_style=use_macaron_style,
            pos_enc_layer_type=pos_enc_layer_type,
            selfattention_layer_type=self_attention_layer_type,
            activation_type=activation_type,
            use_cnn_module=use_conformer_conv,
            cnn_module_kernel=conformer_kernel_size,
        )
        self.proj = torch.nn.Conv1d(attention_dim, out_channels, 1)

        if global_channels > 0:
            self.global_conv = torch.nn.Conv1d(global_channels, attention_dim, 1)

    def forward(self, x, x_lengths, g=None):
        """
        Forward pass of the Decoder.

        Args:
            x (Tensor): Input tensor (B, 2 + attention_dim, T).
            x_lengths (Tensor): Length tensor (B,).
            g (Tensor, optional): Global conditioning tensor (B, global_channels, 1).

        Returns:
            Tensor: Output tensor (B, 1, T).
            Tensor: Output mask (B, 1, T).
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

        if g is not None:
            x = x + self.global_conv(g)

        x = x.transpose(1, 2)
        x, _ = self.decoder(x, x_mask)
        x = x.transpose(1, 2)

        x = self.proj(x) * x_mask

        return x, x_mask
