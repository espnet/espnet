# Copyright 2022 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch

from espnet.nets.pytorch_backend.conformer.encoder import Encoder


class PhonemePredictor(torch.nn.Module):
    """
    Phoneme Predictor module in VISinger.
    """

    def __init__(
        self,
        vocabs: int,
        hidden_channels: int = 192,
        attention_dim: int = 192,
        attention_heads: int = 2,
        linear_units: int = 768,
        blocks: int = 2,
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
        """
        Initialize PhonemePredictor module.

        Args:
            vocabs (int): The number of vocabulary.
            hidden_channels (int): The number of hidden channels.
            attention_dim (int): The number of attention dimension.
            attention_heads (int): The number of attention heads.
            linear_units (int): The number of linear units.
            blocks (int): The number of encoder blocks.
            positionwise_layer_type (str): The type of position-wise layer.
            positionwise_conv_kernel_size (int): The size of position-wise
                                                 convolution kernel.
            positional_encoding_layer_type (str): The type of positional encoding layer.
            self_attention_layer_type (str): The type of self-attention layer.
            activation_type (str): The type of activation function.
            normalize_before (bool): Whether to apply normalization before the
                                     position-wise layer or not.
            use_macaron_style (bool): Whether to use macaron style or not.
            use_conformer_conv (bool): Whether to use Conformer convolution or not.
            conformer_kernel_size (int): The size of Conformer kernel.
            dropout_rate (float): The dropout rate.
            positional_dropout_rate (float): The dropout rate for positional encoding.
            attention_dropout_rate (float): The dropout rate for attention.
        """

        super().__init__()
        self.phoneme_predictor = Encoder(
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
        self.linear1 = torch.nn.Linear(hidden_channels, vocabs)

    def forward(self, x, x_mask):
        """
        Perform forward propagation.

        Args:
            x (Tensor): The input tensor of shape (B, dim, length).
            x_mask (Tensor): The mask tensor for the input tensor of shape (B, length).

        Returns:
            Tensor: The predicted phoneme tensor of shape (length, B, vocab_size).

        """

        x = x * x_mask
        x = x.transpose(1, 2)
        phoneme_embedding, _ = self.phoneme_predictor(x, x_mask)
        phoneme_embedding = phoneme_embedding.transpose(1, 2)
        x1 = self.linear1(phoneme_embedding.transpose(1, 2))
        x1 = x1.log_softmax(2)

        return x1.transpose(0, 1)
