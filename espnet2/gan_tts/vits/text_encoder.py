# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Text encoder module in VITS.

This code is based on https://github.com/jaywalnut310/vits.

"""

import math
from typing import Tuple

import torch

from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class TextEncoder(torch.nn.Module):
    """
        Text encoder module in VITS.

    This module implements a text encoder as described in `Conditional Variational
    Autoencoder with Adversarial Learning for End-to-End Text-to-Speech`_.
    Instead of using a relative positional Transformer, it employs a conformer
    architecture, which incorporates additional convolution layers.

    .. _`Conditional Variational Autoencoder with Adversarial Learning for End-to-End
        Text-to-Speech`: https://arxiv.org/abs/2006.04558

    Attributes:
        attention_dim (int): Dimension of the attention mechanism.
        emb (torch.nn.Embedding): Embedding layer for input indices.
        encoder (Encoder): Encoder module implementing the conformer architecture.
        proj (torch.nn.Conv1d): Convolution layer for projecting outputs.

    Args:
        vocabs (int): Vocabulary size.
        attention_dim (int): Attention dimension.
        attention_heads (int): Number of attention heads.
        linear_units (int): Number of linear units in positionwise layers.
        blocks (int): Number of encoder blocks.
        positionwise_layer_type (str): Type of positionwise layer.
        positionwise_conv_kernel_size (int): Kernel size for positionwise layers.
        positional_encoding_layer_type (str): Type of positional encoding layer.
        self_attention_layer_type (str): Type of self-attention layer.
        activation_type (str): Type of activation function.
        normalize_before (bool): If True, applies LayerNorm before attention.
        use_macaron_style (bool): If True, uses macaron style components.
        use_conformer_conv (bool): If True, uses conformer convolution layers.
        conformer_kernel_size (int): Kernel size for conformer convolution.
        dropout_rate (float): Dropout rate for layers.
        positional_dropout_rate (float): Dropout rate for positional encoding.
        attention_dropout_rate (float): Dropout rate for attention layers.

    Examples:
        >>> encoder = TextEncoder(vocabs=5000)
        >>> input_tensor = torch.randint(0, 5000, (32, 100))  # (B, T_text)
        >>> input_lengths = torch.randint(1, 101, (32,))  # (B,)
        >>> outputs = encoder(input_tensor, input_lengths)
        >>> encoded, mean, logs, mask = outputs

    Raises:
        ValueError: If the input tensor or lengths are not valid.
    """

    def __init__(
        self,
        vocabs: int,
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
        # store for forward
        self.attention_dim = attention_dim

        # define modules
        self.emb = torch.nn.Embedding(vocabs, attention_dim)
        torch.nn.init.normal_(self.emb.weight, 0.0, attention_dim**-0.5)
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
        self.proj = torch.nn.Conv1d(attention_dim, attention_dim * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
                Calculate forward propagation through the TextEncoder module.

        This method takes input tensors, applies an embedding layer, and passes the
        result through an encoder, ultimately returning the encoded hidden
        representation, projected mean, projected scale, and the mask tensor.

        Args:
            x (Tensor): Input index tensor with shape (B, T_text), where B is the batch
                size and T_text is the length of the input text.
            x_lengths (Tensor): Length tensor with shape (B,), representing the actual
                lengths of the sequences in the batch.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: A tuple containing:
                - Tensor: Encoded hidden representation with shape (B, attention_dim, T_text).
                - Tensor: Projected mean tensor with shape (B, attention_dim, T_text).
                - Tensor: Projected scale tensor with shape (B, attention_dim, T_text).
                - Tensor: Mask tensor for input tensor with shape (B, 1, T_text).

        Examples:
            >>> encoder = TextEncoder(vocabs=1000)
            >>> input_tensor = torch.randint(0, 1000, (32, 10))  # Batch of 32, length 10
            >>> input_lengths = torch.tensor([10] * 32)  # All sequences are of length 10
            >>> output = encoder.forward(input_tensor, input_lengths)
            >>> encoded_representation, projected_mean, projected_scale, mask = output
            >>> print(encoded_representation.shape)  # Output: (32, attention_dim, 10)
            >>> print(projected_mean.shape)  # Output: (32, attention_dim, 10)
            >>> print(projected_scale.shape)  # Output: (32, attention_dim, 10)
            >>> print(mask.shape)  # Output: (32, 1, 10)
        """
        x = self.emb(x) * math.sqrt(self.attention_dim)
        x_mask = (
            make_non_pad_mask(x_lengths)
            .to(
                device=x.device,
                dtype=x.dtype,
            )
            .unsqueeze(1)
        )
        # encoder assume the channel last (B, T_text, attention_dim)
        # but mask shape shoud be (B, 1, T_text)
        x, _ = self.encoder(x, x_mask)

        # convert the channel first (B, attention_dim, T_text)
        x = x.transpose(1, 2)
        stats = self.proj(x) * x_mask
        m, logs = stats.split(stats.size(1) // 2, dim=1)

        return x, m, logs, x_mask
