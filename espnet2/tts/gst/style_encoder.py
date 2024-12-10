# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Style encoder of GST-Tacotron."""

from typing import Sequence

import torch
from typeguard import typechecked

from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention as BaseMultiHeadedAttention,
)


class StyleEncoder(torch.nn.Module):
    """
        Style encoder of GST-Tacotron.

    This module implements the style encoder introduced in
    `Style Tokens: Unsupervised Style Modeling, Control and Transfer in
    End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in
       End-to-End Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Attributes:
        ref_enc (ReferenceEncoder): Reference encoder module for feature extraction.
        stl (StyleTokenLayer): Style token layer for generating style embeddings.

    Args:
        idim (int, optional): Dimension of the input mel-spectrogram. Default is 80.
        gst_tokens (int, optional): The number of GST embeddings. Default is 10.
        gst_token_dim (int, optional): Dimension of each GST embedding. Default is 256.
        gst_heads (int, optional): The number of heads in GST multihead attention.
            Default is 4.
        conv_layers (int, optional): The number of conv layers in the reference
            encoder. Default is 6.
        conv_chans_list (Sequence[int], optional): List of the number of channels
            of conv layers in the reference encoder. Default is (32, 32, 64, 64,
            128, 128).
        conv_kernel_size (int, optional): Kernel size of conv layers in the
            reference encoder. Default is 3.
        conv_stride (int, optional): Stride size of conv layers in the reference
            encoder. Default is 2.
        gru_layers (int, optional): The number of GRU layers in the reference
            encoder. Default is 1.
        gru_units (int, optional): The number of GRU units in the reference
            encoder. Default is 128.

    Todo:
        * Support manual weight specification in inference.

    Examples:
        # Initialize the StyleEncoder with default parameters
        style_encoder = StyleEncoder()

        # Forward pass with dummy input
        dummy_input = torch.randn(4, 100, 80)  # (B, Lmax, idim)
        style_embeddings = style_encoder(dummy_input)
        print(style_embeddings.shape)  # Output shape should be (B, gst_token_dim)
    """

    @typechecked
    def __init__(
        self,
        idim: int = 80,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
    ):
        """Initilize global style encoder module."""
        super(StyleEncoder, self).__init__()

        self.ref_enc = ReferenceEncoder(
            idim=idim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
        )
        self.stl = StyleTokenLayer(
            ref_embed_dim=gru_units,
            gst_tokens=gst_tokens,
            gst_token_dim=gst_token_dim,
            gst_heads=gst_heads,
        )

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        """
            Calculate forward propagation.

        This method computes the forward pass of the style encoder by taking
        the input speech features and producing style token embeddings.

        Args:
            speech (Tensor): Batch of padded target features with shape
                (B, Lmax, odim), where B is the batch size, Lmax is the maximum
                length of the sequence, and odim is the dimension of the output
                features.

        Returns:
            Tensor: Style token embeddings with shape (B, token_dim), where
                token_dim is the dimension of the generated style tokens.

        Examples:
            >>> style_encoder = StyleEncoder()
            >>> input_speech = torch.randn(16, 100, 80)  # Example input
            >>> output_style_embs = style_encoder(input_speech)
            >>> print(output_style_embs.shape)  # Should print: torch.Size([16, 256])
        """
        ref_embs = self.ref_enc(speech)
        style_embs = self.stl(ref_embs)

        return style_embs


class ReferenceEncoder(torch.nn.Module):
    """
        Reference encoder module.

    This module is the reference encoder introduced in `Style Tokens: Unsupervised
    Style Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
       Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Attributes:
        conv_layers (int): The number of conv layers in the reference encoder.
        kernel_size (int): Kernel size of conv layers in the reference encoder.
        stride (int): Stride size of conv layers in the reference encoder.
        padding (int): Padding size used in convolution layers.

    Args:
        idim (int, optional): Dimension of the input mel-spectrogram.
        conv_layers (int, optional): The number of conv layers in the reference encoder.
        conv_chans_list (Sequence[int], optional):
            List of the number of channels of conv layers in the reference encoder.
        conv_kernel_size (int, optional):
            Kernel size of conv layers in the reference encoder.
        conv_stride (int, optional):
            Stride size of conv layers in the reference encoder.
        gru_layers (int, optional): The number of GRU layers in the reference encoder.
        gru_units (int, optional): The number of GRU units in the reference encoder.

    Examples:
        >>> encoder = ReferenceEncoder(idim=80, conv_layers=6)
        >>> input_tensor = torch.randn(16, 100, 80)  # (B, Lmax, idim)
        >>> output = encoder(input_tensor)
        >>> print(output.shape)  # Output shape: (B, gru_units)
    """

    @typechecked
    def __init__(
        self,
        idim=80,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
    ):
        """Initilize reference encoder module."""
        super(ReferenceEncoder, self).__init__()

        # check hyperparameters are valid
        assert conv_kernel_size % 2 == 1, "kernel size must be odd."
        assert (
            len(conv_chans_list) == conv_layers
        ), "the number of conv layers and length of channels list must be the same."

        convs = []
        padding = (conv_kernel_size - 1) // 2
        for i in range(conv_layers):
            conv_in_chans = 1 if i == 0 else conv_chans_list[i - 1]
            conv_out_chans = conv_chans_list[i]
            convs += [
                torch.nn.Conv2d(
                    conv_in_chans,
                    conv_out_chans,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=padding,
                    # Do not use bias due to the following batch norm
                    bias=False,
                ),
                torch.nn.BatchNorm2d(conv_out_chans),
                torch.nn.ReLU(inplace=True),
            ]
        self.convs = torch.nn.Sequential(*convs)

        self.conv_layers = conv_layers
        self.kernel_size = conv_kernel_size
        self.stride = conv_stride
        self.padding = padding

        # get the number of GRU input units
        gru_in_units = idim
        for i in range(conv_layers):
            gru_in_units = (
                gru_in_units - conv_kernel_size + 2 * padding
            ) // conv_stride + 1
        gru_in_units *= conv_out_chans
        self.gru = torch.nn.GRU(gru_in_units, gru_units, gru_layers, batch_first=True)

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        """
        Calculate forward propagation.

        This method performs forward propagation through the style encoder,
        taking a batch of padded target features and returning style token
        embeddings.

        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, odim).

        Returns:
            Tensor: Style token embeddings (B, token_dim).

        Examples:
            >>> encoder = StyleEncoder()
            >>> speech_input = torch.randn(8, 100, 80)  # Example input
            >>> style_embeddings = encoder.forward(speech_input)
            >>> print(style_embeddings.shape)  # Should output: torch.Size([8, 256])
        """
        batch_size = speech.size(0)
        xs = speech.unsqueeze(1)  # (B, 1, Lmax, idim)
        hs = self.convs(xs).transpose(1, 2)  # (B, Lmax', conv_out_chans, idim')
        # NOTE(kan-bayashi): We need to care the length?
        time_length = hs.size(1)
        hs = hs.contiguous().view(batch_size, time_length, -1)  # (B, Lmax', gru_units)
        self.gru.flatten_parameters()
        _, ref_embs = self.gru(hs)  # (gru_layers, batch_size, gru_units)
        ref_embs = ref_embs[-1]  # (batch_size, gru_units)

        return ref_embs


class StyleTokenLayer(torch.nn.Module):
    """
    Style token layer module.

    This module is a style token layer introduced in `Style Tokens: Unsupervised
    Style Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Args:
        ref_embed_dim (int, optional): Dimension of the input reference embedding.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_token_dim (int, optional): Dimension of each GST embedding.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        dropout_rate (float, optional): Dropout rate in multi-head attention.

    Returns:
        Tensor: Style token embeddings (B, gst_token_dim).

    Examples:
        >>> layer = StyleTokenLayer(ref_embed_dim=128, gst_tokens=10,
        ...                          gst_token_dim=256, gst_heads=4)
        >>> ref_embs = torch.randn(32, 128)  # Batch of reference embeddings
        >>> style_embs = layer(ref_embs)
        >>> style_embs.shape
        torch.Size([32, 256])  # Shape of style token embeddings
    """

    @typechecked
    def __init__(
        self,
        ref_embed_dim: int = 128,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        dropout_rate: float = 0.0,
    ):
        """Initilize style token layer module."""
        super(StyleTokenLayer, self).__init__()

        gst_embs = torch.randn(gst_tokens, gst_token_dim // gst_heads)
        self.register_parameter("gst_embs", torch.nn.Parameter(gst_embs))
        self.mha = MultiHeadedAttention(
            q_dim=ref_embed_dim,
            k_dim=gst_token_dim // gst_heads,
            v_dim=gst_token_dim // gst_heads,
            n_head=gst_heads,
            n_feat=gst_token_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, ref_embs: torch.Tensor) -> torch.Tensor:
        """
        Calculate forward propagation.

        This method computes the forward pass of the StyleEncoder module. It takes
        a batch of padded target features and processes them to extract style token
        embeddings.

        Args:
            speech (Tensor): Batch of padded target features with shape
                (B, Lmax, odim), where B is the batch size, Lmax is the maximum
                sequence length, and odim is the dimension of the output.

        Returns:
            Tensor: Style token embeddings with shape (B, token_dim), where token_dim
                is the dimension of the style token embeddings.

        Examples:
            >>> style_encoder = StyleEncoder()
            >>> input_tensor = torch.randn(32, 100, 80)  # Example input
            >>> output = style_encoder(input_tensor)
            >>> print(output.shape)
            torch.Size([32, 256])  # Example output shape
        """
        batch_size = ref_embs.size(0)
        # (num_tokens, token_dim) -> (batch_size, num_tokens, token_dim)
        gst_embs = torch.tanh(self.gst_embs).unsqueeze(0).expand(batch_size, -1, -1)
        # NOTE(kan-bayashi): Shoule we apply Tanh?
        ref_embs = ref_embs.unsqueeze(1)  # (batch_size, 1 ,ref_embed_dim)
        style_embs = self.mha(ref_embs, gst_embs, gst_embs, None)

        return style_embs.squeeze(1)


class MultiHeadedAttention(BaseMultiHeadedAttention):
    """
    Multi head attention module with different input dimension.

    This module extends the base multi-headed attention mechanism to support
    different input dimensions for queries, keys, and values. It is used in
    various architectures, including transformers and attention-based models.

    Args:
        q_dim (int): Dimension of the input queries.
        k_dim (int): Dimension of the input keys.
        v_dim (int): Dimension of the input values.
        n_head (int): Number of attention heads.
        n_feat (int): Total dimension of the input features.
        dropout_rate (float, optional): Dropout rate for attention weights (default: 0.0).

    Raises:
        AssertionError: If `n_feat` is not divisible by `n_head`.

    Examples:
        >>> attention_layer = MultiHeadedAttention(q_dim=64, k_dim=64, v_dim=64,
        ...                                        n_head=8, n_feat=64)
        >>> query = torch.rand(10, 20, 64)  # (batch_size, seq_length, q_dim)
        >>> key = torch.rand(10, 15, 64)    # (batch_size, seq_length, k_dim)
        >>> value = torch.rand(10, 15, 64)  # (batch_size, seq_length, v_dim)
        >>> output = attention_layer(query, key, value)
        >>> print(output.shape)  # Output shape should be (10, 20, 64)

    Note:
        The `d_v` (dimension of values) is assumed to be equal to `d_k`
        (dimension of keys).
    """

    def __init__(self, q_dim, k_dim, v_dim, n_head, n_feat, dropout_rate=0.0):
        """Initialize multi head attention module."""
        # NOTE(kan-bayashi): Do not use super().__init__() here since we want to
        #   overwrite BaseMultiHeadedAttention.__init__() method.
        torch.nn.Module.__init__(self)
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = torch.nn.Linear(q_dim, n_feat)
        self.linear_k = torch.nn.Linear(k_dim, n_feat)
        self.linear_v = torch.nn.Linear(v_dim, n_feat)
        self.linear_out = torch.nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.use_flash_attn = False
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()
