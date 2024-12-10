#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Derived from OpenAI Whisper model file:
# https://github.com/openai/whisper/blob/main/whisper/model.py

# (1) A concise implementation of Transformer Decoder-Only Architecture.
# (2) This is the built-in implementation from ESPnet. Users can also
# adopt HuggingFace transformer models besides this.
# (3) We intentionally keep this implementation simple and will not keep
# many configuration choices for it.
# (4) Similar to HuggingFace models, this module contains stacked Transformer
# layers and positional embeddings, but no embedding table and lm_head.
# (5) Attention is based on Pytorch built-in flash attention. Please use
# compatible Pytorch versions.


from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm(nn.LayerNorm):
    """
        LayerNorm is a custom implementation of PyTorch's LayerNorm that ensures the
    input tensor is cast to float before applying layer normalization. This class
    inherits from the nn.LayerNorm module.

    Attributes:
        normalized_shape (tuple): The shape of the input tensor for normalization.
        eps (float): A value added to the denominator for numerical stability.
        elementwise_affine (bool): Whether to include learnable parameters for
            scaling and shifting.

    Args:
        normalized_shape (int or tuple): Input shape for normalization.
        eps (float, optional): A value added to the denominator for numerical
            stability (default: 1e-5).
        elementwise_affine (bool, optional): If True, learnable parameters are
            added (default: True).

    Returns:
        Tensor: The layer-normalized output tensor.

    Examples:
        >>> layer_norm = LayerNorm(normalized_shape=10)
        >>> input_tensor = torch.randn(5, 10)
        >>> output_tensor = layer_norm(input_tensor)
        >>> output_tensor.shape
        torch.Size([5, 10])
    """

    def forward(self, x: Tensor) -> Tensor:
        """
                Applies Layer Normalization to the input tensor.

        This method overrides the forward method of the nn.LayerNorm class to ensure
        that the input tensor is converted to float before applying layer normalization,
        and then the output is cast back to the original dtype of the input tensor.

        Args:
            x (Tensor): The input tensor to be normalized. The tensor should be of
                type float, but the method will convert it to float for processing.

        Returns:
            Tensor: The layer-normalized tensor, returned in the original dtype of
                the input tensor.

        Examples:
            >>> layer_norm = LayerNorm(normalized_shape=10)
            >>> input_tensor = torch.randn(2, 10)
            >>> output_tensor = layer_norm(input_tensor)
            >>> output_tensor.dtype  # Should match the dtype of input_tensor
            torch.float32
        """
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    """
    Linear layer with automatic dtype adjustment for inputs.

    This class extends the PyTorch `nn.Linear` layer to ensure that the
    weights and biases are cast to the appropriate dtype of the input
    tensor `x` during the forward pass. This is particularly useful
    when working with mixed-precision training.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to False, the layer will not
            learn an additive bias. Default is True.

    Returns:
        Tensor: The output tensor after applying the linear transformation.

    Examples:
        >>> linear_layer = Linear(10, 5)
        >>> input_tensor = torch.randn(2, 10, dtype=torch.float32)
        >>> output_tensor = linear_layer(input_tensor)
        >>> output_tensor.shape
        torch.Size([2, 5])
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the forward pass for the linear layer.

        This method applies a linear transformation to the input tensor `x` using
        the layer's weights and bias (if provided). The weights are cast to the
        same data type as the input tensor, ensuring compatibility for operations.

        Args:
            x (Tensor): The input tensor of shape (N, *, in_features) where N is
                        the batch size and * represents any number of additional
                        dimensions.

        Returns:
            Tensor: The output tensor of shape (N, *, out_features) where
                    out_features is the number of features in the output. The
                    output tensor will have the same data type as the input tensor.

        Examples:
            >>> linear_layer = Linear(5, 3)
            >>> input_tensor = torch.randn(10, 5)
            >>> output_tensor = linear_layer(input_tensor)
            >>> output_tensor.shape
            torch.Size([10, 3])

        Note:
            The bias is optional; if not provided, the layer will perform
            the linear transformation without an additive bias term.
        """
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class MultiHeadAttention(nn.Module):
    """
        MultiHeadAttention is a module that implements multi-head attention as part of
    the Transformer architecture. This implementation supports both self-attention
    and cross-attention mechanisms and utilizes PyTorch's built-in flash attention
    for efficiency.

    Attributes:
        n_head (int): The number of attention heads.
        query (Linear): Linear layer for the query projection.
        key (Linear): Linear layer for the key projection.
        value (Linear): Linear layer for the value projection.
        out (Linear): Linear layer for the output projection.
        causal (bool): Indicates whether the attention is causal.

    Args:
        n_state (int): The dimension of the input state.
        n_head (int): The number of attention heads.
        causal (bool, optional): Whether to use causal attention. Defaults to False.

    Raises:
        ValueError: If the number of heads does not divide the state dimension or if
                     the PyTorch version is incompatible with flash attention.

    Examples:
        >>> attention = MultiHeadAttention(n_state=64, n_head=8)
        >>> x = torch.rand(10, 20, 64)  # (batch_size, sequence_length, n_state)
        >>> output = attention(x)
        >>> print(output.shape)  # (10, 20, 64)

        >>> x_a = torch.rand(10, 20, 64)
        >>> x_b = torch.rand(10, 30, 64)
        >>> output = attention(x_a, xa=x_b)
        >>> print(output.shape)  # (10, 20, 64)
    """

    def __init__(self, n_state: int, n_head: int, causal: bool = False):
        super().__init__()
        assert n_state % n_head == 0
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.causal = causal

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ValueError("Install torch 2.0.1+ to support Flash Attention")

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        """
            Computes the forward pass of the MultiHeadAttention module.

        This method applies multi-head attention to the input tensor `x`, and can
        optionally take in a second input tensor `xa` for cross-attention, a mask
        for attention scores, and a key-value cache for efficiency during
        autoregressive decoding.

        Args:
            x (Tensor): The input tensor of shape (batch_size, seq_length, n_state).
            xa (Optional[Tensor]): An optional tensor for cross-attention, with the
                same shape as `x`. Default is None.
            mask (Optional[Tensor]): An optional mask tensor of shape (batch_size,
                n_head, seq_length, seq_length) to prevent attending to certain
                positions. Default is None.
            kv_cache (Optional[dict]): A cache for key and value tensors, used for
                optimizing cross-attention. Default is None.

        Returns:
            Tensor: The output tensor of shape (batch_size, seq_length, n_state).

        Raises:
            ValueError: If `mask` is provided while `causal` is set to True.

        Examples:
            >>> mha = MultiHeadAttention(n_state=64, n_head=8, causal=True)
            >>> x = torch.rand(10, 20, 64)  # (batch_size, seq_length, n_state)
            >>> output = mha(x)
            >>> output.shape
            torch.Size([10, 20, 64])

            >>> xa = torch.rand(10, 30, 64)  # Cross-attention input
            >>> mask = torch.ones(10, 8, 20, 20)  # Example mask
            >>> output = mha(x, xa=xa, mask=mask)
            >>> output.shape
            torch.Size([10, 20, 64])
        """
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None)
            #   will prepend the cached kv tensors;
            # otherwise, perform key/value projections
            #   for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once
            # then reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv = self.qkv_attention(q, k, v, mask)

        return self.out(wv)

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        """
            Computes the query-key-value attention mechanism.

        This method applies the multi-head attention mechanism by computing the
        scaled dot-product attention between the query (q), key (k), and value (v)
        tensors. The attention is calculated with consideration for causal masking,
        if applicable.

        Args:
            q (Tensor): The query tensor of shape (batch_size, seq_len, n_state).
            k (Tensor): The key tensor of shape (batch_size, seq_len, n_state).
            v (Tensor): The value tensor of shape (batch_size, seq_len, n_state).
            mask (Optional[Tensor]): An optional tensor for masking of shape
                (batch_size, n_head, seq_len, seq_len). Defaults to None.

        Returns:
            Tensor: The output tensor after applying the attention mechanism,
            with shape (batch_size, seq_len, n_state).

        Raises:
            ValueError: If causal attention is requested but a mask is provided.

        Examples:
            >>> attention_layer = MultiHeadAttention(n_state=64, n_head=8)
            >>> q = torch.rand(2, 10, 64)  # (batch_size, seq_len, n_state)
            >>> k = torch.rand(2, 10, 64)
            >>> v = torch.rand(2, 10, 64)
            >>> output = attention_layer.qkv_attention(q, k, v)
            >>> output.shape
            torch.Size([2, 10, 64])

        Note:
            The method uses PyTorch's built-in scaled dot-product attention
            and assumes that the input tensors are of appropriate shapes.

        Todo:
            Add support for additional attention mechanisms in the future.
        """
        if self.causal and mask is not None:
            raise ValueError("mask is not allowed when the attention is causal")

        if self.causal and q.size(1) == k.size(1):
            causal = True
        else:
            causal = False

        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        wv = (
            F.scaled_dot_product_attention(q, k, v, mask, is_causal=causal)
            .permute(0, 2, 1, 3)
            .flatten(start_dim=2)
        )

        return wv


class ResidualAttentionBlock(nn.Module):
    """
        A residual attention block that combines multi-head attention and feedforward
    network with residual connections and layer normalization.

    This block is designed to facilitate the construction of Transformer-like
    architectures, enabling both self-attention and cross-attention mechanisms.
    It applies layer normalization and residual connections to enhance training
    stability and model performance.

    Attributes:
        attn (MultiHeadAttention): The multi-head self-attention layer.
        attn_ln (LayerNorm): Layer normalization applied to the self-attention output.
        cross_attn (Optional[MultiHeadAttention]): The multi-head cross-attention layer
            if cross_attention is enabled.
        cross_attn_ln (Optional[LayerNorm]): Layer normalization applied to the
            cross-attention output if cross_attention is enabled.
        mlp (Sequential): A feedforward network consisting of two linear layers with
            a GELU activation in between.
        mlp_ln (LayerNorm): Layer normalization applied to the output of the MLP.

    Args:
        n_state (int): The dimensionality of the input and output features.
        n_head (int): The number of attention heads.
        cross_attention (bool, optional): Whether to enable cross-attention. Defaults
            to False.
        causal (bool, optional): Whether to enable causal attention. Defaults to
            False.

    Returns:
        Tensor: The output tensor after applying the attention and feedforward layers.

    Examples:
        >>> block = ResidualAttentionBlock(n_state=256, n_head=8)
        >>> input_tensor = torch.randn(10, 32, 256)  # (batch_size, seq_len, n_state)
        >>> output = block(input_tensor)
        >>> print(output.shape)
        torch.Size([10, 32, 256])

    Note:
        This module is designed to be used as a building block for Transformer
        architectures and should be integrated within a larger model for practical
        applications.
    """

    def __init__(
        self,
        n_state: int,
        n_head: int,
        cross_attention: bool = False,
        causal: bool = False,
    ):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head, causal=causal)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        """
            Residual Attention Block implementing multi-head attention with residual
        connections and layer normalization.

        This module includes a multi-head self-attention mechanism, an optional
        cross-attention mechanism, and a feed-forward neural network (MLP). The
        attention mechanism can be causal or non-causal based on the configuration.
        Residual connections are applied to facilitate training deep networks.

        Attributes:
            attn (MultiHeadAttention): The multi-head self-attention layer.
            attn_ln (LayerNorm): Layer normalization applied to the input of the
                attention layer.
            cross_attn (Optional[MultiHeadAttention]): The multi-head cross-attention
                layer, if enabled.
            cross_attn_ln (Optional[LayerNorm]): Layer normalization for the cross-attention
                input, if enabled.
            mlp (Sequential): A feed-forward network consisting of two linear layers
                with GELU activation in between.
            mlp_ln (LayerNorm): Layer normalization applied to the input of the MLP.

        Args:
            n_state (int): Dimensionality of the input and output features.
            n_head (int): Number of attention heads.
            cross_attention (bool, optional): Whether to enable cross-attention.
                Defaults to False.
            causal (bool, optional): Whether to use causal attention. Defaults to
                False.

        Returns:
            Tensor: The output tensor after applying the attention and MLP layers.

        Examples:
            >>> block = ResidualAttentionBlock(n_state=256, n_head=8)
            >>> input_tensor = torch.rand(10, 20, 256)  # (batch_size, seq_len, n_state)
            >>> output_tensor = block(input_tensor)
            >>> output_tensor.shape
            torch.Size([10, 20, 256])

        Raises:
            ValueError: If the cross-attention mechanism is used but no input tensor
                for cross-attention is provided.

        Note:
            The input tensors should have the same feature size as `n_state`.
        """
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class TransformerDecoder(nn.Module):
    """
        TransformerDecoder implements a Transformer Decoder-Only Architecture.

    This class is part of the ESPnet framework and provides a straightforward
    implementation of a Transformer decoder. It supports stacked Transformer
    layers and positional embeddings but does not include an embedding table
    or language model head. The attention mechanism utilizes PyTorch's built-in
    flash attention, and it is recommended to use compatible PyTorch versions
    (2.0.1 or higher).

    Attributes:
        pos_emb (nn.Embedding): Positional embeddings for the input sequences.
        blocks (nn.ModuleList): List of residual attention blocks.
        ln (LayerNorm): Layer normalization applied at the end.
        causal (bool): Indicates if the decoder should operate in causal mode.

    Args:
        n_ctx (int): The number of context tokens.
        n_state (int): The size of the hidden state.
        n_head (int): The number of attention heads.
        n_layer (int): The number of stacked layers.
        causal (bool): If True, enables causal attention (default: True).
        layer_class (type): The class used for the attention layers
            (default: ResidualAttentionBlock).

    Returns:
        Tensor: The output tensor after passing through the decoder.

    Raises:
        ValueError: If causal attention is enabled and a mask is provided.

    Examples:
        >>> decoder = TransformerDecoder(n_ctx=512, n_state=768, n_head=12,
        ...                              n_layer=6)
        >>> x = torch.randn(1, 10, 768)  # (batch_size, seq_len, n_state)
        >>> output = decoder(x)  # output shape: (1, 10, 768)

    Note:
        This implementation intentionally remains simple and may not cover
        all configuration choices available in other libraries like HuggingFace.
    """

    def __init__(
        self,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        causal: bool = True,
        layer_class=ResidualAttentionBlock,
    ):
        super().__init__()

        self.pos_emb = nn.Embedding(n_ctx, n_state)

        self.blocks = nn.ModuleList(
            [layer_class(n_state, n_head, False, causal) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        self.causal = causal

    def forward(
        self, x: Tensor, mask: torch.Tensor = None, kv_cache: Optional[dict] = None
    ):
        """
                Applies the forward pass of the TransformerDecoder.

        This method processes the input tensor `x` through the Transformer
        decoder architecture, applying positional embeddings and a series of
        attention blocks. It supports causal and masked attention mechanisms.

        Args:
            x (Tensor): The input tensor of shape (batch_size, seq_length, n_state).
            mask (torch.Tensor, optional): A tensor used to mask out certain
                positions in the input. This is not allowed if `causal` is True.
            kv_cache (Optional[dict], optional): A cache for key-value pairs
                used in attention layers to improve efficiency during decoding.
                If provided, it should contain pre-computed keys and values
                for faster cross-attention.

        Returns:
            Tensor: The output tensor after passing through the Transformer
            decoder layers of shape (batch_size, seq_length, n_state).

        Raises:
            ValueError: If `causal` is True and `mask` is not None.

        Examples:
            >>> decoder = TransformerDecoder(n_ctx=512, n_state=768, n_head=12, n_layer=6)
            >>> input_tensor = torch.randn(2, 10, 768)  # (batch_size, seq_length, n_state)
            >>> output = decoder(input_tensor)
            >>> output.shape
            torch.Size([2, 10, 768])
        """
        if self.causal and mask is not None:
            raise ValueError("Causal Transformer dones't allow mask")

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = x + self.pos_emb.weight[offset : offset + x.shape[1]].unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask=mask, kv_cache=kv_cache)

        x = self.ln(x)
        return x
