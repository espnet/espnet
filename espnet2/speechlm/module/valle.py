from typing import Optional

from torch import Tensor, nn

from espnet2.speechlm.module.transformer import (
    ResidualAttentionBlock,
    TransformerDecoder,
)


class AdaLN(nn.Module):
    """
        AdaLN is a custom layer normalization module that adapts the normalization
    parameters based on an additional embedding vector. This class extends
    torch.nn.Module and allows for the dynamic adjustment of layer normalization
    weights and biases using an input tensor, `level_emb`.

    Attributes:
        n_state (int): The dimensionality of the input and output tensors.
        eps (float): A small value added for numerical stability in layer normalization.

    Args:
        n_state (int): The number of input features for the layer.
        eps (float, optional): The epsilon value for numerical stability in
            layer normalization. Default is 1e-5.

    Returns:
        Tensor: The output tensor after applying the adaptive layer normalization.

    Examples:
        >>> ada_ln = AdaLN(n_state=128)
        >>> x = torch.randn(10, 128)  # Batch of 10, 128 features
        >>> level_emb = torch.randn(10, 128)  # Batch of 10, 128 features
        >>> output = ada_ln(x, level_emb)
        >>> print(output.shape)
        torch.Size([10, 128])

    Note:
        This implementation uses two linear layers without biases to compute
        the weight and bias for the layer normalization. The weight and bias
        are computed from the `level_emb` tensor, which allows for dynamic
        adjustments based on contextual embeddings.
    """

    def __init__(self, n_state, eps=1e-5):
        super().__init__()
        self.weight = nn.Linear(n_state, n_state, bias=False)
        self.bias = nn.Linear(n_state, n_state, bias=False)
        nn.init.constant_(self.weight.weight, 1.0)
        nn.init.constant_(self.bias.weight, 0.0)

        self.n_state = n_state
        self.eps = eps

    def forward(self, x: Tensor, level_emb: Tensor):
        """
            Apply the AdaLN layer normalization with learned scaling and bias.

        This method performs layer normalization on the input tensor `x` using
        scaling and bias that are learned based on the `level_emb` tensor. The
        scaling and bias are obtained by passing `level_emb` through the weight
        and bias linear layers, respectively.

        Args:
            x (Tensor): The input tensor to be normalized, typically of shape
                (batch_size, n_state).
            level_emb (Tensor): The embedding tensor used to compute the scaling
                and bias, of shape (batch_size, n_state).

        Returns:
            Tensor: The output tensor after applying layer normalization and
            scaling and bias adjustment, with the same shape as the input tensor `x`.

        Examples:
            >>> ada_ln = AdaLN(n_state=64)
            >>> x = torch.randn(10, 64)  # Batch of 10 samples
            >>> level_emb = torch.randn(10, 64)  # Corresponding level embeddings
            >>> output = ada_ln.forward(x, level_emb)
            >>> print(output.shape)  # Should be (10, 64)

        Note:
            The layer normalization is applied with an epsilon value to avoid
            division by zero, which can be set during the initialization of the
            AdaLN instance.
        """
        w = self.weight(level_emb).unsqueeze(1)
        b = self.bias(level_emb).unsqueeze(1)
        x = nn.functional.layer_norm(x, (self.n_state,), eps=self.eps)
        x = w * x + b
        return x


class ResidualAttentionBlockAdaLM(ResidualAttentionBlock):
    """
        ResidualAttentionBlockAdaLM is a class that implements a residual attention
    block for adaptive layer normalization in the context of language modeling.
    It extends the ResidualAttentionBlock class and utilizes AdaLN for
    normalization.

    Attributes:
        n_state (int): The dimensionality of the input and output states.
        n_head (int): The number of attention heads.
        cross_attention (bool): A flag indicating whether to use cross-attention.

    Args:
        n_state (int): The dimensionality of the input and output states.
        n_head (int): The number of attention heads.
        cross_attention (bool, optional): A flag to enable cross-attention.
            Defaults to False.

    Methods:
        forward(x: Tensor, level: Tensor, xa: Optional[Tensor] = None,
                mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None) -> Tensor:
            Computes the forward pass of the residual attention block.

    Raises:
        ValueError: If the input tensor dimensions do not match the expected
            dimensions.

    Examples:
        >>> block = ResidualAttentionBlockAdaLM(n_state=512, n_head=8)
        >>> x = torch.randn(10, 20, 512)  # (batch_size, seq_len, n_state)
        >>> level = torch.randint(0, 5, (10,))  # (batch_size,)
        >>> output = block(x, level)
        >>> print(output.shape)
        torch.Size([10, 20, 512])
    """

    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super(ResidualAttentionBlockAdaLM, self).__init__(
            n_state=n_state,
            n_head=n_head,
            cross_attention=cross_attention,
        )

        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                setattr(self, name, AdaLN(n_state))

    def forward(
        self,
        x: Tensor,
        level: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        """
                Executes the forward pass of the ResidualAttentionBlockAdaLM module.

        This method takes an input tensor and performs attention and feed-forward
        operations, applying layer normalization and residual connections. The
        method can handle optional cross-attention if specified during
        initialization.

        Args:
            x (Tensor): The input tensor of shape (batch_size, seq_length, n_state).
            level (Tensor): The level embedding tensor of shape (batch_size,).
            xa (Optional[Tensor], optional): The cross-attention input tensor of shape
                (batch_size, seq_length, n_state). Defaults to None.
            mask (Optional[Tensor], optional): The attention mask tensor to prevent
                attending to certain positions. Defaults to None.
            kv_cache (Optional[dict], optional): A cache dictionary for key-value pairs
                in attention. Defaults to None.

        Returns:
            Tensor: The output tensor after applying attention and feed-forward
            operations, with the same shape as input tensor x.

        Examples:
            >>> block = ResidualAttentionBlockAdaLM(n_state=512, n_head=8)
            >>> x = torch.rand(10, 20, 512)  # batch_size=10, seq_length=20
            >>> level = torch.randint(0, 5, (10,))  # 5 levels
            >>> output = block(x, level)
            >>> output.shape
            torch.Size([10, 20, 512])

        Note:
            This method is part of the ResidualAttentionBlockAdaLM class, which is
            derived from ResidualAttentionBlock and implements additional features
            for attention mechanisms.
        """
        x = x + self.attn(self.attn_ln(x, level), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = (
                x
                + self.cross_attn(self.cross_attn_ln(x, level), xa, kv_cache=kv_cache)[
                    0
                ]
            )
        x = x + self.mlp(self.mlp_ln(x, level))
        return x


class ValleNARDecoder(TransformerDecoder):
    """
        ValleNARDecoder is a non-autoregressive transformer decoder designed for
    speech processing tasks. It utilizes a series of residual attention blocks
    with AdaLN for layer normalization, enabling effective context handling and
    embedding representations.

    Attributes:
        level_emb (nn.Embedding): Embedding layer for level inputs.
        ln (AdaLN): Adaptive layer normalization layer.

    Args:
        n_level (int): Number of different levels for the input.
        n_ctx (int): Context size for the input sequences.
        n_state (int): Dimensionality of the model's hidden states.
        n_head (int): Number of attention heads in each layer.
        n_layer (int): Total number of layers in the decoder.
        causal (bool, optional): Whether to use causal masking (default is True).
        layer_class (type, optional): Class for the residual attention block
            (default is ResidualAttentionBlockAdaLM).

    Returns:
        Tensor: The output tensor after processing through the decoder.

    Examples:
        >>> decoder = ValleNARDecoder(n_level=10, n_ctx=20, n_state=512,
        ...                            n_head=8, n_layer=6)
        >>> x = torch.randn(1, 20, 512)  # Batch size of 1, sequence length 20
        >>> level = torch.randint(0, 10, (1, 20))  # Random level inputs
        >>> output = decoder(x, level)

    Note:
        This decoder is particularly suited for tasks in speech language modeling
        and is part of the ESPnet2 framework.

    Todo:
        - Implement additional functionality for handling variable-length inputs.
    """

    def __init__(
        self,
        n_level: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        causal: bool = True,
        layer_class=ResidualAttentionBlockAdaLM,
    ):
        super(ValleNARDecoder, self).__init__(
            n_ctx=n_ctx,
            n_state=n_state,
            n_head=n_head,
            n_layer=n_layer,
            causal=causal,
            layer_class=layer_class,
        )

        self.level_emb = nn.Embedding(n_level, n_state)
        self.ln = AdaLN(n_state)

    def forward(self, x: Tensor, level: Tensor, kv_cache: Optional[dict] = None):
        """
                Forward pass for the ValleNARDecoder class.

        This method processes the input tensor `x` through a series of attention blocks
        and layer normalization, incorporating positional embeddings and level embeddings.
        It can optionally utilize a key-value cache for efficient decoding in scenarios
        such as autoregressive generation.

        Args:
            x (Tensor): The input tensor of shape (batch_size, sequence_length, n_state).
            level (Tensor): The level indices tensor of shape (batch_size,).
            kv_cache (Optional[dict], optional): A dictionary containing cached key-value
                pairs for cross-attention. Defaults to None.

        Returns:
            Tensor: The output tensor after processing through the decoder layers,
                of shape (batch_size, sequence_length, n_state).

        Examples:
            >>> decoder = ValleNARDecoder(n_level=10, n_ctx=512, n_state=256,
            ...                            n_head=8, n_layer=6)
            >>> input_tensor = torch.randn(2, 20, 256)  # Batch of 2, sequence length of 20
            >>> level_tensor = torch.tensor([0, 1])  # Two levels for the batch
            >>> output = decoder(input_tensor, level_tensor)
            >>> print(output.shape)  # Output shape should be (2, 20, 256)

        Note:
            This method assumes that the input `x` has already been embedded into the
            appropriate shape and dimensionality.

        Todo:
            - Add support for additional types of embeddings if required in the future.
        """
        level = self.level_emb(level)

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = x + self.pos_emb.weight[offset : offset + x.shape[1]].unsqueeze(0)

        for block in self.blocks:
            x = block(x, level=level, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x, level)
        return x
