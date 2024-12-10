"""Fastformer attention definition.

Reference:
    Wu et al., "Fastformer: Additive Attention Can Be All You Need"
    https://arxiv.org/abs/2108.09084
    https://github.com/wuch15/Fastformer

"""

import numpy
import torch


class FastSelfAttention(torch.nn.Module):
    """
    Fast self-attention mechanism used in Fastformer.

    This class implements the Fast Self-Attention mechanism as described in
    the paper "Fastformer: Additive Attention Can Be All You Need" by Wu et al.
    The FastSelfAttention layer is designed to efficiently compute attention
    scores and update representations in sequence-to-sequence models.

    Attributes:
        attention_head_size (int): The size of each attention head.
        num_attention_heads (int): The number of attention heads.
        query (torch.nn.Linear): Linear layer for query transformation.
        query_att (torch.nn.Linear): Linear layer for query attention scores.
        key (torch.nn.Linear): Linear layer for key transformation.
        key_att (torch.nn.Linear): Linear layer for key attention scores.
        transform (torch.nn.Linear): Linear layer for final transformation.
        dropout (torch.nn.Dropout): Dropout layer for regularization.

    Args:
        size (int): Total size of the input features.
        attention_heads (int): Number of attention heads to use.
        dropout_rate (float): Dropout rate to apply to the outputs.

    Raises:
        ValueError: If `size` is not an integer multiple of `attention_heads`.

    Examples:
        >>> fast_attention = FastSelfAttention(size=64, attention_heads=8,
        ...                                     dropout_rate=0.1)
        >>> xs_pad = torch.randn(32, 10, 64)  # (batch, time, size)
        >>> mask = torch.ones(32, 1, 10)  # Non-padding mask
        >>> output = fast_attention(xs_pad, mask)
        >>> print(output.shape)  # Should output: (32, 10, 64)

    Note:
        The implementation uses query-key-value parameter sharing for
        computational efficiency, treating the value as equal to the query.

    Todo:
        Consider adding support for different input types or integrating
        with additional layers in the ESPnet2 framework.
    """

    def __init__(
        self,
        size,
        attention_heads,
        dropout_rate,
    ):
        super().__init__()
        if size % attention_heads != 0:
            raise ValueError(
                f"Hidden size ({size}) is not an integer multiple "
                f"of attention heads ({attention_heads})"
            )
        self.attention_head_size = size // attention_heads
        self.num_attention_heads = attention_heads

        self.query = torch.nn.Linear(size, size)
        self.query_att = torch.nn.Linear(size, attention_heads)
        self.key = torch.nn.Linear(size, size)
        self.key_att = torch.nn.Linear(size, attention_heads)
        self.transform = torch.nn.Linear(size, size)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def espnet_initialization_fn(self):
        """
        Initializes the weights of the FastSelfAttention module.

        This method applies the weight initialization strategy defined in the
        `init_weights` method to all submodules of the FastSelfAttention instance.
        The initialization is done using a normal distribution for the weights
        and zeros for the biases of linear layers.

        Note:
            This function should be called after the model has been instantiated
            to ensure that all weights are initialized correctly.

        Examples:
            >>> attention_layer = FastSelfAttention(size=128, attention_heads=8,
            ...                                     dropout_rate=0.1)
            >>> attention_layer.espnet_initialization_fn()  # Initialize weights

        Raises:
            ValueError: If the weight initialization process fails or if there
            are no linear layers in the module.
        """

        def init_weights(self, module):
            """
            Initialize weights for neural network layers.

            This method applies weight initialization for layers of the neural
            network. Specifically, it initializes the weights of linear layers
            using a normal distribution with a mean of 0 and a standard
            deviation of 0.02. Additionally, it sets the bias of linear layers
            to zero if they exist.

            Args:
                module (torch.nn.Module): The module (layer) to initialize.

            Note:
                This function is typically called during the model's
                initialization process to ensure that the weights are
                appropriately set before training.

            Examples:
                >>> model = FastSelfAttention(size=64, attention_heads=8, dropout_rate=0.1)
                >>> model.espnet_initialization_fn()  # Initializes weights of the model

            Raises:
                ValueError: If the module type is not a recognized layer.
            """

        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transpose_for_scores(self, x):
        """
        Reshape and transpose input tensor for attention score computation.

        This method reshapes the input tensor `x` from a shape of
        (batch, time, size) to (batch, n_heads, time, attn_dim) by
        splitting the last dimension into the number of attention heads
        and the size of each attention head. This transformation is
        essential for computing attention scores across multiple heads.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch, time, size), where `size` is equal to
                `n_heads * attn_dim`.

        Returns:
            torch.Tensor: Reshaped tensor of shape
                (batch, n_heads, time, attn_dim).

        Examples:
            >>> attention = FastSelfAttention(size=64, attention_heads=4,
            ... dropout_rate=0.1)
            >>> x = torch.randn(2, 10, 64)  # (batch, time, size)
            >>> transposed_x = attention.transpose_for_scores(x)
            >>> transposed_x.shape
            torch.Size([2, 4, 10, 16])  # (batch, n_heads, time, attn_dim)
        """

        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        return x.reshape(*new_x_shape).transpose(1, 2)

    def forward(self, xs_pad, mask):
        """
        Compute the forward pass for the FastSelfAttention layer.

        This method performs the forward computation for the FastSelfAttention
        layer. It takes input embeddings and computes attention weights to
        produce the output embeddings.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape
                (batch, time, size = n_heads * attn_dim), where
                'batch' is the number of sequences, 'time' is the
                sequence length, and 'size' is the dimensionality
                of the input embeddings.
            mask (torch.Tensor): A binary tensor of shape (batch, 1, time),
                where non-padding positions are represented by 1 and
                padding positions by 0. This mask is used to ignore
                padding tokens during attention calculation.

        Returns:
            torch.Tensor: Output tensor of shape
                (batch, time, size), which represents the attention
                weighted output embeddings.

        Examples:
            >>> model = FastSelfAttention(size=64, attention_heads=8,
            ... dropout_rate=0.1)
            >>> xs_pad = torch.randn(32, 10, 64)  # batch of 32, seq_len of 10
            >>> mask = torch.ones(32, 1, 10)  # no padding
            >>> output = model(xs_pad, mask)
            >>> output.shape
            torch.Size([32, 10, 64])

        Note:
            The attention mechanism used here is based on the Fastformer
            architecture which leverages additive attention.

        Raises:
            ValueError: If the input size is not an integer multiple
                of the number of attention heads.
        """

        batch_size, seq_len, _ = xs_pad.shape
        mixed_query_layer = self.query(xs_pad)  # (batch, time, size)
        mixed_key_layer = self.key(xs_pad)  # (batch, time, size)

        if mask is not None:
            mask = mask.eq(0)  # padding is 1, nonpadding is 0

        # (batch, n_heads, time)
        query_for_score = (
            self.query_att(mixed_query_layer).transpose(1, 2)
            / self.attention_head_size**0.5
        )
        if mask is not None:
            min_value = float(
                numpy.finfo(
                    torch.tensor(0, dtype=query_for_score.dtype).numpy().dtype
                ).min
            )
            query_for_score = query_for_score.masked_fill(mask, min_value)
            query_weight = torch.softmax(query_for_score, dim=-1).masked_fill(mask, 0.0)
        else:
            query_weight = torch.softmax(query_for_score, dim=-1)

        query_weight = query_weight.unsqueeze(2)  # (batch, n_heads, 1, time)
        query_layer = self.transpose_for_scores(
            mixed_query_layer
        )  # (batch, n_heads, time, attn_dim)

        pooled_query = (
            torch.matmul(query_weight, query_layer)
            .transpose(1, 2)
            .reshape(-1, 1, self.num_attention_heads * self.attention_head_size)
        )  # (batch, 1, size = n_heads * attn_dim)
        pooled_query = self.dropout(pooled_query)
        pooled_query_repeat = pooled_query.repeat(1, seq_len, 1)  # (batch, time, size)

        mixed_query_key_layer = (
            mixed_key_layer * pooled_query_repeat
        )  # (batch, time, size)

        # (batch, n_heads, time)
        query_key_score = (
            self.key_att(mixed_query_key_layer) / self.attention_head_size**0.5
        ).transpose(1, 2)
        if mask is not None:
            min_value = float(
                numpy.finfo(
                    torch.tensor(0, dtype=query_key_score.dtype).numpy().dtype
                ).min
            )
            query_key_score = query_key_score.masked_fill(mask, min_value)
            query_key_weight = torch.softmax(query_key_score, dim=-1).masked_fill(
                mask, 0.0
            )
        else:
            query_key_weight = torch.softmax(query_key_score, dim=-1)

        query_key_weight = query_key_weight.unsqueeze(2)  # (batch, n_heads, 1, time)
        key_layer = self.transpose_for_scores(
            mixed_query_key_layer
        )  # (batch, n_heads, time, attn_dim)
        pooled_key = torch.matmul(
            query_key_weight, key_layer
        )  # (batch, n_heads, 1, attn_dim)
        pooled_key = self.dropout(pooled_key)

        # NOTE: value = query, due to param sharing
        weighted_value = (pooled_key * query_layer).transpose(
            1, 2
        )  # (batch, time, n_heads, attn_dim)
        weighted_value = weighted_value.reshape(
            weighted_value.shape[:-2]
            + (self.num_attention_heads * self.attention_head_size,)
        )  # (batch, time, size)
        weighted_value = (
            self.dropout(self.transform(weighted_value)) + mixed_query_layer
        )

        return weighted_value
