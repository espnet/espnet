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

    This class implements the fast self-attention mechanism as described in the
    Fastformer paper by Wu et al. It provides an efficient alternative to
    traditional self-attention mechanisms used in transformer models.

    Attributes:
        attention_head_size (int): The size of each attention head.
        num_attention_heads (int): The number of attention heads.
        query (torch.nn.Linear): Linear layer for query transformation.
        query_att (torch.nn.Linear): Linear layer for query attention.
        key (torch.nn.Linear): Linear layer for key transformation.
        key_att (torch.nn.Linear): Linear layer for key attention.
        transform (torch.nn.Linear): Linear layer for final transformation.
        dropout (torch.nn.Dropout): Dropout layer for regularization.

    Args:
        size (int): The input size (hidden size) of the model.
        attention_heads (int): The number of attention heads.
        dropout_rate (float): The dropout rate to use for regularization.

    Raises:
        ValueError: If the hidden size is not an integer multiple of the number
            of attention heads.

    Example:
        >>> model = FastSelfAttention(512, 8, 0.1)
        >>> input_tensor = torch.randn(32, 100, 512)  # (batch, seq_len, hidden_size)
        >>> mask = torch.ones(32, 1, 100)  # (batch, 1, seq_len)
        >>> output = model(input_tensor, mask)
        >>> print(output.shape)
        torch.Size([32, 100, 512])

    Note:
        This implementation is based on the paper "Fastformer: Additive Attention
        Can Be All You Need" by Wu et al. and the corresponding GitHub repository.
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
            Initialize the weights of the FastSelfAttention module.

        This method applies the `init_weights` function to all submodules of the
        FastSelfAttention module. It is designed to be used as an initialization
        function in the ESPnet framework.

        Note:
            This method does not take any arguments and does not return anything.
            It modifies the module's weights in-place.

        Example:
            >>> model = FastSelfAttention(512, 8, 0.1)
            >>> model.espnet_initialization_fn()
        """

    def init_weights(self, module):
        """
                Initialize the weights of a given module.

        This method initializes the weights of linear layers in the module. It sets
        the weights to a normal distribution with mean 0.0 and standard deviation 0.02,
        and initializes biases to zero.

        Args:
            module (torch.nn.Module): The module whose weights are to be initialized.

        Note:
            This method is typically called by `espnet_initialization_fn` and is not
            meant to be used directly in most cases.

        Example:
            >>> linear = torch.nn.Linear(10, 10)
            >>> model = FastSelfAttention(512, 8, 0.1)
            >>> model.init_weights(linear)
        """
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transpose_for_scores(self, x):
        """
                Reshape and transpose the input tensor for computing attention scores.

        This method reshapes the input tensor and transposes it to prepare for
        computing attention scores in the fast self-attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, size),
                where size = n_heads * attn_dim.

        Returns:
            torch.Tensor: Reshaped and transposed tensor of shape
                (batch, n_heads, time, attn_dim).

        Example:
            >>> model = FastSelfAttention(512, 8, 0.1)
            >>> x = torch.randn(32, 100, 512)
            >>> transposed = model.transpose_for_scores(x)
            >>> print(transposed.shape)
            torch.Size([32, 8, 100, 64])

        Note:
            This method is used internally in the forward pass of the FastSelfAttention
            module and is not typically called directly by users.
        """

        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        return x.reshape(*new_x_shape).transpose(1, 2)

    def forward(self, xs_pad, mask):
        """
                Perform the forward pass of the FastSelfAttention module.

        This method implements the fast self-attention mechanism, processing the input
        tensor and applying the attention weights to produce the output.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (batch, time, size),
                where size = n_heads * attn_dim.
            mask (torch.Tensor): Mask tensor of shape (batch, 1, time), where
                non-padding positions are 1 and padding positions are 0.

        Returns:
            torch.Tensor: Output tensor after applying fast self-attention,
                of shape (batch, time, size).

        Example:
            >>> model = FastSelfAttention(512, 8, 0.1)
            >>> xs_pad = torch.randn(32, 100, 512)
            >>> mask = torch.ones(32, 1, 100)
            >>> output = model.forward(xs_pad, mask)
            >>> print(output.shape)
            torch.Size([32, 100, 512])

        Note:
            The mask tensor should have 1s for non-padding positions and 0s for
            padding positions. The method internally inverts this mask for
            compatibility with the implementation.
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
