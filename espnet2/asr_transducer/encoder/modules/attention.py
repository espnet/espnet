"""Multi-Head attention layers with relative positional encoding."""

import math
from typing import Optional, Tuple

import torch


class RelPositionMultiHeadedAttention(torch.nn.Module):
    """
    Multi-Head attention layers with relative positional encoding.

    This class implements multi-headed attention with the capability to use
    relative positional encoding. It allows for efficient attention
    computation in tasks such as speech recognition and natural language
    processing.

    Attributes:
        d_k (int): Dimensionality of each attention head.
        num_heads (int): Number of attention heads.
        linear_q (torch.nn.Linear): Linear transformation for query.
        linear_k (torch.nn.Linear): Linear transformation for key.
        linear_v (torch.nn.Linear): Linear transformation for value.
        linear_out (torch.nn.Linear): Linear transformation for output.
        linear_pos (torch.nn.Linear): Linear transformation for positional encoding.
        pos_bias_u (torch.nn.Parameter): Parameter for position bias U.
        pos_bias_v (torch.nn.Parameter): Parameter for position bias V.
        dropout (torch.nn.Dropout): Dropout layer for regularization.
        attn (torch.Tensor): Tensor to store attention weights.

    Args:
        num_heads (int): Number of attention heads.
        embed_size (int): Size of the input embeddings.
        dropout_rate (float, optional): Dropout rate for regularization. Default is 0.0.
        simplified_attention_score (bool, optional): Use simplified attention score
            computation. Default is False.

    Methods:
        rel_shift(x: torch.Tensor, left_context: int = 0) -> torch.Tensor:
            Compute relative positional encoding.

        compute_simplified_attention_score(query: torch.Tensor, key: torch.Tensor,
            pos_enc: torch.Tensor, left_context: int = 0) -> torch.Tensor:
            Simplified attention score computation.

        compute_attention_score(query: torch.Tensor, key: torch.Tensor,
            pos_enc: torch.Tensor, left_context: int = 0) -> torch.Tensor:
            Attention score computation.

        forward_qkv(query: torch.Tensor, key: torch.Tensor,
            value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Transform query, key and value.

        forward_attention(value: torch.Tensor, scores: torch.Tensor,
            mask: torch.Tensor, chunk_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            Compute attention context vector.

        forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            pos_enc: torch.Tensor, mask: torch.Tensor,
            chunk_mask: Optional[torch.Tensor] = None, left_context: int = 0) -> torch.Tensor:
            Compute scaled dot product attention with relative positional encoding.

    Examples:
        # Initialize the attention layer
        attention_layer = RelPositionMultiHeadedAttention(num_heads=8,
                                                        embed_size=512)

        # Example input tensors
        query = torch.rand(32, 10, 512)  # (B, T_1, size)
        key = torch.rand(32, 20, 512)    # (B, T_2, size)
        value = torch.rand(32, 20, 512)  # (B, T_2, size)
        pos_enc = torch.rand(32, 39, 512) # (B, 2 * T_1 - 1, size)
        mask = torch.ones(32, 20)         # (B, T_2)

        # Forward pass
        output = attention_layer(query, key, value, pos_enc, mask)

    Note:
        The input tensors should be properly sized and batched for the attention
        computation to work as expected.

    Todo:
        - Implement additional features or optimizations as needed.
    """

    def __init__(
        self,
        num_heads: int,
        embed_size: int,
        dropout_rate: float = 0.0,
        simplified_attention_score: bool = False,
    ) -> None:
        """Construct an MultiHeadedAttention object."""
        super().__init__()

        self.d_k = embed_size // num_heads
        self.num_heads = num_heads

        assert self.d_k * num_heads == embed_size, (
            "embed_size (%d) must be divisible by num_heads (%d)",
            (embed_size, num_heads),
        )

        self.linear_q = torch.nn.Linear(embed_size, embed_size)
        self.linear_k = torch.nn.Linear(embed_size, embed_size)
        self.linear_v = torch.nn.Linear(embed_size, embed_size)

        self.linear_out = torch.nn.Linear(embed_size, embed_size)

        if simplified_attention_score:
            self.linear_pos = torch.nn.Linear(embed_size, num_heads)

            self.compute_att_score = self.compute_simplified_attention_score
        else:
            self.linear_pos = torch.nn.Linear(embed_size, embed_size, bias=False)

            self.pos_bias_u = torch.nn.Parameter(torch.Tensor(num_heads, self.d_k))
            self.pos_bias_v = torch.nn.Parameter(torch.Tensor(num_heads, self.d_k))
            torch.nn.init.xavier_uniform_(self.pos_bias_u)
            torch.nn.init.xavier_uniform_(self.pos_bias_v)

            self.compute_att_score = self.compute_attention_score

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.attn = None

    def rel_shift(self, x: torch.Tensor, left_context: int = 0) -> torch.Tensor:
        """
        Compute relative positional encoding.

        This function performs a relative shift on the input tensor `x` to
        facilitate attention computation with respect to the given context.
        The output tensor will have an additional dimension that represents
        the shifted context based on the specified `left_context`.

        Args:
            x: Input sequence tensor of shape (B, H, T_1, 2 * T_1 - 1), where:
                B is the batch size,
                H is the number of attention heads,
                T_1 is the length of the first sequence,
                and (2 * T_1 - 1) represents the concatenated lengths for
                relative positional encoding.
            left_context: Number of previous frames to use for current chunk
                        attention computation. This controls how much
                        context is included in the attention mechanism.

        Returns:
            torch.Tensor: Output sequence tensor of shape (B, H, T_1, T_2), where:
                T_2 is T_1 plus the `left_context`, representing the
                new length of the sequence after applying the relative shift.

        Examples:
            >>> attention = RelPositionMultiHeadedAttention(num_heads=8, embed_size=64)
            >>> input_tensor = torch.randn(2, 8, 10, 19)  # (B, H, T_1, 2*T_1-1)
            >>> output_tensor = attention.rel_shift(input_tensor, left_context=2)
            >>> output_tensor.shape
            torch.Size([2, 8, 10, 12])  # (B, H, T_1, T_2)

        Note:
            The relative shift operation is crucial for implementing
            relative positional encoding in multi-headed attention
            mechanisms, enabling the model to leverage context effectively.
        """
        batch_size, n_heads, time1, n = x.shape
        time2 = time1 + left_context

        batch_stride, n_heads_stride, time1_stride, n_stride = x.stride()

        return x.as_strided(
            (batch_size, n_heads, time1, time2),
            (batch_stride, n_heads_stride, time1_stride - n_stride, n_stride),
            storage_offset=(n_stride * (time1 - 1)),
        )

    def compute_simplified_attention_score(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        pos_enc: torch.Tensor,
        left_context: int = 0,
    ) -> torch.Tensor:
        """
        Compute simplified attention scores using query, key, and positional
        encodings.

        This method computes the attention scores by combining the dot product of
        the query and key tensors with a positional encoding that has been shifted
        to account for the specified left context. The scores are normalized by
        the square root of the dimension of the keys.

        Reference: https://github.com/k2-fsa/icefall/pull/458

        Args:
            query: Transformed query tensor of shape (B, H, T_1, d_k), where B is
                   the batch size, H is the number of heads, T_1 is the length
                   of the query sequence, and d_k is the dimension of each head.
            key: Transformed key tensor of shape (B, H, T_2, d_k), where T_2 is
                 the length of the key sequence.
            pos_enc: Positional embedding tensor of shape (B, 2 * T_1 - 1, size),
                      which provides positional information for the attention
                      mechanism.
            left_context: An integer representing the number of previous frames to
                          use for current chunk attention computation. Default is 0.

        Returns:
            A tensor representing the attention scores of shape (B, H, T_1, T_2).

        Examples:
            >>> query = torch.randn(2, 4, 10, 64)  # Example query tensor
            >>> key = torch.randn(2, 4, 15, 64)    # Example key tensor
            >>> pos_enc = torch.randn(2, 19, 64)   # Example positional encoding
            >>> left_context = 3
            >>> attention_scores = compute_simplified_attention_score(query, key,
            ... pos_enc, left_context)
            >>> attention_scores.shape
            torch.Size([2, 4, 10, 15])  # Shape of attention scores
        """
        pos_enc = self.linear_pos(pos_enc)

        matrix_ac = torch.matmul(query, key.transpose(2, 3))

        matrix_bd = self.rel_shift(
            pos_enc.transpose(1, 2).unsqueeze(2).repeat(1, 1, query.size(2), 1),
            left_context=left_context,
        )

        return (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

    def compute_attention_score(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        pos_enc: torch.Tensor,
        left_context: int = 0,
    ) -> torch.Tensor:
        """
        Compute attention scores based on the query, key, and positional encoding.

        This method calculates the attention scores using a more complex
        mechanism that incorporates relative positional encoding through learned
        bias vectors. It is designed to work within the multi-headed attention
        framework, enabling the model to better capture dependencies in the
        input sequences.

        Args:
            query: Transformed query tensor. Shape: (B, H, T_1, d_k)
            key: Transformed key tensor. Shape: (B, H, T_2, d_k)
            pos_enc: Positional embedding tensor. Shape: (B, 2 * T_1 - 1, size)
            left_context: Number of previous frames to use for current chunk
                          attention computation. Default is 0.

        Returns:
            Tensor: Attention scores. Shape: (B, H, T_1, T_2)

        Examples:
            >>> query = torch.rand(2, 4, 10, 16)  # (B, H, T_1, d_k)
            >>> key = torch.rand(2, 4, 12, 16)    # (B, H, T_2, d_k)
            >>> pos_enc = torch.rand(2, 19, 32)   # (B, 2 * T_1 - 1, size)
            >>> left_context = 2
            >>> attention_scores = compute_attention_score(query, key, pos_enc, left_context)
            >>> print(attention_scores.shape)  # Should output: (2, 4, 10, 12)
        """
        p = self.linear_pos(pos_enc).view(pos_enc.size(0), -1, self.num_heads, self.d_k)

        query = query.transpose(1, 2)
        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))

        matrix_bd = torch.matmul(q_with_bias_v, p.permute(0, 2, 3, 1))
        matrix_bd = self.rel_shift(matrix_bd, left_context=left_context)

        return (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform query, key and value tensors into multi-head format.

        This method takes the input query, key, and value tensors and applies
        linear transformations to each of them, reshaping the resulting tensors
        into the appropriate multi-head format. Each tensor is divided into
        multiple heads for attention computation.

        Args:
            query: Query tensor. Shape: (B, T_1, size)
            key: Key tensor. Shape: (B, T_2, size)
            value: Value tensor. Shape: (B, T_2, size)

        Returns:
            Tuple of transformed tensors:
                - q: Transformed query tensor. Shape: (B, H, T_1, d_k)
                - k: Transformed key tensor. Shape: (B, H, T_2, d_k)
                - v: Transformed value tensor. Shape: (B, H, T_2, d_k)

        Examples:
            >>> attention = RelPositionMultiHeadedAttention(num_heads=8, embed_size=64)
            >>> query = torch.rand(32, 10, 64)  # Batch of 32, seq length 10
            >>> key = torch.rand(32, 20, 64)    # Batch of 32, seq length 20
            >>> value = torch.rand(32, 20, 64)  # Batch of 32, seq length 20
            >>> q, k, v = attention.forward_qkv(query, key, value)
            >>> q.shape  # (32, 8, 10, 8)
            >>> k.shape  # (32, 8, 20, 8)
            >>> v.shape  # (32, 8, 20, 8)
        """
        n_batch = query.size(0)

        q = (
            self.linear_q(query)
            .view(n_batch, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        k = (
            self.linear_k(key)
            .view(n_batch, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.linear_v(value)
            .view(n_batch, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        return q, k, v

    def forward_attention(
        self,
        value: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention context vector.

        This method computes the attention context vector by applying the attention
        scores to the transformed value tensor. It also applies masking to prevent
        attention to certain positions based on the provided masks.

        Args:
            value: Transformed value tensor. Shape: (B, H, T_2, d_k)
            scores: Attention score tensor. Shape: (B, H, T_1, T_2)
            mask: Source mask tensor. Shape: (B, T_2)
            chunk_mask: Optional chunk mask tensor. Shape: (T_1, T_1)

        Returns:
            attn_output: The transformed value weighted by the attention scores.
                          Shape: (B, T_1, H * d_k)

        Examples:
            >>> import torch
            >>> attention_layer = RelPositionMultiHeadedAttention(num_heads=8,
            ...                                                      embed_size=512)
            >>> value = torch.randn(32, 8, 50, 64)  # (B, H, T_2, d_k)
            >>> scores = torch.randn(32, 8, 10, 50)  # (B, H, T_1, T_2)
            >>> mask = torch.ones(32, 50).bool()     # (B, T_2)
            >>> chunk_mask = torch.zeros(10, 10).bool()  # (T_1, T_1)
            >>> output = attention_layer.forward_attention(value, scores, mask,
            ...                                             chunk_mask)
        """
        batch_size = scores.size(0)
        mask = mask.unsqueeze(1).unsqueeze(2)

        if chunk_mask is not None:
            mask = chunk_mask.unsqueeze(0).unsqueeze(1) | mask

        scores = scores.masked_fill(mask, float("-inf"))
        self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)

        attn_output = self.dropout(self.attn)
        attn_output = torch.matmul(attn_output, value)

        attn_output = self.linear_out(
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_k)
        )

        return attn_output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
        left_context: int = 0,
    ) -> torch.Tensor:
        """
        Compute scaled dot product attention with relative positional encoding.

        This method computes the attention output based on the input query, key,
        and value tensors while incorporating relative positional encoding. It
        calculates the attention scores and applies a mask to prevent attention
        to certain positions as defined by the mask and chunk_mask.

        Args:
            query: Query tensor. Shape (B, T_1, size), where B is the batch size,
                   T_1 is the sequence length of the query, and size is the
                   embedding dimension.
            key: Key tensor. Shape (B, T_2, size), where T_2 is the sequence length
                 of the key.
            value: Value tensor. Shape (B, T_2, size), where T_2 is the sequence
                   length of the value.
            pos_enc: Positional embedding tensor. Shape (B, 2 * T_1 - 1, size)
                      which provides the positional information.
            mask: Source mask. Shape (B, T_2) used to prevent attention to certain
                  positions in the key/value sequences.
            chunk_mask: Optional chunk mask. Shape (T_1, T_1) used to restrict
                        attention within chunks.
            left_context: Number of previous frames to use for current chunk
                          attention computation. Default is 0.

        Returns:
            Output tensor. Shape (B, T_1, H * d_k), where H is the number of
            attention heads and d_k is the dimension of each head.

        Examples:
            >>> attention_layer = RelPositionMultiHeadedAttention(num_heads=8,
            ... embed_size=512)
            >>> query = torch.randn(2, 10, 512)  # Batch of 2, T_1=10
            >>> key = torch.randn(2, 15, 512)    # T_2=15
            >>> value = torch.randn(2, 15, 512)
            >>> pos_enc = torch.randn(2, 19, 512)  # 2 * T_1 - 1 = 19
            >>> mask = torch.zeros(2, 15).bool()
            >>> output = attention_layer(query, key, value, pos_enc, mask)
            >>> output.shape
            torch.Size([2, 10, 512])
        """
        q, k, v = self.forward_qkv(query, key, value)

        scores = self.compute_att_score(q, k, pos_enc, left_context=left_context)

        return self.forward_attention(v, scores, mask, chunk_mask=chunk_mask)
