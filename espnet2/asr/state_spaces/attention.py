"""Multi-Head Attention layer definition."""

import math

import numpy
import torch
from torch import nn

from espnet2.asr.state_spaces.base import SequenceModule


class MultiHeadedAttention(SequenceModule):
    """
    Multi-Head Attention layer inheriting from SequenceModule.

    This module implements a Multi-Head Attention mechanism that allows the model 
    to jointly attend to information from different representation subspaces at 
    different positions. It returns an additional dummy state and includes a 
    step function for autoregressive inference.

    Attributes:
        d_k (int): Dimensionality of each head's key and value vectors.
        h (int): Number of attention heads.
        linear_q (nn.Linear): Linear layer for transforming the query.
        linear_k (nn.Linear): Linear layer for transforming the key.
        linear_v (nn.Linear): Linear layer for transforming the value.
        linear_out (nn.Linear): Linear layer for the final output.
        attn (torch.Tensor): Attention weights.
        dropout (nn.Dropout): Dropout layer for regularization.
        d_output (int): Output dimensionality of the attention layer.

    Args:
        n_head (int): The number of attention heads.
        n_feat (int): The number of features in the input tensors.
        dropout_rate (float): Dropout rate applied to the attention weights.
        transposed (bool): Flag indicating whether to use transposed inputs.
        **kwargs: Additional keyword arguments for the parent class.

    Methods:
        forward_qkv(query, key, value):
            Transforms the input query, key, and value tensors.
        forward_attention(value, scores, mask):
            Computes the attention context vector based on the value, scores, and mask.
        forward(query, memory=None, mask=None, *args, **kwargs):
            Computes scaled dot-product attention given the query and optional memory.
        step(query, state, memory=None, mask=None, **kwargs):
            Performs a single step of attention for autoregressive inference.

    Examples:
        >>> mha = MultiHeadedAttention(n_feat=512, n_head=8, dropout_rate=0.1)
        >>> query = torch.rand(32, 10, 512)  # (batch_size, seq_len, n_feat)
        >>> memory = torch.rand(32, 20, 512)
        >>> mask = torch.ones(32, 10, 20)  # Example mask
        >>> output, _ = mha(query, memory=memory, mask=mask)
        >>> print(output.shape)  # Output shape: (32, 10, 512)

    Note:
        The input features should be divisible by the number of heads to ensure 
        even distribution of dimensions across the heads.

    Todo:
        - Implement support for additional input types and shapes.
    """

    def __init__(self, n_feat, n_head, dropout=0.0, transposed=False, **kwargs):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.d_output = n_feat

    def forward_qkv(self, query, key, value):
        """
        Transform query, key, and value tensors for multi-headed attention.

        This method projects the input query, key, and value tensors into multiple 
        heads. The resulting tensors are reshaped to facilitate parallel attention 
        computations across different heads.

        Args:
            query (torch.Tensor): Query tensor of shape 
                (#batch, time1, size).
            key (torch.Tensor): Key tensor of shape 
                (#batch, time2, size).
            value (torch.Tensor): Value tensor of shape 
                (#batch, time2, size).

        Returns:
            tuple: A tuple containing three tensors:
                - torch.Tensor: Transformed query tensor of shape 
                  (#batch, n_head, time1, d_k).
                - torch.Tensor: Transformed key tensor of shape 
                  (#batch, n_head, time2, d_k).
                - torch.Tensor: Transformed value tensor of shape 
                  (#batch, n_head, time2, d_k).

        Examples:
            >>> mha = MultiHeadedAttention(n_feat=64, n_head=8)
            >>> query = torch.randn(32, 10, 64)  # Batch of 32, time1=10
            >>> key = torch.randn(32, 20, 64)    # Batch of 32, time2=20
            >>> value = torch.randn(32, 20, 64)  # Batch of 32, time2=20
            >>> q, k, v = mha.forward_qkv(query, key, value)
            >>> print(q.shape)  # Output: torch.Size([32, 8, 10, 8])
            >>> print(k.shape)  # Output: torch.Size([32, 8, 20, 8])
            >>> print(v.shape)  # Output: torch.Size([32, 8, 20, 8])
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """
        Compute attention context vector.

        This method computes the attention context vector by applying the
        attention scores to the transformed value tensor. The attention scores
        are normalized using the softmax function, and if a mask is provided,
        it is used to prevent attending to certain positions.

        Args:
            value (torch.Tensor): Transformed value tensor of shape
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention scores of shape
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Optional mask tensor of shape
                (#batch, 1, time2) or (#batch, time1, time2) to
                restrict attention to certain positions.

        Returns:
            torch.Tensor: The context vector of shape (#batch, time1, d_model),
                which is the weighted sum of the value tensor according to the
                attention scores.

        Examples:
            >>> mha = MultiHeadedAttention(n_feat=64, n_head=8)
            >>> value = torch.rand(32, 8, 10, 8)  # (#batch, n_head, time2, d_k)
            >>> scores = torch.rand(32, 8, 5, 10)  # (#batch, n_head, time1, time2)
            >>> mask = torch.ones(32, 1, 10)  # Mask for time2
            >>> output = mha.forward_attention(value, scores, mask)
            >>> output.shape
            torch.Size([32, 5, 64])  # (#batch, time1, d_model)

        Note:
            The input `mask` should have the appropriate shape to ensure that
            it can be broadcast correctly with the attention scores.
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, memory=None, mask=None, *args, **kwargs):
        """
        Compute scaled dot product attention.

        This method calculates the scaled dot product attention using the provided
        query, memory (keys and values), and an optional mask. If memory is not 
        provided, the query is used as the memory.

        Args:
            query (torch.Tensor): Query tensor of shape 
                (#batch, time1, size).
            memory (torch.Tensor, optional): Memory tensor (keys and values) of 
                shape (#batch, time2, size). If None, query is used as memory.
            mask (torch.Tensor, optional): Mask tensor of shape 
                (#batch, 1, time2) or (#batch, time1, time2) to 
                prevent attention to certain positions.

        Returns:
            torch.Tensor: Output tensor of shape (#batch, time1, d_model) 
                representing the attention context vector.
            None: This method also returns None as an additional dummy state.

        Examples:
            >>> mha = MultiHeadedAttention(n_feat=512, n_head=8)
            >>> query = torch.rand(32, 10, 512)  # Batch of 32, 10 time steps
            >>> output, _ = mha(query)  # Memory is None, self-attention

            >>> key_value = torch.rand(32, 20, 512)  # 20 time steps for memory
            >>> output, _ = mha(query, memory=key_value)  # Cross-attention

        Note:
            The attention mechanism applies a softmax operation on the 
            attention scores, which are computed as the dot product of 
            the query and key matrices. The scores are scaled by the 
            square root of the dimension of the key vectors.
        """
        # self-attention
        if memory is None:
            memory = query
        q, k, v = self.forward_qkv(query=query, key=memory, value=memory)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), None

    def step(self, query, state, memory=None, mask=None, **kwargs):
        """
        Multi-Head Attention layer inheriting SequenceModule.

        This module implements the Multi-Head Attention mechanism, which allows 
        the model to focus on different parts of the input sequence simultaneously. 
        In comparison to the default Multi-Head Attention module in ESPnet, this 
        module returns an additional dummy state and includes a step function for 
        autoregressive inference.

        Attributes:
            d_k (int): The dimensionality of each head.
            h (int): The number of attention heads.
            linear_q (nn.Linear): Linear layer for transforming the query.
            linear_k (nn.Linear): Linear layer for transforming the key.
            linear_v (nn.Linear): Linear layer for transforming the value.
            linear_out (nn.Linear): Linear layer for the output.
            attn (torch.Tensor): Attention scores.
            dropout (nn.Dropout): Dropout layer for regularization.
            d_output (int): The output feature dimension.

        Args:
            n_head (int): The number of heads.
            n_feat (int): The number of features.
            dropout_rate (float): Dropout rate.

        Examples:
            >>> mha = MultiHeadedAttention(n_feat=512, n_head=8, dropout=0.1)
            >>> query = torch.rand(10, 20, 512)  # (batch, time1, size)
            >>> output, _ = mha(query)

        Raises:
            AssertionError: If `n_feat` is not divisible by `n_head`.

        Note:
            The `step` function is intended for use in autoregressive scenarios 
            where the output from the previous step is used as input for the 
            current step.
        """
        if memory is None:
            memory = query
        return self.forward(query, memory, mask=mask, **kwargs)[0].squeeze(1), state
