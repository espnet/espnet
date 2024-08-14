"""Multi-Head Attention layer definition."""

import math

import numpy
import torch
from torch import nn

from espnet2.asr.state_spaces.base import SequenceModule


class MultiHeadedAttention(SequenceModule):
    """
        Multi-Head Attention layer inheriting from SequenceModule.

    This class implements a Multi-Head Attention mechanism, which allows the model
    to jointly attend to information from different representation subspaces at
    different positions. It extends the functionality of the default Multi-Head
    Attention module in ESPnet by returning an additional dummy state and
    providing a step function for autoregressive inference.

    Attributes:
        d_k (int): The dimension of the key and query vectors.
        h (int): The number of attention heads.
        linear_q (nn.Linear): Linear transformation for queries.
        linear_k (nn.Linear): Linear transformation for keys.
        linear_v (nn.Linear): Linear transformation for values.
        linear_out (nn.Linear): Linear transformation for output.
        attn (torch.Tensor): Attention weights.
        dropout (nn.Dropout): Dropout layer.
        d_output (int): The output dimension.

    Args:
        n_feat (int): The number of expected features in the input.
        n_head (int): The number of attention heads.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        transposed (bool, optional): If True, expects transposed input. Defaults to False.
        **kwargs: Additional keyword arguments.

    Note:
        This implementation assumes that the dimension of the key (d_k) always
        equals the dimension of the value (d_v), and that n_feat is divisible by n_head.

    Example:
        >>> mha = MultiHeadedAttention(n_feat=512, n_head=8, dropout=0.1)
        >>> query = torch.randn(32, 10, 512)  # (batch_size, seq_len, n_feat)
        >>> output, _ = mha(query)
        >>> output.shape
        torch.Size([32, 10, 512])
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
                Transform query, key, and value tensors for multi-head attention.

        This method applies linear transformations to the input query, key, and value
        tensors, reshapes them to separate the head dimension, and transposes the
        resulting tensors to prepare them for the attention mechanism.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, time1, size).
            key (torch.Tensor): Key tensor of shape (batch_size, time2, size).
            value (torch.Tensor): Value tensor of shape (batch_size, time2, size).

        Returns:
            tuple: A tuple containing:
                - q (torch.Tensor): Transformed query tensor of shape (batch_size, n_head, time1, d_k).
                - k (torch.Tensor): Transformed key tensor of shape (batch_size, n_head, time2, d_k).
                - v (torch.Tensor): Transformed value tensor of shape (batch_size, n_head, time2, d_k).

        Note:
            The input tensors are expected to have the same size in their last dimension,
            which should be equal to n_feat (the number of features) in the MultiHeadedAttention class.

        Example:
            >>> mha = MultiHeadedAttention(n_feat=512, n_head=8)
            >>> query = torch.randn(32, 10, 512)
            >>> key = torch.randn(32, 15, 512)
            >>> value = torch.randn(32, 15, 512)
            >>> q, k, v = mha.forward_qkv(query, key, value)
            >>> q.shape, k.shape, v.shape
            (torch.Size([32, 8, 10, 64]), torch.Size([32, 8, 15, 64]), torch.Size([32, 8, 15, 64]))
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

        This method applies the attention mechanism to the transformed value tensor
        using the provided attention scores and mask. It computes the weighted sum
        of values based on the attention distribution.

        Args:
            value (torch.Tensor): Transformed value tensor of shape (batch_size, n_head, time2, d_k).
            scores (torch.Tensor): Attention score tensor of shape (batch_size, n_head, time1, time2).
            mask (torch.Tensor): Mask tensor of shape (batch_size, 1, time2) or (batch_size, time1, time2).
                If provided, positions with 1 are masked (i.e., set to -inf before softmax).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, time1, d_model) containing
                the weighted sum of values according to the attention distribution.

        Note:
            - The method updates the `self.attn` attribute with the computed attention weights.
            - If a mask is provided, it is applied before the softmax to prevent attention
              to certain positions.
            - Dropout is applied to the attention weights before computing the weighted sum.

        Example:
            >>> mha = MultiHeadedAttention(n_feat=512, n_head=8)
            >>> value = torch.randn(32, 8, 15, 64)
            >>> scores = torch.randn(32, 8, 10, 15)
            >>> mask = torch.ones(32, 1, 15).bool()
            >>> output = mha.forward_attention(value, scores, mask)
            >>> output.shape
            torch.Size([32, 10, 512])
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

        This method performs the full multi-head attention operation, including the
        transformation of inputs, computation of attention scores, and application
        of attention weights to produce the final output.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, time1, size).
            memory (torch.Tensor, optional): Memory tensor of shape (batch_size, time2, size).
                If None, self-attention is performed using the query as both key and value.
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, 1, time2) or
                (batch_size, time1, time2). Positions with 1 are masked.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Output tensor of shape (batch_size, time1, d_model).
                - None: A placeholder for compatibility with other modules that return a state.

        Note:
            - If memory is None, self-attention is performed using the query as both key and value.
            - The attention scores are scaled by 1/sqrt(d_k) before applying the mask and softmax.
            - The output is obtained by applying the attention weights to the value vectors
              and then passing through a final linear transformation.

        Example:
            >>> mha = MultiHeadedAttention(n_feat=512, n_head=8)
            >>> query = torch.randn(32, 10, 512)
            >>> memory = torch.randn(32, 15, 512)
            >>> mask = torch.ones(32, 1, 15).bool()
            >>> output, _ = mha.forward(query, memory, mask)
            >>> output.shape
            torch.Size([32, 10, 512])
        """
        # self-attention
        if memory is None:
            memory = query
        q, k, v = self.forward_qkv(query=query, key=memory, value=memory)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), None

    def step(self, query, state, memory=None, mask=None, **kwargs):
        """
                Perform a single step of multi-head attention for autoregressive inference.

        This method is designed for use in autoregressive decoding, where attention is
        computed for a single time step. It wraps the forward method to handle the
        single-step case efficiently.

        Args:
            query (torch.Tensor): Query tensor for the current step, shape (batch_size, 1, size).
            state (Any): The previous state (unused in this implementation).
            memory (torch.Tensor, optional): Memory tensor of shape (batch_size, time2, size).
                If None, self-attention is performed using the query as both key and value.
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, 1, time2) or
                (batch_size, 1, time2). Positions with 1 are masked.
            **kwargs: Additional keyword arguments passed to the forward method.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Output tensor for the current step, shape (batch_size, size).
                - Any: The updated state (same as input state in this implementation).

        Note:
            - This method assumes that the query is for a single time step.
            - The output is squeezed to remove the time dimension, as it's always 1 in this case.
            - The state is passed through unchanged, as the current implementation
              doesn't utilize persistent state between steps.

        Example:
            >>> mha = MultiHeadedAttention(n_feat=512, n_head=8)
            >>> query = torch.randn(32, 1, 512)
            >>> memory = torch.randn(32, 15, 512)
            >>> mask = torch.ones(32, 1, 15).bool()
            >>> output, new_state = mha.step(query, None, memory, mask)
            >>> output.shape
            torch.Size([32, 512])
        """
        if memory is None:
            memory = query
        return self.forward(query, memory, mask=mask, **kwargs)[0].squeeze(1), state
