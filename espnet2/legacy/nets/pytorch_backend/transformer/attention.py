#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import logging
import math

import torch
from torch import nn

from espnet2.legacy.nets.pytorch_backend.transformer.layer_norm import LayerNorm

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
except Exception as e:
    print(f"Failed to import Flash Attention, using ESPnet default: {e}")


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        qk_norm (bool): Normalize q and k before dot product.
        use_flash_attn (bool): Use flash_attn implementation.
        causal (bool): Apply causal attention.
        cross_attn (bool): Cross attention instead of self attention.
        use_sdpa (bool): Use PyTorch's scaled dot product attention.

    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        qk_norm=False,
        use_flash_attn=False,
        causal=False,
        cross_attn=False,
        use_sdpa=False,
    ):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()

        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = (
            nn.Dropout(p=dropout_rate) if not use_flash_attn else nn.Identity()
        )
        self.dropout_rate = dropout_rate

        # LayerNorm for q and k
        self.q_norm = LayerNorm(self.d_k) if qk_norm else nn.Identity()
        self.k_norm = LayerNorm(self.d_k) if qk_norm else nn.Identity()

        self.use_flash_attn = use_flash_attn
        self.causal = causal  # only used with flash_attn
        self.cross_attn = cross_attn  # only used with flash_attn

        self.use_sdpa = use_sdpa

    def forward_qkv(self, query, key, value, expand_kv=False):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            expand_kv (bool): Used only for partially autoregressive (PAR) decoding.

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)

        if expand_kv:
            k_shape = key.shape
            k = (
                self.linear_k(key[:1, :, :])
                .expand(n_batch, k_shape[1], k_shape[2])
                .view(n_batch, -1, self.h, self.d_k)
            )
            v_shape = value.shape
            v = (
                self.linear_v(value[:1, :, :])
                .expand(n_batch, v_shape[1], v_shape[2])
                .view(n_batch, -1, self.h, self.d_k)
            )
        else:
            k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
            v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)

        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        q = self.q_norm(q)
        k = self.k_norm(k)

        return q, k, v

    def project_kv(self, key, value):
        """Transform key and value only, for reuse across decoding steps.

        The result is exactly the ``(k, v)`` that :meth:`forward_qkv` would
        return for the same ``key`` and ``value``, so a caller that attends to
        a memory that does not change from one step to the next (the encoder
        output during autoregressive decoding) can compute it once and hand it
        back through ``forward(..., kv=(k, v))``.

        Args:
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = key.size(0)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        return self.k_norm(k), v

    def _forward_with_kv(self, query, kv, mask):
        """Attend with key/value projections computed earlier by `project_kv`.

        ``kv`` may hold one row per query, or one row per *group* of
        consecutive queries: query row ``b * n_rep + i`` then attends to kv
        row ``b``. That is the layout of an utterance-major beam search, where
        the ``n_rep`` hypotheses of an utterance all attend to the same
        encoder output. The grouped case folds the ``n_rep`` queries of a group
        into the query-time axis, so the projections are never replicated.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            kv (tuple[torch.Tensor, torch.Tensor]): Projected key and value,
                each (#batch or #batch / n_rep, n_head, time2, d_k).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2), or
                (#batch, time1, time2), or None. In the grouped case a
                per-position mask cannot be folded, so the projections are
                expanded over the group instead.

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        k, v = kv
        n_batch, time1 = query.size(0), query.size(1)
        n_kv = k.size(0)
        q = self.linear_q(query).view(n_batch, time1, self.h, self.d_k)
        q = self.q_norm(q.transpose(1, 2))  # (batch, head, time1, d_k)

        n_rep = 1
        if n_kv != n_batch:
            if n_kv == 0 or n_batch % n_kv != 0:
                raise ValueError(
                    f"{n_batch} queries cannot share {n_kv} key/value rows"
                )
            n_rep = n_batch // n_kv
            per_position = mask is not None and mask.size(1) != 1
            if per_position:
                # fall back to one kv row per query
                k = (
                    k.unsqueeze(1)
                    .expand(n_kv, n_rep, *k.shape[1:])
                    .reshape(n_batch, *k.shape[1:])
                )
                v = (
                    v.unsqueeze(1)
                    .expand(n_kv, n_rep, *v.shape[1:])
                    .reshape(n_batch, *v.shape[1:])
                )
                n_rep = 1
            else:
                # (batch, head, time1, d_k) -> (group, head, n_rep * time1, d_k)
                q = (
                    q.view(n_kv, n_rep, self.h, time1, self.d_k)
                    .transpose(1, 2)
                    .reshape(n_kv, self.h, n_rep * time1, self.d_k)
                )
                if mask is not None and mask.size(0) == n_batch:
                    # every query of a group shares the mask of that group
                    mask = mask.view(n_kv, n_rep, 1, mask.size(-1))[:, 0]

        if getattr(self, "use_sdpa", False):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                mask.unsqueeze(1) if mask is not None else None,
                dropout_p=self.dropout_rate if self.training else 0.0,
            )  # (group, head, n_rep * time1, d_k)
            out = out.transpose(1, 2)
            out = self.linear_out(out.reshape(out.shape[0], out.shape[1], -1))
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            out = self.forward_attention(v, scores, mask)
            if n_rep > 1:
                # keep `self.attn` in its usual (batch, head, time1, time2) layout
                time2 = self.attn.size(-1)
                self.attn = (
                    self.attn.view(n_kv, self.h, n_rep, time1, time2)
                    .transpose(1, 2)
                    .reshape(n_batch, self.h, time1, time2)
                )
        # (group, n_rep * time1, d_model) -> (batch, time1, d_model): row
        # b * n_rep + i of the batch is group b, repeat i, in this order
        return out.reshape(n_batch, time1, -1)

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
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

    def forward(self, query, key, value, mask, expand_kv=False, kv=None):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            expand_kv (bool): Used only for partially autoregressive (PAR) decoding.
                When set to `True`, `Linear` layers are computed only for the first
                batch. This is useful to reduce the memory usage during decoding
                when the batch size is #beam_size x #mask_count, which can be large.
                Typically, in single waveform inference of PAR, `Linear` layers
                should not be computed for all batches for source-attention.
            kv (tuple[torch.Tensor, torch.Tensor]): Key and value already
                projected by :meth:`project_kv`. When given, ``key`` and
                ``value`` are not read at all. The projections may have one row
                per query or one row per group of consecutive queries; see
                :meth:`_forward_with_kv`. Used by the transformer decoder to
                project the encoder output once per utterance instead of once
                per decoding step and hypothesis.

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        if kv is not None:
            return self._forward_with_kv(query, kv, mask)

        # Use PyTorch's Scaled Dot Product Attention implementation
        if getattr(self, "use_sdpa", False):
            q, k, v = self.forward_qkv(query, key, value, expand_kv)

            # The shape of mask must be broadcastable to the shape of attention weights
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                mask.unsqueeze(1) if mask is not None else None,
                dropout_p=self.dropout_rate if self.training else 0.0,
            )  # (batch, head, time1, d_k)

            out = out.transpose(1, 2)  # (batch, time1, head, d_k)
            out = out.reshape(out.shape[0], out.shape[1], -1)  # (batch, time1, d_model)
            return self.linear_out(out)  # (batch, time1, d_model)

        # Use Flash Attention implementation
        if self.use_flash_attn:
            try:
                # In the causal case, the last row will be the key mask
                key_nonpad_mask = mask[:, -1, :]  # (#batch, time2)
                if self.cross_attn:
                    # For cross attention, we do not know the query padding
                    query_nonpad_mask = torch.ones(
                        size=query.shape[:2], dtype=torch.bool, device=query.device
                    )
                else:
                    query_nonpad_mask = key_nonpad_mask

                if key_nonpad_mask.eq(0).any():
                    # Use variable length implementation if padded
                    q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
                        query, query_nonpad_mask
                    )[:4]
                    k, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(
                        key, key_nonpad_mask
                    )[:4]
                    v, _, _, _ = unpad_input(value, key_nonpad_mask)[:4]

                    q = self.linear_q(q).reshape(-1, self.h, self.d_k)
                    k = self.linear_k(k).reshape(-1, self.h, self.d_k)
                    v = self.linear_v(v).reshape(-1, self.h, self.d_k)

                    q = self.q_norm(q)
                    k = self.k_norm(k)

                    out = flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        dropout_p=self.dropout_rate if self.training else 0.0,
                        causal=self.causal,
                    )  # (total, nheads, headdim)

                    out = out.reshape(out.shape[0], -1)
                    out = self.linear_out(out)

                    out = pad_input(out, indices_q, query.shape[0], query.shape[1])
                    return out

                else:
                    # Use fixed length implementation if not padded,
                    # which is faster than the variable length implementation
                    del key_nonpad_mask
                    q, k, v = self.forward_qkv(query, key, value)

                    out = flash_attn_func(
                        q.transpose(1, 2),
                        k.transpose(1, 2),
                        v.transpose(1, 2),
                        dropout_p=self.dropout_rate if self.training else 0.0,
                        causal=self.causal,
                    )  # (batch_size, seqlen, nheads, headdim)
                    del q, k, v

                    out = out.reshape(out.shape[0], out.shape[1], -1)
                    out = self.linear_out(out)
                    return out

            except Exception as e:
                logging.warning(
                    f"Flash Attention failed, falling back to default attention: {e}"
                )
                self.use_flash_attn = False

        # Fall back to the default implementation
        q, k, v = self.forward_qkv(query, key, value, expand_kv)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class LegacyRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, time2).

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor (#batch, time1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, 2*time1-1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)
