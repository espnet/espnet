"""Efficient sliding-window (local) relative-position attention.

A memory-efficient O(N*W) local self-attention with the Transformer-XL relative
positional term, for long-form encoding (process a whole long recording in one
pass instead of O(N^2) full attention). This is the mechanism Parakeet-v3 uses
for long audio (``self_attention_model="rel_pos_local_attn"``).

``flash_attn`` / FlexAttention cannot express the Transformer-XL rel-pos term, so
the efficient path is the **chunked / overlapping-window** band matmul
(Longformer-style ``as_strided`` blocks), ported faithfully from NeMo:
``RelPositionMultiHeadAttentionLongformer`` (Apache-2.0).

Extras for streaming Sortformer: ``n_global`` leading frames (the speaker cache
prefix) are treated as **global tokens** — every query attends to them fully, and
those frames attend to everything — so the global speaker summary is always
visible regardless of the local window.

Reusable by any encoder using NeMo-convention Transformer-XL rel-pos attention
(this FastConformer port; ESPnet conformer / XEUS / Parakeet need a thin adapter
for their pos-emb layout — documented follow-on).
"""

import math
from functools import lru_cache
from typing import List, Tuple

import torch
import torch.nn.functional as F

from espnet2.diar.sortformer.fastconformer_encoder import (
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
)

INF_VAL = 10000.0


# --------------------------------------------------------------------------- #
# Shared band-attention kernel (Longformer-style overlapping chunks, O(N*W)).
# Ported verbatim from NeMo RelPositionMultiHeadAttentionLongformer.
# --------------------------------------------------------------------------- #
def _skew(x, direction: List[int], padding_value: float):
    x = F.pad(x, direction, value=padding_value)
    return x.view(*x.size()[:-2], x.size(-1), x.size(-2))


def _skew2(x, padding_value: float):
    b, c, m, ll = x.size()
    x = F.pad(x, (0, m + 1), value=padding_value)
    x = x.view(b, c, -1)
    x = x[:, :, :-m]
    x = x.view(b, c, m, m + ll)
    x = x[:, :, :, :-1]
    return x


def _chunk_overlap(x, w: int):
    x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))
    size = list(x.size())
    size[1] = size[1] * 2 - 1
    stride = list(x.stride())
    stride[1] = stride[1] // 2
    return x.as_strided(size=size, stride=stride)


@lru_cache()
def _invalid_locations_mask(w: int, device):
    diagonals = []
    for j in range(-w, 1):
        d = torch.zeros(w, device="cpu", dtype=torch.uint8)
        d[:-j] = 1
        diagonals.append(d)
    mask = torch.stack(diagonals, dim=-1)[None, None, :, :]
    ending = mask.flip(dims=(2, 3)).bool().to(device)
    return mask.bool().to(device), ending


def mask_invalid_locations(x, w: int):
    beg, end = _invalid_locations_mask(w, x.device)
    seq_len = x.size(2)
    bi = x[:, :, :w, : w + 1]
    bi.masked_fill_(beg[:, :, :seq_len].expand(bi.size()), -float("inf"))
    ei = x[:, :, -w:, -(w + 1) :]
    ei.masked_fill_(end[:, :, -seq_len:].expand(ei.size()), -float("inf"))


def sliding_chunks_matmul_qk(q, k, w: int, padding_value: float):
    """Banded q@k^T over a sliding window -> (B, H, T, 2w+1). O(N*W) memory."""
    bsz, n_head, seqlen, head_dim = q.size()
    chunks_count = seqlen // w - 1
    q = q.reshape(bsz * n_head, seqlen, head_dim)
    k = k.reshape(bsz * n_head, seqlen, head_dim)
    cq = _chunk_overlap(q, w)
    ck = _chunk_overlap(k, w)
    chunk_attn = torch.einsum("bcxd,bcyd->bcxy", (cq, ck))
    diag = _skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)
    out = diag.new_empty((bsz * n_head, chunks_count + 1, w, w * 2 + 1))
    out[:, :-1, :, w:] = diag[:, :, :w, : w + 1]
    out[:, -1, :, w:] = diag[:, -1, w:, : w + 1]
    out[:, 1:, :, :w] = diag[:, :, -(w + 1) : -1, w + 1 :]
    out[:, 0, 1:w, 1:w] = diag[:, 0, : w - 1, 1 - w :]
    out = out.view(bsz, n_head, seqlen, 2 * w + 1)
    mask_invalid_locations(out, w)
    return out


def sliding_chunks_matmul_pv(prob, v, w: int):
    """Banded attn@v over a sliding window -> (B, T, H, d_k)."""
    bsz, n_head, seqlen, head_dim = v.size()
    chunks_count = seqlen // w - 1
    chunk_prob = prob.reshape(bsz * n_head, seqlen // w, w, 2 * w + 1)
    v = v.reshape(bsz * n_head, seqlen, head_dim)
    padded_v = F.pad(v, (0, 0, w, w), value=-1)
    size = (bsz * n_head, chunks_count + 1, 3 * w, head_dim)
    stride = padded_v.stride()
    stride = stride[0], w * stride[1], stride[1], stride[2]
    chunk_v = padded_v.as_strided(size=size, stride=stride)
    skewed = _skew2(chunk_prob, padding_value=0)
    context = torch.einsum("bcwd,bcdh->bcwh", (skewed, chunk_v))
    return context.view(bsz, n_head, seqlen, head_dim).transpose(1, 2)


class LocalAttRelPositionalEncoding(RelPositionalEncoding):
    """Window-sized Transformer-XL positional table ``(1, left+right+1, d_model)``.

    Positions run from ``+left`` down to ``-right`` (NeMo convention), so the
    table indexes relative offsets within the local attention window.
    """

    def __init__(self, att_context_size, **kwargs):
        self.left_context, self.right_context = att_context_size
        super().__init__(**kwargs)

    def extend_pe(self, length, device, dtype):
        if self.pe is not None:
            self.pe = self.pe.to(device=device, dtype=dtype)
            return
        positions = torch.arange(
            self.left_context,
            -self.right_context - 1,
            -1,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x.size(1), x.device, x.dtype)
        if self.xscale:
            x = x * self.xscale
        return self.dropout(x), self.pe


class RelPositionLocalAttention(RelPositionMultiHeadAttention):
    """Sliding-window Transformer-XL attention (O(N*W)) with optional global prefix.

    Parameters/weights are identical to :class:`RelPositionMultiHeadAttention`
    (``q_proj/k_proj/v_proj/o_proj/relative_k_proj/bias_u/bias_v``), so converted
    or trained checkpoints load unchanged -- only the *compute* differs.
    """

    is_local = True  # marks the local-attention call convention for ConformerLayer

    def __init__(self, n_head, n_feat, dropout_rate, att_context_size):
        super().__init__(n_head, n_feat, dropout_rate)
        self.att_context_size = list(att_context_size)

    def forward(self, x, pos_emb, pad_mask, n_global: int = 0):
        """x: (B, T, d_model); pos_emb: (1, left+right+1, d_model);
        pad_mask: (B, T) True=pad; n_global: leading global-token (cache) count."""
        q, k, v = self.forward_qkv(x, x, x)  # (B, H, T, d_k)
        n_batch, _, T, _ = q.size()
        left, right = self.att_context_size
        w = max(left, right)
        pad_len = (2 * w - T % (2 * w)) % (2 * w)
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        mask = F.pad(pad_mask, (0, pad_len), value=True)  # True=pad

        qu = q + self.bias_u.unsqueeze(0).unsqueeze(2)  # (B,H,Tp,d_k)
        qv = q + self.bias_v.unsqueeze(0).unsqueeze(2)
        ac = sliding_chunks_matmul_qk(qu, k, w, padding_value=0.0)  # (B,H,Tp,2w+1)
        p = self.relative_k_proj(pos_emb).view(pos_emb.size(0), -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (1, H, L, d_k)
        bd = torch.matmul(qv, p.transpose(-2, -1))  # (B,H,Tp,L)
        start_pos, end_pos = w - left, w + right
        ac[:, :, :, :left] += bd[:, :, :, :left]
        ac[:, :, :, -(right + 1) :] += bd[:, :, :, left:]
        scores = ac / self.s_d_k
        scores[:, :, :, :start_pos] = -INF_VAL
        scores[:, :, :, end_pos + 1 :] = -INF_VAL

        # validity (padding) mask, in band form
        fmask = mask.unsqueeze(1).unsqueeze(-1).type_as(scores)  # (B,1,Tp,1)
        fmask = fmask.masked_fill(mask.unsqueeze(1).unsqueeze(-1), -INF_VAL)
        ones = fmask.new_ones(fmask.size())
        d_mask = sliding_chunks_matmul_qk(ones, fmask, w, padding_value=0.0)
        scores += d_mask

        if n_global > 0:
            # Global keys: every query also attends (content-only) to the cache
            # prefix [0:n_global]. Mask band columns that point into the prefix to
            # avoid double counting.
            gk = k[:, :, :n_global, :]  # (B,H,g,d_k)
            gscore = torch.matmul(qu, gk.transpose(-2, -1)) / self.s_d_k  # (B,H,Tp,g)
            idx = torch.arange(scores.size(2), device=scores.device)
            col = torch.arange(2 * w + 1, device=scores.device)
            abs_key = idx.unsqueeze(1) - w + col.unsqueeze(0)  # (Tp, 2w+1)
            scores = scores.masked_fill(
                (abs_key < n_global).unsqueeze(0).unsqueeze(0), -INF_VAL
            )
            full = torch.cat([gscore, scores], dim=-1)  # (B,H,Tp, g+2w+1)
            attn = torch.softmax(full, dim=-1)
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(-1), 0.0)
            attn = self.dropout(attn)
            g_attn, band_attn = attn[..., :n_global], attn[..., n_global:]
            out = sliding_chunks_matmul_pv(band_attn, v, w)  # (B,Tp,H,d_k)
            out = out + torch.matmul(g_attn, v[:, :, :n_global, :]).transpose(1, 2)
            out = out.reshape(n_batch, -1, self.h * self.d_k)[:, :T]
            # Global queries (the cache prefix) attend fully to everything.
            out_g = self._full_attention(qu[:, :, :n_global], k, v, mask)
            out[:, :n_global] = out_g
        else:
            attn = torch.softmax(scores, dim=-1)
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(-1), 0.0)
            attn = self.dropout(attn)
            out = sliding_chunks_matmul_pv(attn, v, w)
            out = out.reshape(n_batch, -1, self.h * self.d_k)[:, :T]
        return self.o_proj(out)

    def _full_attention(self, qu_g, k, v, mask):
        """Full attention for the (few) global query rows -> (B, g, d_model)."""
        scores = torch.matmul(qu_g, k.transpose(-2, -1)) / self.s_d_k  # (B,H,g,Tp)
        scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), -INF_VAL)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B,H,g,d_k)
        b, h, g, dk = out.size()
        return out.transpose(1, 2).reshape(b, g, h * dk)


class TransformerLocalAttention(torch.nn.Module):
    """Standard (no positional bias) sliding-window attention, O(N*W).

    Drop-in for the Sortformer ``TransformerMultiHeadAttention`` (same projection
    names ``q_proj/k_proj/v_proj/out_proj`` and pre-scaling), so weights load
    unchanged. ``n_global`` leading frames (speaker cache) are global tokens.
    """

    is_local = True

    def __init__(
        self,
        hidden_size,
        num_heads,
        attn_score_dropout,
        attn_layer_dropout,
        att_context_size,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.attn_scale = math.sqrt(math.sqrt(self.head_size))
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = torch.nn.Dropout(attn_score_dropout)
        self.layer_dropout = torch.nn.Dropout(attn_layer_dropout)
        self.att_context_size = list(att_context_size)

    def _transpose(self, x):
        new = x.size()[:-1] + (self.num_heads, self.head_size)
        return x.view(*new).permute(0, 2, 1, 3)

    def forward(self, hidden, pad_mask, n_global: int = 0):
        """hidden: (B, T, H); pad_mask: (B, T) True=pad."""
        q = self._transpose(self.q_proj(hidden)) / self.attn_scale  # (B,H,T,d)
        k = self._transpose(self.k_proj(hidden)) / self.attn_scale
        v = self._transpose(self.v_proj(hidden))
        n_batch, _, T, _ = q.size()
        left, right = self.att_context_size
        w = max(left, right)
        pad_len = (2 * w - T % (2 * w)) % (2 * w)
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        mask = F.pad(pad_mask, (0, pad_len), value=True)

        scores = sliding_chunks_matmul_qk(q, k, w, padding_value=0.0)  # (B,H,Tp,2w+1)
        start_pos, end_pos = w - left, w + right
        scores[:, :, :, :start_pos] = -INF_VAL
        scores[:, :, :, end_pos + 1 :] = -INF_VAL
        fmask = mask.unsqueeze(1).unsqueeze(-1).type_as(scores)
        fmask = fmask.masked_fill(mask.unsqueeze(1).unsqueeze(-1), -INF_VAL)
        scores += sliding_chunks_matmul_qk(fmask.new_ones(fmask.size()), fmask, w, 0.0)

        if n_global > 0:
            gk = k[:, :, :n_global, :]
            gscore = torch.matmul(q, gk.transpose(-2, -1))  # (B,H,Tp,g)
            idx = torch.arange(scores.size(2), device=scores.device)
            col = torch.arange(2 * w + 1, device=scores.device)
            abs_key = idx.unsqueeze(1) - w + col.unsqueeze(0)
            scores = scores.masked_fill(
                (abs_key < n_global).unsqueeze(0).unsqueeze(0), -INF_VAL
            )
            full = torch.cat([gscore, scores], dim=-1)
            attn = torch.softmax(full, dim=-1).masked_fill(
                mask.unsqueeze(1).unsqueeze(-1), 0.0
            )
            attn = self.attn_dropout(attn)
            g_attn, band_attn = attn[..., :n_global], attn[..., n_global:]
            out = sliding_chunks_matmul_pv(band_attn, v, w)
            out = out + torch.matmul(g_attn, v[:, :, :n_global, :]).transpose(1, 2)
            out = out.reshape(n_batch, -1, self.num_heads * self.head_size)[:, :T]
            out[:, :n_global] = self._full_attention(q[:, :, :n_global], k, v, mask)
        else:
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask.unsqueeze(1).unsqueeze(-1), 0.0
            )
            attn = self.attn_dropout(attn)
            out = sliding_chunks_matmul_pv(attn, v, w)
            out = out.reshape(n_batch, -1, self.num_heads * self.head_size)[:, :T]
        return self.layer_dropout(self.out_proj(out))

    def _full_attention(self, q_g, k, v, mask):
        scores = torch.matmul(q_g, k.transpose(-2, -1))  # (B,H,g,Tp) already scaled
        scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), -INF_VAL)
        out = torch.matmul(torch.softmax(scores, dim=-1), v)  # (B,H,g,d)
        b, h, g, d = out.size()
        return out.transpose(1, 2).reshape(b, g, h * d)
