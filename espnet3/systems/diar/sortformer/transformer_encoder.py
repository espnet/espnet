"""NeMo-free port of the Transformer encoder used by offline Sortformer.

Reimplements ``nemo.collections.asr.modules.transformer.TransformerEncoder``
(post-LayerNorm variant, ``pre_ln=False``) used on top of the FastConformer
encoder in ``nvidia/diar_sortformer_4spk-v1``.

Notes on faithfulness to the released checkpoint (``tf_encoder.*``):

* The released model uses ``pre_ln=False`` and therefore has **no** top-level
  final LayerNorm.
* Positional information is *not* added in this encoder (the checkpoint stores
  an all-zero ``embed_positions`` buffer which is a no-op); the FastConformer's
  relative positions are the only positional signal. We therefore omit it.
* The key projection has **no bias**: in scaled dot-product attention a key
  bias is constant across key positions and cancels in the softmax, so NVIDIA's
  HF export dropped it. We match that exactly.

Parameter names mirror the HF checkpoint: ``self_attn.{q_proj,k_proj,v_proj,
out_proj}``, ``self_attn_layer_norm``, ``fc1``, ``fc2``, ``final_layer_norm``.

Reference (Apache-2.0): NVIDIA/NeMo
    nemo/collections/asr/modules/transformer/transformer_modules.py
    nemo/collections/asr/modules/transformer/transformer_encoders.py
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn

NEG_INF = -10000.0


def form_attention_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Build an additive attention mask from per-sequence valid lengths.

    Produces a mask that is added to the attention scores before softmax: ``0``
    for allowed positions and ``NEG_INF`` for padded key positions. Only padded
    *key* columns are masked (a padded query row is harmless because its output
    is discarded downstream), so the returned shape ``(B, 1, 1, L_k)`` broadcasts
    over heads and queries.

    Args:
        lengths: Valid frame counts ``(B,)``.
        max_len: Sequence length ``L`` to build the mask for.

    Returns:
        Additive mask of shape ``(B, 1, 1, L_k)`` (float), broadcastable to
        ``(B, num_heads, L_q, L_k)``.

    Example:
        >>> m = form_attention_mask(torch.tensor([2, 3]), max_len=3)
        >>> m.shape
        torch.Size([2, 1, 1, 3])
    """
    device = lengths.device
    valid = torch.arange(max_len, device=device).expand(
        lengths.size(0), max_len
    ) < lengths.unsqueeze(
        1
    )  # (B, L)
    # Mask padded *key* positions only; padded query rows are zeroed downstream
    # via the encoder mask. Shape (B, 1, 1, L_k) broadcasts over heads & queries.
    key_valid = valid.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L_k)
    mask = (1.0 - key_valid.to(torch.float)) * NEG_INF
    return mask


class TransformerMultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head self-attention (NeMo HF-export layout).

    Matches the released ``tf_encoder.*`` checkpoint:

    * the key projection has **no bias** (a constant key bias cancels in the
      softmax, so NVIDIA's export dropped it);
    * scaling is split symmetrically across query and key (each divided by
      ``head_size ** 0.25``), which is numerically equivalent to the usual
      ``1 / sqrt(head_size)`` on the scores.

    Parameter names mirror the checkpoint: ``q_proj``, ``k_proj``, ``v_proj``,
    ``out_proj``.

    Args:
        hidden_size: Model dimension (must be divisible by ``num_heads``).
        num_heads: Number of attention heads.
        attn_score_dropout: Dropout on the attention probabilities.
        attn_layer_dropout: Dropout on the output projection.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_score_dropout: float,
        attn_layer_dropout: float,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.attn_scale = math.sqrt(math.sqrt(self.head_size))
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attn_score_dropout)
        self.layer_dropout = nn.Dropout(attn_layer_dropout)

    def _transpose(self, x):
        """Reshape ``(B, L, H)`` into per-head ``(B, num_heads, L, head_size)``."""
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden, attention_mask):
        """Apply multi-head self-attention over ``hidden``.

        Args:
            hidden: Input sequence ``(B, L, hidden_size)``.
            attention_mask: Additive mask broadcastable to the attention scores
                (see :func:`form_attention_mask`); ``None`` for full attention.

        Returns:
            Attention output ``(B, L, hidden_size)``.
        """
        q = self._transpose(self.q_proj(hidden)) / self.attn_scale
        k = self._transpose(self.k_proj(hidden)) / self.attn_scale
        v = self._transpose(self.v_proj(hidden))
        scores = torch.matmul(q, k.transpose(-1, -2))
        if attention_mask is not None:
            scores = scores + attention_mask.to(scores.dtype)
        probs = self.attn_dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.num_heads * self.head_size,)
        context = context.view(*new_shape)
        return self.layer_dropout(self.out_proj(context))


class TransformerEncoderLayer(nn.Module):
    """Post-LayerNorm Transformer encoder block.

    Layout is ``LN(attn(x) + x)`` then ``LN(ffn(x) + x)`` with a ReLU feed-forward
    (``pre_ln=False``), matching the released checkpoint. The self-attention may
    be swapped for a sliding-window variant by :class:`TransformerEncoder` when
    ``att_context_size`` is set (detected at runtime via the ``is_local``
    attribute in :meth:`forward`).

    Args:
        hidden_size: Model dimension.
        inner_size: Feed-forward hidden size.
        num_heads: Attention heads.
        attn_score_dropout: Dropout on attention probabilities.
        attn_layer_dropout: Dropout on the attention output projection.
        ffn_dropout: Dropout on the feed-forward output.
    """

    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        num_heads: int,
        attn_score_dropout: float,
        attn_layer_dropout: float,
        ffn_dropout: float,
    ):
        super().__init__()
        self.self_attn = TransformerMultiHeadAttention(
            hidden_size, num_heads, attn_score_dropout, attn_layer_dropout
        )
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.fc1 = nn.Linear(hidden_size, inner_size)
        self.fc2 = nn.Linear(inner_size, hidden_size)
        self.ffn_dropout = nn.Dropout(ffn_dropout)
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, x, attention_mask, pad_mask=None, n_global=0):
        """Apply the post-LayerNorm Transformer block.

        Args:
            x: Input ``(B, L, hidden_size)``.
            attention_mask: Additive attention mask for the full-attention path;
                ignored on the sliding-window path.
            pad_mask: Padding mask ``(B, L)`` (True = padded) used as the key
                mask on the sliding-window path.
            n_global: Number of leading frames (speaker-cache prefix) kept
                globally attended on the sliding-window path; ignored otherwise.

        Returns:
            Block output ``(B, L, hidden_size)``.
        """
        if getattr(self.self_attn, "is_local", False):
            attn = self.self_attn(x, pad_mask, n_global=n_global)
        else:
            attn = self.self_attn(x, attention_mask)
        x = self.self_attn_layer_norm(attn + x)
        ff = self.fc2(torch.relu(self.fc1(x)))
        ff = self.ffn_dropout(ff)
        x = self.final_layer_norm(ff + x)
        return x


class TransformerEncoder(nn.Module):
    """Post-LayerNorm Transformer encoder stacked on top of the FastConformer.

    A stack of ``num_layers`` :class:`TransformerEncoderLayer` blocks with no
    positional embedding (the FastConformer's relative positions are the only
    positional signal) and no top-level final LayerNorm, matching the released
    ``tf_encoder.*`` checkpoint.

    Attention modes:
        * Full attention (default, ``att_context_size=None``): an additive mask
          is built from ``lengths`` via :func:`form_attention_mask`.
        * Efficient local attention (``att_context_size=[left, right]``): each
          block's attention is replaced at construction time by an O(L*W)
          sliding-window variant. The replacement loads the original weights, so
          checkpoints stay compatible; window units are frames and the leading
          ``n_global`` frames (speaker cache) remain globally attended.

    Args:
        num_layers: Number of Transformer blocks.
        hidden_size: Model dimension.
        inner_size: Feed-forward hidden size.
        num_attention_heads: Attention heads per block.
        attn_score_dropout: Dropout on attention probabilities.
        attn_layer_dropout: Dropout on the attention output projection.
        ffn_dropout: Dropout on the feed-forward output.
        att_context_size: Optional ``[left, right]`` local-attention window in
            frames; ``None`` keeps full attention.

    Example:
        >>> enc = TransformerEncoder(num_layers=18, hidden_size=192)
        >>> x = torch.randn(2, 100, 192)
        >>> out = enc(x, lengths=torch.tensor([100, 80]))
        >>> out.shape
        torch.Size([2, 100, 192])
    """

    def __init__(
        self,
        num_layers: int = 18,
        hidden_size: int = 192,
        inner_size: int = 768,
        num_attention_heads: int = 8,
        attn_score_dropout: float = 0.5,
        attn_layer_dropout: float = 0.5,
        ffn_dropout: float = 0.5,
        att_context_size: Optional[List[int]] = None,
    ):
        super().__init__()
        self.att_context_size = att_context_size
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_size,
                    inner_size,
                    num_attention_heads,
                    attn_score_dropout,
                    attn_layer_dropout,
                    ffn_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        if att_context_size is not None:
            # Efficient O(N*W) sliding-window attention (no positional bias; the
            # transformer is position-free). Same weights -> checkpoints load.
            from .sliding_window_attention import (
                TransformerLocalAttention,
            )

            for layer in self.layers:
                local = TransformerLocalAttention(
                    hidden_size,
                    num_attention_heads,
                    attn_score_dropout,
                    attn_layer_dropout,
                    att_context_size,
                )
                local.load_state_dict(layer.self_attn.state_dict())
                layer.self_attn = local

    def forward(
        self,
        encoder_states: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        n_global: int = 0,
    ) -> torch.Tensor:
        """Run the Transformer encoder.

        Args:
            encoder_states: Input ``(B, L, hidden_size)``.
            lengths: Valid frame counts ``(B,)``; ``None`` disables masking.
            n_global: Number of leading frames (speaker-cache prefix) kept
                globally attended on the local-attention path; ignored on the
                full-attention path.

        Returns:
            Encoder output ``(B, L, hidden_size)``.
        """
        x = encoder_states
        if self.att_context_size is not None:
            max_len = x.size(1)
            if lengths is not None:
                pad_mask = ~(
                    torch.arange(max_len, device=x.device).expand(
                        lengths.size(0), max_len
                    )
                    < lengths.unsqueeze(1)
                )
            else:
                pad_mask = x.new_zeros(x.size(0), max_len, dtype=torch.bool)
            for layer in self.layers:
                x = layer(x, None, pad_mask=pad_mask, n_global=n_global)
            return x
        attn_mask = (
            form_attention_mask(lengths, x.size(1)) if lengths is not None else None
        )
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x
