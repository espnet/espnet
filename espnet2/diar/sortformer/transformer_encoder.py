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
from typing import Optional

import torch
import torch.nn as nn

NEG_INF = -10000.0


def form_attention_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Additive attention mask of shape (B, 1, L, L), 0 to attend / NEG_INF to mask."""
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
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden, attention_mask):
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
    """Post-LayerNorm Transformer encoder block."""

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

    def forward(self, x, attention_mask):
        attn = self.self_attn(x, attention_mask)
        x = self.self_attn_layer_norm(attn + x)
        ff = self.fc2(torch.relu(self.fc1(x)))
        ff = self.ffn_dropout(ff)
        x = self.final_layer_norm(ff + x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 18,
        hidden_size: int = 192,
        inner_size: int = 768,
        num_attention_heads: int = 8,
        attn_score_dropout: float = 0.5,
        attn_layer_dropout: float = 0.5,
        ffn_dropout: float = 0.5,
    ):
        super().__init__()
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

    def forward(
        self, encoder_states: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """encoder_states: (B, L, H). lengths: (B,) valid frame counts."""
        if lengths is not None:
            attn_mask = form_attention_mask(lengths, encoder_states.size(1))
        else:
            attn_mask = None
        x = encoder_states
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x
