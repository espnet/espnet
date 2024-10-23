#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Derived from OpenAI Whisper model file:
# https://github.com/openai/whisper/blob/main/whisper/model.py

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from espnet2.speechlm.module.abs_transformer import AbsTransformer
from espnet2.speechlm.net_utils import install_kv_cache_hook


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        # return super().forward(x) # For full BF16 training
        return super().forward(x.float()).type(x.dtype)  # For AMP / FP32 training


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        causal: bool = False,
        qk_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert n_state % n_head == 0
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.causal = causal
        self.dropout = dropout

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = LayerNorm(n_state // n_head)
            self.k_norm = LayerNorm(n_state // n_head)

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ValueError("Install torch 2.0.1+ to support Flash Attention")

        try:
            from flash_attn import flash_attn_func

            self.flash_attn_func = flash_attn_func
        except:
            self.flash_attn_func = None

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv = self.qkv_attention(q, k, v, mask)

        return self.out(wv)

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        if self.causal and mask is not None:
            raise ValueError("mask is not allowed when the attention is causal")

        if self.causal and q.size(1) == k.size(1):
            causal = True
        else:
            causal = False

        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.flash_attn_func is not None and mask is None and self.training:
            wv = self.flash_attn_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=self.dropout,
                causal=causal,
            ).flatten(start_dim=2)
        else:
            wv = (
                F.scaled_dot_product_attention(
                    q, k, v, mask, is_causal=causal, dropout_p=self.dropout
                )
                .permute(0, 2, 1, 3)
                .flatten(start_dim=2)
            )

        return wv


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        cross_attention: bool = False,
        causal: bool = False,
        qk_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.attn = MultiHeadAttention(
            n_state,
            n_head,
            causal=causal,
            qk_norm=qk_norm,
            dropout=dropout,
        )
        self.attn_ln = LayerNorm(n_state)
        self.attn_dropout = nn.Dropout(p=dropout)

        self.cross_attn = (
            MultiHeadAttention(
                n_state,
                n_head,
                causal=False,
                qk_norm=qk_norm,
                dropout=dropout,
            )
            if cross_attention
            else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None
        self.cross_attn_dropout = nn.Dropout(p=dropout) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)
        self.mlp_dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn_dropout(
            self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        )
        if self.cross_attn:
            x = x + self.cross_attn_dropout(
                self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
            )
        x = x + self.mlp_dropout(self.mlp(self.mlp_ln(x)))
        return x


class TransformerDecoder(AbsTransformer):
    def __init__(
        self,
        token_bias: dict,
        n_ctx: int = 128,
        n_state: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        causal: bool = True,
        qk_norm: bool = False,
        dropout: float = 0.0,
        layer_class=ResidualAttentionBlock,
    ):
        super().__init__()

        self.pos_emb = nn.Embedding(n_ctx, n_state)

        self.blocks = nn.ModuleList(
            [
                layer_class(
                    n_state=n_state,
                    n_head=n_head,
                    cross_attention=False,
                    causal=causal,
                    qk_norm=qk_norm,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        self.causal = causal
        self.d_model = n_state
        self._n_ctx = n_ctx
        
        self.kv_cache = None
        self.hooks = None

    def forward(self, x: Tensor, mask: torch.Tensor = None):
        if self.causal and mask is not None:
            raise ValueError("Causal Transformer dones't allow mask")

        offset = next(iter(self.kv_cache.values())).shape[1] if self.kv_cache else 0
        x = x + self.pos_emb.weight[offset : offset + x.shape[1]].unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask=mask, kv_cache=self.kv_cache)

        x = self.ln(x)
        return x
    
    def init(self):
        self.kv_cache, self.hooks = install_kv_cache_hook(
            self.blocks, 
            self.kv_cache,
            attn_module=MultiHeadAttention,
        )

    def reset(self):
        for h in self.hooks:
            h.remove()
        self.kv_cache = None
        self.hooks = None
    
    def select_state(self, index):
        if self.kv_cache is None:
            raise ValueError("Transformer is not initialized or doesn't have kv_cache")
        
        for k, v in self.kv_cache.items():
            self.kv_cache[k] = v[index]
    
    @property
    def n_ctx(self):
        return self._n_ctx
        
