#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
from typing import Optional

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    GPTNeoXForCausalLM,
    GPTNeoXModel,
)
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
)

from espnet2.speechlm.net_utils import install_kv_cache_hook

HF_OBJ = {
    "EleutherAI/pythia": [GPTNeoXModel, GPTNeoXForCausalLM],
    "Qwen/Qwen2": [AutoModel, AutoModelForCausalLM],
    "allenai/OLMo": [AutoModel, AutoModelForCausalLM],
    "meta-llama/Meta-Llama-3.1": [AutoModel, AutoModelForCausalLM],
}

class TransformerDecoder(torch.nn.Module):
    """ Unified interface of ESPnet Transformer and HuggingFace Transformer """

<<<<<<< HEAD
=======

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int, causal: bool = False):
        super().__init__()
        assert n_state % n_head == 0
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.causal = causal

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ValueError("Install torch 2.0.1+ to support Flash Attention")

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None)
            #   will prepend the cached kv tensors;
            # otherwise, perform key/value projections
            #   for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once
            # then reuse in subsequent calls.
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

        wv = (
            F.scaled_dot_product_attention(q, k, v, mask, is_causal=causal)
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
    ):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head, causal=causal)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class TransformerDecoder(nn.Module):
>>>>>>> b1046403ec7a20469594cb9f6ad3cbe58a7e6c81
    def __init__(
        self,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        qk_norm: bool,
        dropout: float,
        hf_model_tag: str = None,
        token_bias: dict = None,
    ):
        super(TransformerDecoder, self).__init__()

        if hf_model_tag is None:
            logging.info("Build Transformer Decoder with internal implementation")
            from espnet2.speechlm.module.builtin import (
                TransformerDecoder as BuiltinTransformerDecoder,
            )

<<<<<<< HEAD
            self.model = BuiltinTransformerDecoder(
                n_ctx=n_ctx,
                n_state=n_state,
                n_head=n_head,
                n_layer=n_layer,
                qk_norm=qk_norm,
                dropout=dropout,
            )

            self.emb = None
            self.lm_head = None

            self.model_type = "builtin"
            self.hooks = dict()

        else:
            logging.info(f"Building Transformer Decoder with HF model: {hf_model_tag}")

            import transformers
            if not (is_flash_attn_2_available(), is_flash_attn_greater_or_equal_2_10):
                logging.warning("Flash Attention is not properly used")

            base_class, causal_class = None, None
            for name in HF_OBJ.keys():
                if hf_model_tag.startswith(name):
                    base_class, causal_class = HF_OBJ[name]
                    break
            if base_class is None and causal_class is None:
                raise ValueError(f"HF model {hf_model_tag} is not supported yet")
=======
        self.blocks = nn.ModuleList(
            [layer_class(n_state, n_head, False, causal) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        self.causal = causal

    def forward(
        self, x: Tensor, mask: torch.Tensor = None, kv_cache: Optional[dict] = None
    ):
        if self.causal and mask is not None:
            raise ValueError("Causal Transformer dones't allow mask")

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = x + self.pos_emb.weight[offset : offset + x.shape[1]].unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask=mask, kv_cache=kv_cache)
>>>>>>> b1046403ec7a20469594cb9f6ad3cbe58a7e6c81

            self.lm_head = causal_class.from_pretrained(
                hf_model_tag
            ).get_output_embeddings()
            self.model = base_class.from_pretrained(hf_model_tag)
            self.emb = self.model.get_input_embeddings()

            self.model_type = "huggingface"

        self.token_bias = token_bias

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        kv_cache: Optional[dict] = None,
    ):
        if self.model_type == "builtin":
            return self.model(x=x, mask=mask, kv_cache=kv_cache)
        else:
            return self.model(inputs_embeds=x).last_hidden_state

    def init(self, kv_cache):
        if self.model_type == "builtin":
            kv_cache, self.hooks = install_kv_cache_hook(self.model, kv_cache)
        else:
            pass

        return kv_cache

    def reset(self, kv_cache):
        if self.model_type == "builtin":
            for hook in self.hooks:
                hook.remove()

            kv_cache.clear()

        else:
            pass

    @torch.no_grad()
    def init_embeddings(self, emb, lm_head):
        """When using HF pretrained model, inherit the embeddings and lm_head"""
        if (
            self.model_type == "builtin" or
            "text_bpe" not in self.token_bias or
            (self.emb is None or self.lm_head is None)
        ):
            del self.lm_head, self.emb
            return

        # find the range of text vocab
        vocab_size = emb.weight.size(0)
        start = self.token_bias["text_bpe"]
        values = [v for v in self.token_bias.values() if v > start] + [vocab_size]
        values.sort()
        end = values[0]

        # fulfill the pre-trained vocab from HF model. Other non-text embeddings
        # should have the same variance as the text embedding table.
        assert end - start == self.emb.weight.size(0)
        assert end - start == self.lm_head.weight.size(0)
        assert self.emb.weight.size(1) == emb.weight.size(1)
        assert self.lm_head.weight.size(1) == lm_head.weight.size(1)

        std = torch.var(self.emb.weight.data, dim=None)
        torch.nn.init.normal_(emb.weight, mean=0, std=std)
        emb.weight[start:end] = self.emb.weight

        std = torch.var(self.lm_head.weight.data, dim=None)
        torch.nn.init.normal_(lm_head.weight, mean=0, std=std)
        lm_head.weight[start:end] = self.lm_head.weight

        # clean up the original embeddings
        self.model.set_input_embeddings(None)
        del self.lm_head, self.emb
