#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    GPTNeoXForCausalLM,
    GPTNeoXModel,
)
from espnet2.speechlm.module.abs_transformer import AbsTransformer

HF_OBJ = {
    "EleutherAI/pythia": [GPTNeoXModel, GPTNeoXForCausalLM],
    "Qwen/Qwen2": [AutoModel, AutoModelForCausalLM],
    "allenai/OLMo": [AutoModel, AutoModelForCausalLM],
    "meta-llama/Meta-Llama-3.1": [AutoModel, AutoModelForCausalLM],
    "HuggingFaceTB/SmolLM": [AutoModel, AutoModelForCausalLM],
    "facebook/opt": [AutoModel, AutoModelForCausalLM],
}

class HFTransformerDecoder(AbsTransformer):
    """Unified interface of ESPnet Transformer and HuggingFace Transformer"""

    def __init__(
        self,
        hf_model_tag: str,
        token_bias: dict,
        attention_choice: str = "sdpa",
        activation_checkpointing: bool = False,
        n_ctx: int = 8192,
    ):
        super(HFTransformerDecoder, self).__init__()

        base_class, causal_class = None, None
        for name in HF_OBJ.keys():
            if hf_model_tag.startswith(name):
                base_class, causal_class = HF_OBJ[name]
                break
        if base_class is None and causal_class is None:
            raise ValueError(f"HF model {hf_model_tag} is not supported yet")

        # NOTE(Jinchuan): lm_head and emb are only used in self.init_embeddings
        # and then removed. So this object only contains the transformer body,
        # i.e., self.model
        self.lm_head = causal_class.from_pretrained(
            hf_model_tag,
            attn_implementation=attention_choice,
        ).get_output_embeddings()
        
        self.model = base_class.from_pretrained(
            hf_model_tag,
            attn_implementation=attention_choice,
        )
        self.emb = self.model.get_input_embeddings()

        if activation_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.kv_cache = None
        self.use_cache = False

        self.token_bias = token_bias.copy()
        self.d_model = self.lm_head.in_features
        self._n_ctx = n_ctx
        self.attention_choice = attention_choice

    def forward(
        self,
        x: torch.Tensor,
        pos_id: torch.Tensor = None,
    ):
        if pos_id is not None:
            assert self.attention_choice  == "flash_attention_2"

        output = self.model(
            inputs_embeds=x,
            position_ids=pos_id,
            past_key_values=self.kv_cache,
            use_cache=self.use_cache,
        )

        self.kv_cache = output.past_key_values

        return output.last_hidden_state

    def init(self):
        self.use_cache = True

    def reset(self):
        self.kv_cache = None
        self.use_cache = False
    
    @property
    def n_ctx(self):
        return self._n_ctx

    @torch.no_grad()
    def init_embeddings(self, emb, lm_head):
        """When using HF pretrained model, inherit the embeddings and lm_head"""
        if (
            "text_bpe" not in self.token_bias
            or (self.emb is None or self.lm_head is None)
        ):
            del self.lm_head, self.emb
            return

        # find the range of text vocab
        start, end = self.token_bias["text_bpe"]

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
