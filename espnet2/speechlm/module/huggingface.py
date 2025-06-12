#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch

from espnet2.speechlm.module.abs_transformer import AbsTransformer


class HFTransformerDecoder(AbsTransformer):
    """Unified interface of ESPnet Transformer and HuggingFace Transformer"""

    def __init__(
        self,
        hf_model_tag: str,
        token_bias: dict,
        revision: str = None,
        attention_choice: str = "sdpa",
        activation_checkpointing: bool = False,
        dtype: str = "bfloat16",
        n_ctx: int = 8192,
    ):
        super(HFTransformerDecoder, self).__init__()
        from transformers import AutoModel, AutoModelForCausalLM

        # NOTE(Jinchuan): lm_head and emb are only used in self.init_embeddings
        # and then removed. So this object only contains the transformer body,
        # i.e., self.model
        self.lm_head = AutoModelForCausalLM.from_pretrained(
            hf_model_tag,
            attn_implementation=attention_choice,
            torch_dtype=dtype,
        ).get_output_embeddings()

        self.model = AutoModel.from_pretrained(
            hf_model_tag,
            attn_implementation=attention_choice,
            revision=revision,
            torch_dtype=dtype,
        )
        self.emb = self.model.get_input_embeddings()

        if activation_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.kv_cache = None
        self.use_cache = False

        self.token_bias = token_bias.copy()
        self.d_model = self.lm_head.in_features
        self._n_ctx = n_ctx

    def forward(
        self,
        x: torch.Tensor,
    ):

        output = self.model(
            inputs_embeds=x,
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
        if "text_bpe" not in self.token_bias or (
            self.emb is None or self.lm_head is None
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

        # The input padding vector should still be 0 vector
        emb.weight[0] = 0

        # clean up the original embeddings
        self.model.set_input_embeddings(None)
        del self.lm_head, self.emb
