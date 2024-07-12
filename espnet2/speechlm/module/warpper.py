from typing import Optional

import torch
import logging

from espnet2.speechlm.net_utils import install_kv_cache_hook, logits_to_tokens

from transformers import(
    GPTNeoXModel,
    GPTNeoXForCausalLM,
    AutoModel,
    AutoModelForCausalLM,
)

HF_OBJ = {
    "EleutherAI/pythia": [GPTNeoXModel, GPTNeoXForCausalLM],
    "Qwen/Qwen2": [AutoModel, AutoModelForCausalLM],
}

class TransformerDecoder(torch.nn.Module):
    """ 
    The Transformer Decoder model implementation. It supports:
    (1) Builtin Espnet Implementation; and
    (2) HuggingFace models
    """
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
            from espnet2.speechlm.module.transformer import TransformerDecoder as BuiltinTransformerDecoder
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

        else:
            logging.info(f"Building Transformer Decoder with HF model: {hf_model_tag}")
            import transformers

            base_class, causal_class = None, None
            for name in HF_OBJ.keys():
                if hf_model_tag.startswith(name):
                    base_class, causal_class = HF_OBJ[name]
                    break
            if base_class is None and  causal_class is None:
                raise ValueError(f"HF model {hf_model_tag} is not supported yet")
            
            self.lm_head = causal_class.from_pretrained(hf_model_tag).get_output_embeddings()
            self.model = base_class.from_pretrained(hf_model_tag)
            self.emb = self.model.get_input_embeddings()
            
            self.model_type = "huggingface"

        self.token_bias = token_bias
        
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, kv_cache: Optional[dict] = None
    ):
        if self.model_type == "builtin":
            return self.model(x=x, mask=mask, kv_cache=kv_cache)
        else:
            return self.model(inputs_embeds=x,).last_hidden_state
    
    @torch.no_grad()
    def init_embeddings(self, emb, lm_head):
        """ When using HF pretrained model, inherit the embeddings and lm_head """
        if self.model_type == "builtin":
            return

        if "text_bpe" not in self.token_bias:
            return
        
        if self.emb is None or self.lm_head is None:
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