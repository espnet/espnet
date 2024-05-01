#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Tuple, Dict
import torch

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet2.speechlm.module.module import (
    TransformerLayer,
    PositionalEncoding,
    MultiHeadedAttention
)
from espnet2.speechlm.net_utils import length_mask


class ARCoreLM(AbsCoreLM):
    def __init__(
        self,
        encoder_decoder_format: bool,
        pos_enc: str = None,
        att_unit: int = 256,
        head: int = 2,
        unit: int = 1024,
        encoder_layer: int = 4,
        decoder_layer: int = 4,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        flashattention: bool = False,
        causal_encoder: bool = False,
    ):
        super(ARCoreLM, self).__init__()

        if pos_enc == "sinusoidal":
            pos_enc_class = PositionalEncoding
        else:
            raise ValueError(f"unknown pos-enc option: {pos_enc}")

        self.decoders = torch.nn.ModuleList(
            [
                TransformerLayer(
                    att_unit=att_unit,
                    head=head,
                    unit=unit,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    causal=True,
                    flashattention=flashattention,
                    cross_attention=encoder_decoder_format,
                )
                for _ in range(decoder_layer)
            ]
        )
        self.decoder_post_ln = torch.nn.LayerNorm(att_unit)
        self.dec_pos_enc = pos_enc_class(att_unit, positional_dropout_rate)

        if encoder_decoder_format:
            self.encoders = torch.nn.ModuleList(
                [
                    TransformerLayer(
                        att_unit=att_unit,
                        head=head,
                        unit=unit,
                        dropout_rate=dropout_rate,
                        attention_dropout_rate=attention_dropout_rate,
                        causal=causal_encoder,
                        flashattention=flashattention,
                        cross_attention=False,
                    )
                    for _ in range(encoder_layer)
                ]
            )
            self.encoder_post_ln = torch.nn.LayerNorm(att_unit)
            self.enc_pos_enc = pos_enc_class(att_unit, positional_dropout_rate)
        else:
            self.encoders = None
            self.encoder_post_ln = None
            self.enc_pos_enc = None

        self._encoder_decoder_format = encoder_decoder_format
        self._model_dim = att_unit

        # Inference KV-cache related
        self.cache = {}
        self.hook = {}

    def forward(
        self,
        decoder_input: torch.Tensor,
        decoder_input_lengths: torch.Tensor = None,
        encoder_input: torch.Tensor = None,
        encoder_input_lengths: torch.Tensor = None,
        cache: Dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        if decoder_input.dim() == 4:
            decoder_input = decoder_input.sum(dim=2)
        if encoder_input is not None and encoder_input.dim() == 4:
            encoder_input = encoder_input.sum(dim=2)

        if self.encoder_decoder_format:
            assert encoder_input is not None and encoder_input_lengths is not None
            
            encoder_input = self.enc_pos_enc(encoder_input)
            encoder_mask = length_mask(encoder_input_lengths).unsqueeze(1)
            for layer in self.encoders:
                encoder_input = layer(
                    encoder_input,
                    encoder_mask,
                    cache=self.cache,
                )
            encoder_output = self.encoder_post_ln(encoder_input)
        else:
            encoder_output = None
            encoder_mask = None

        decoder_input = self.dec_pos_enc(decoder_input, self.cache)
        for layer in self.decoders:
            decoder_input = layer(
                decoder_input,
                None,
                encoder_output,
                encoder_mask,
                cache=self.cache,
            )
        decoder_output = self.decoder_post_ln(decoder_input)

        return decoder_output, decoder_input_lengths, {}

    def init_cache(self):
        self.cache, self.hooks = install_kv_cache_hook(self.decoders, {})
    
    def remove_cache(self):
        self.cache = {}
        for hook in self.hooks:
            hook.remove()

def install_kv_cache_hook(model, cache):
    cache = {**cache} if cache is not None else {}
    hooks = []

    def save_to_cache(module, _, output):
        if module not in cache:
            # save as-is, for the first token or cross attention
            cache[module] = output
        else:
            cache[module] = torch.cat([cache[module], output], dim=1).detach()
        return cache[module]

    def install_hooks(layer: torch.nn.Module):
        if isinstance(layer, MultiHeadedAttention):
            hooks.append(layer.linear_k.register_forward_hook(save_to_cache))
            hooks.append(layer.linear_v.register_forward_hook(save_to_cache))

    model.apply(install_hooks)
    return cache, hooks