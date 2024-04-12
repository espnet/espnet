#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Tuple, Dict
from packaging import version

import math
import torch

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet2.speechlm.net_utils import length_mask, causal_mask


class MultiHeadedAttention(torch.nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate, causal=False, flashattention=True):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = torch.nn.Linear(n_feat, n_feat)
        self.linear_k = torch.nn.Linear(n_feat, n_feat)
        self.linear_v = torch.nn.Linear(n_feat, n_feat)
        self.linear_out = torch.nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.dropout_rate = dropout_rate
        self.causal = causal
        self.flashattention = flashattention

        # check flashattention availability
        torch_version = torch.__version__
        if flashattention and version.parse(torch_version) < version.parse("2.0.1"):
            raise ValueError(f"Upgrade Pytorch 2.0.1+ to use flashattention")

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

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

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        if not self.flashattention:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            return self.forward_attention(v, scores, mask)
        else:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout=self.dropout_rate,
                is_causal=self.causal,
                scale=None,
            )
            return self.linear_out(x)


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        att_unit: int = 256,
        head: int = 2,
        unit: int = 1024,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        causal: bool = True,
        flashattention: bool = False,
        cross_attention: bool = False,
    ):
        super(TransformerLayer, self).__init__()

        self.attn = MultiHeadedAttention(
            head, 
            att_unit,
            attention_dropout_rate, 
            causal,
            flashattention,
        )
        self.attn_ln = torch.nn.LayerNorm(att_unit)

        if cross_attention:
            self.cross_attn = MultiHeadedAttention(
                head, 
                att_unit, 
                attention_dropout_rate, 
                causal,
                flashattention,
            )
            self.cross_attn_ln = torch.nn.LayerNorm(att_unit)
        else:
            self.cross_attn = None
            self.cross_attn_ln = None
        
        self.ffn = PositionwiseFeedForward(att_unit, unit, dropout_rate)
    
    def forward(
        self,
        input: torch.Tensor,
        input_masks: torch.Tensor = None,
        src: torch.Tensor = None,
        src_masks: torch.Tensor = None,
        cache: Dict = None
    ):
        # self-attn
        x = self.attn_ln(input)
        x = x + self.attn(x, x, x, input_masks, cache)

        # cross-attn
        if self.cross_attn and src is not None:
            x = self.cross_attn_ln(x)
            x = x + self.cross_attn(x, x, src, src_masks, cache)
        
        # feed-forward
        x = x + self.ffn(x)

        return x
    
class BuiltinCoreLM(AbsCoreLM):
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
    ):
        super(BuiltinCoreLM, self).__init__()

        if pos_enc == "sinusoidal":
            pos_enc_class = PositionalEncoding
        else:
            raise ValueError(f"unknown pos-enc option: {pos_enc}")
        
        self.decoders = torch.nn.ModuleList([
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
        ])
        self.decoder_post_ln = torch.nn.LayerNorm(att_unit)
        self.dec_pos_enc = pos_enc_class(att_unit, positional_dropout_rate)

        if encoder_decoder_format:
            self.encoders = torch.nn.ModuleList([
                TransformerLayer(
                    att_unit=att_unit,
                    head=head,
                    unit=unit,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    causal=False,
                    flashattention=flashattention,
                    cross_attention=False,
                )
                for _ in range(encoder_layer)
            ])
            self.encoder_post_ln = torch.nn.LayerNorm(att_unit)
            self.enc_pos_enc = pos_enc_class(att_unit, positional_dropout_rate)
        else:
            self.encoders = None
            self.encoder_post_ln = None
            self.enc_pos_enc= None

        self._model_dim = att_unit
    
    def model_dim(self) -> int:
        return self._model_dim
        
    def forward(
        self, 
        decoder_input: torch.Tensor, 
        decoder_input_lengths: torch.Tensor,
        encoder_input: torch.Tensor = None,
        encoder_input_lengths: torch.Tensor = None,
        cache: Dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        
        if self.encoder_decoder_format:
            assert encoder_input is not None and encoder_input_lengths is not None

            encoder_input = self.enc_pos_enc(encoder_input)
            encoder_mask = length_mask(encoder_input_lengths)
            for layer in self.encoders:
                encoder_input = layer(
                    encoder_input, 
                    encoder_mask, 
                    cache=cache,
                )
            encoder_output = self.encoder_post_ln(encoder_input)
        else:
            encoder_output = None
            encoder_mask = None

        decoder_input = self.dec_pos_enc(decoder_input)
        for layer in self.decoders:
            decoder_input = layer(
                decoder_input,
                None,
                encoder_output,
                encoder_mask,
                cache=cache,
            )
        decoder_output = self.decoder_post_ln(decoder_output)

        return decoder_output, decoder_input_lengths