# Copyright 2025 Qingzheng Wang, 2026 ESPnet contributors
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""GeoLID conditioning implemented using an unmodified Transformers installation.

This file is also copied into exported Hugging Face checkpoints. Keep it independent
of ESPnet and S3PRL. Layer indices follow the released checkpoints: index i means
*after* zero-based Transformer block i, before the final encoder layer norm.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput, Wav2Vec2BaseModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2EncoderStableLayerNorm,
)


@dataclass
class ConditionEncoderOutput(BaseModelOutput):
    intermediate_lang2vec_preds: Optional[Tuple[torch.Tensor, ...]] = None


@dataclass
class ConditionModelOutput(Wav2Vec2BaseModelOutput):
    intermediate_lang2vec_preds: Optional[Tuple[torch.Tensor, ...]] = None


class Wav2Vec2EncoderCondition(Wav2Vec2EncoderStableLayerNorm):
    """Stable-layer-norm encoder with utterance-level residual conditioning."""

    def __init__(self, config):
        super().__init__(config)
        self.ecapa_encoder = None
        self.pooling = None
        self.projector = None
        self.lang2vec_head = None
        self.lang2vec_conditioning_projs = None
        self.aamsoftmax_loss = None
        self.lang2vec_conditioning_layers = []
        self.apply_intermediate_lang2vec_condition = True
        self.shared_conditioning_proj = False
        self.cutoff_gradient_from_backbone = False
        self.cutoff_gradient_before_condproj = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        predictions = ()
        lengths = hidden_states.new_full(
            (hidden_states.size(0),), hidden_states.size(1), dtype=torch.long
        )
        if attention_mask is not None:
            lengths = attention_mask.long().sum(-1)
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
            attention_mask = (
                1.0 - attention_mask[:, None, None, :].to(hidden_states.dtype)
            ) * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(-1, 1, hidden_states.size(1), -1)

        hidden_states = self.dropout(hidden_states + self.pos_conv_embed(hidden_states))
        # LayerDrop was disabled when the published GeoLID models were trained.
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__, hidden_states, attention_mask, output_attentions
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )
            hidden_states = layer_outputs[0]
            if i in (self.lang2vec_conditioning_layers or []):
                key = str(i)
                frames = self.ecapa_encoder[key](hidden_states)
                pooled = self.pooling[key](frames, feat_lengths=lengths)
                embedding = self.projector[key](pooled)
                prediction = self.lang2vec_head[key](embedding)
                predictions += (prediction,)
                if self.apply_intermediate_lang2vec_condition:
                    value = (
                        prediction.detach()
                        if self.cutoff_gradient_before_condproj
                        else prediction
                    )
                    projection = (
                        self.lang2vec_conditioning_projs
                        if self.shared_conditioning_proj
                        else self.lang2vec_conditioning_projs[key]
                    )
                    condition = projection(value).unsqueeze(1)
                    if self.cutoff_gradient_from_backbone:
                        condition = condition.detach()
                    hidden_states = hidden_states + condition
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        output = ConditionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            intermediate_lang2vec_preds=predictions,
        )
        return output if return_dict else output.to_tuple()


class Wav2Vec2ModelCondition(Wav2Vec2Model):
    """Wav2Vec2 with conditioning; accepts ordinary Wav2Vec2 pretrained weights."""

    def __init__(self, config):
        if not config.do_stable_layer_norm:
            raise ValueError("GeoLID requires do_stable_layer_norm=True (e.g. MMS).")
        if config._attn_implementation not in (None, "eager", "sdpa"):
            raise ValueError("GeoLID supports eager and sdpa attention.")
        super().__init__(config)
        self.encoder = Wav2Vec2EncoderCondition(config)
        self.encoder.apply(self._init_weights)

    def forward(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            self.config.output_attentions
            if output_attentions is None
            else output_attentions
        )
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        features = self.feature_extractor(input_values).transpose(1, 2)
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                features.size(1), attention_mask, add_adapter=False
            )
        hidden_states, features = self.feature_projection(features)
        hidden_states = self._mask_hidden_states(
            hidden_states,
            mask_time_indices=mask_time_indices,
            attention_mask=attention_mask,
        )
        encoded = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_states = encoded.last_hidden_state
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)
        output = ConditionModelOutput(
            last_hidden_state=hidden_states,
            extract_features=features,
            hidden_states=encoded.hidden_states,
            attentions=encoded.attentions,
            intermediate_lang2vec_preds=encoded.intermediate_lang2vec_preds,
        )
        return output if return_dict else output.to_tuple()
