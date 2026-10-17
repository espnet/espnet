# Copyright 2025 Qingzheng Wang, 2026 ESPnet contributors
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Standalone inference for ESPnet's released GeoLID checkpoints.

Inputs are 16 kHz waveforms normalized by Wav2Vec2FeatureExtractor. Scores are
scaled subcenter cosine similarities; their argmax matches ESPnet's LID output.
The original training objective is implemented by ESPnet, not this inference API.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel, Wav2Vec2Config
from transformers.utils import ModelOutput

from .configuration_geolid import GeoLIDConfig
from .modeling_layers import (
    ChnAttnStatPooling,
    EcapaTdnnEncoder,
    IdentityEncoder,
    LegacyChnAttnStatPooling,
    RawNet3Projector,
    legacy_mean_normalize,
)
from .modeling_wav2vec2_condition import Wav2Vec2ModelCondition


@dataclass
class GeoLIDOutput(ModelOutput):
    logits: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None
    intermediate_lang2vec_preds: Optional[Tuple[torch.Tensor, ...]] = None


class GeoLIDForAudioClassification(PreTrainedModel):
    config_class = GeoLIDConfig
    base_model_prefix = "wav2vec2"
    main_input_name = "input_values"

    def __init__(self, config):
        super().__init__(config)
        wav_config = Wav2Vec2Config(**config.wav2vec2_config)
        # Eager attention works across the original 4.x and current 5.x releases.
        wav_config._attn_implementation = "eager"
        self.wav2vec2 = Wav2Vec2ModelCondition(wav_config)
        hidden = wav_config.hidden_size
        enc = self.wav2vec2.encoder
        enc.lang2vec_conditioning_layers = config.conditioning_layers
        enc.shared_conditioning_proj = config.shared_conditioning_proj
        pooling_class = (
            LegacyChnAttnStatPooling if config.legacy_inference else ChnAttnStatPooling
        )
        enc.ecapa_encoder = nn.ModuleDict()
        enc.pooling = nn.ModuleDict()
        enc.projector = nn.ModuleDict()
        enc.lang2vec_head = nn.ModuleDict()
        if config.shared_conditioning_proj:
            enc.lang2vec_conditioning_projs = nn.Linear(config.lang2vec_dim, hidden)
        else:
            enc.lang2vec_conditioning_projs = nn.ModuleDict()
        for i in config.conditioning_layers:
            key = str(i)
            enc.ecapa_encoder[key] = IdentityEncoder(hidden)
            enc.pooling[key] = pooling_class(hidden)
            enc.projector[key] = RawNet3Projector(
                hidden * 2, config.conditioning_embedding_dim
            )
            enc.lang2vec_head[key] = nn.Sequential(
                nn.Linear(config.conditioning_embedding_dim, config.lang2vec_dim)
            )
            if not config.shared_conditioning_proj:
                enc.lang2vec_conditioning_projs[key] = nn.Sequential(
                    nn.Linear(config.lang2vec_dim, hidden)
                )
        self.layer_weights = nn.Parameter(torch.zeros(wav_config.num_hidden_layers + 1))
        self.encoder = EcapaTdnnEncoder(hidden, **config.encoder_conf)
        self.pooling = pooling_class(self.encoder.output_size())
        self.projector = RawNet3Projector(
            self.pooling.output_size(), config.embedding_dim
        )
        # Preserve the trained classifier and auxiliary head, without loss aliases.
        self.loss = nn.Module()
        self.loss.weight = nn.Parameter(
            torch.empty(config.num_labels * config.subcenters, config.embedding_dim)
        )
        self.loss.lang2vec_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.lang2vec_dim)
        )
        nn.init.xavier_uniform_(self.loss.weight)
        self.post_init()

    def forward(self, input_values, attention_mask=None, return_dict=None):
        """Classify normalized waveforms, padded on the right.

        An attention mask is required for batches containing different durations.
        Clips must contain at least 800 samples (50 ms), as in the recipe frontend.
        ``logits.argmax(-1)`` returns the original checkpoint's category indices.
        """
        return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_values, dtype=torch.long)
        lengths = attention_mask.sum(-1)
        if (lengths < 800).any():
            raise ValueError("GeoLID inference requires at least 800 samples per clip.")
        output = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        stride = math.prod(self.wav2vec2.config.conv_stride)
        lengths = torch.div(lengths - 1, stride, rounding_mode="floor") + 1
        target = int(lengths.max())
        states = []
        for h in output.hidden_states:
            if h.size(1) < target:
                h = torch.cat((h, h[:, -1:].expand(-1, target - h.size(1), -1)), 1)
            states.append(h[:, :target])
        weights = self.layer_weights.softmax(0)
        features = (torch.stack(states) * weights[:, None, None, None]).sum(0)
        # UtteranceMVN(norm_means=True, norm_vars=False).
        if self.config.legacy_inference:
            features = legacy_mean_normalize(features, lengths)
        else:
            mask = torch.arange(target, device=features.device)[None] < lengths[:, None]
            features = features.masked_fill(~mask[:, :, None], 0)
            mean = features.sum(1, keepdim=True) / lengths[:, None, None]
            features = (features - mean).masked_fill(~mask[:, :, None], 0)
        embeddings = self.projector(
            self.pooling(self.encoder(features), feat_lengths=lengths)
        )
        cosine = nn.functional.linear(
            nn.functional.normalize(embeddings),
            nn.functional.normalize(self.loss.weight),
        )
        logits = (
            cosine.view(-1, self.config.num_labels, self.config.subcenters)
            .max(-1)
            .values
        )
        result = GeoLIDOutput(
            logits=logits * self.config.scale,
            embeddings=embeddings,
            intermediate_lang2vec_preds=output.intermediate_lang2vec_preds,
        )
        return result if return_dict else result.to_tuple()


GeoLIDForAudioClassification.register_for_auto_class("AutoModelForAudioClassification")
