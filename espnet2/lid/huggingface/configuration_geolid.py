# Copyright 2026 ESPnet contributors
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Portable configuration for the five released geolocation-aware LID models."""

from transformers import PretrainedConfig


class GeoLIDConfig(PretrainedConfig):
    model_type = "geolid"

    def __init__(
        self,
        wav2vec2_config=None,
        conditioning_layers=None,
        shared_conditioning_proj=True,
        lang2vec_dim=299,
        conditioning_embedding_dim=192,
        encoder_conf=None,
        embedding_dim=192,
        subcenters=3,
        scale=30.0,
        label_mapping_verified=False,
        legacy_inference=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wav2vec2_config = wav2vec2_config or {}
        self.conditioning_layers = conditioning_layers or []
        self.shared_conditioning_proj = shared_conditioning_proj
        self.lang2vec_dim = lang2vec_dim
        self.conditioning_embedding_dim = conditioning_embedding_dim
        self.encoder_conf = encoder_conf or {
            "model_scale": 8,
            "ndim": 512,
            "output_size": 1536,
        }
        self.embedding_dim = embedding_dim
        self.subcenters = subcenters
        self.scale = scale
        self.label_mapping_verified = label_mapping_verified
        self.legacy_inference = legacy_inference


GeoLIDConfig.register_for_auto_class()
