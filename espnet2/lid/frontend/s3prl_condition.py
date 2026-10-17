# Copyright 2025 Qingzheng Wang, 2026 ESPnet contributors
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""GeoLID frontend using stock Transformers, with legacy checkpoint key names."""

import math
from typing import List, Optional, Union

import humanfriendly
import torch
import torch.nn.functional as F
from torch import nn
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend


class S3prlFrontendCondition(AbsFrontend):
    """Conditioned Wav2Vec2 frontend without an S3PRL dependency.

    The historical class/configuration name and parameter hierarchy are retained
    so that the five published GeoLID ESPnet checkpoints load without conversion.
    ``forward`` implements AbsFrontend; ``forward_with_condition`` additionally
    returns the intermediate predictions used by the GeoLID training objective.
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        frontend_conf: Optional[dict] = None,
        download_dir: Optional[str] = None,
        multilayer_feature: bool = False,
        layer: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__()
        from espnet2.lid.huggingface.modeling_wav2vec2_condition import (
            Wav2Vec2ModelCondition,
        )

        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        if fs != 16000:
            raise ValueError("GeoLID expects 16 kHz audio.")
        conf = dict(frontend_conf or {})
        if conf.get("upstream", "hf_wav2vec2_condition") != "hf_wav2vec2_condition":
            raise ValueError("Only hf_wav2vec2_condition is supported.")
        model = Wav2Vec2ModelCondition.from_pretrained(
            conf.get("path_or_url", "facebook/mms-1b"),
            cache_dir=download_dir,
            **(conf.get("extra_conf") or {}),
        )
        model.eval()
        # Preserve frontend.upstream.upstream.model.* in the original state dicts.
        self.upstream = nn.Module()
        self.upstream.upstream = nn.Module()
        self.upstream.upstream.model = model
        self.featurizer = nn.Module()
        n_layers = model.config.num_hidden_layers + 1
        if isinstance(layer, int):
            selections = [layer]
            if multilayer_feature:
                raise ValueError("An integer layer disables multilayer_feature.")
        elif layer is not None:
            selections = sorted(layer)
            if not multilayer_feature:
                raise ValueError("A list of layers requires multilayer_feature=True.")
        else:
            selections = list(range(n_layers))
        if (
            not selections
            or len(set(selections)) != len(selections)
            or any(i < 0 or i >= n_layers for i in selections)
        ):
            raise ValueError(f"layer must select unique indices in [0, {n_layers}).")
        self.featurizer.weights = nn.Parameter(torch.zeros(len(selections)))
        self.layer_selections = selections
        self.multilayer_feature = multilayer_feature
        self.layer = layer
        self.normalize = conf.get("normalize", False)
        self.tile_factor = conf.get("tile_factor", 1)
        if not isinstance(self.tile_factor, int) or self.tile_factor < 1:
            raise ValueError("tile_factor must be a positive integer.")
        self.hop_length = math.prod(model.config.conv_stride)
        self.frontend_type = "s3prl"

    def output_size(self) -> int:
        return self.upstream.upstream.model.config.hidden_size

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor):
        feats, lengths, _ = self.forward_with_condition(input, input_lengths)
        return feats, lengths

    def forward_with_condition(self, input, input_lengths):
        """Return features, frame lengths, and intermediate lang2vec predictions."""
        if input.ndim == 3:
            input = input.squeeze(-1)
        lengths = input_lengths.to(device=input.device, dtype=torch.long)
        if (lengths <= 0).any() or (lengths > input.size(1)).any():
            raise ValueError("Audio lengths must be positive and within the input.")
        original_lengths = lengths
        max_length = int(lengths.max())
        input = input[:, :max_length]
        # Match S3PRL's short-waveform handling before feature-extractor normalization.
        if max_length < 800:
            input = F.pad(input, (0, 800 - max_length))
            lengths = lengths + (800 - max_length)
        mask = torch.arange(input.size(1), device=input.device)[None] < lengths[:, None]
        values = input.masked_fill(~mask, 0)
        mean = values.sum(-1, keepdim=True) / lengths[:, None]
        values = (values - mean).masked_fill(~mask, 0)
        var = values.square().sum(-1, keepdim=True) / lengths[:, None]
        values = values / torch.sqrt(var + 1e-7)
        output = self.upstream.upstream.model(
            values, attention_mask=mask, output_hidden_states=True, return_dict=True
        )
        # S3PRL used ceil(samples / stride), repeating the final frame when needed.
        feat_lengths = (
            torch.div(original_lengths - 1, self.hop_length, rounding_mode="floor") + 1
        )
        target = int(feat_lengths.max())
        states = output.hidden_states
        selected = (
            self.layer_selections
            if self.multilayer_feature
            else [self.layer if isinstance(self.layer, int) else len(states) - 1]
        )
        features = []
        for i in selected:
            h = states[i]
            if h.size(1) < target:
                h = torch.cat((h, h[:, -1:].expand(-1, target - h.size(1), -1)), 1)
            h = h[:, :target]
            if self.normalize:
                h = F.layer_norm(h, h.shape[-1:])
            features.append(h)
        if len(features) == 1:
            feats = features[0]
        else:
            weights = self.featurizer.weights.softmax(0)
            feats = (torch.stack(features) * weights[:, None, None, None]).sum(0)
        if self.tile_factor != 1:
            feats = feats.repeat_interleave(self.tile_factor, dim=1)
            feat_lengths = feat_lengths * self.tile_factor
        return feats, feat_lengths, output.intermediate_lang2vec_preds
