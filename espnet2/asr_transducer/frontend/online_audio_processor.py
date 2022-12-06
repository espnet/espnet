"""Online processor for Transducer models chunk-by-chunk streaming decoding."""

import math
from typing import Dict

import torch


class OnlineAudioProcessor:
    """OnlineProcessor module definition.

    Args:
        feature_extractor: Feature extractor module.
        normalization_module: Normalization module.
        decoding_window: Size of the decoding window (in ms).
        encoder_subsampling_factor: Encoder subsampling factor.
        encoder_minimum_feats: Encoder minimum features before subsampling
        frontend_conf: Fronted configuration.
        audio_sampling_rate: Input sampling rate.

    """

    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        normalization_module: torch.nn.Module,
        decoding_window: int,
        encoder_subsampling_factor: int,
        encoder_minimum_feats: int,
        frontend_conf: Dict,
        audio_sampling_rate: int = 16000,
    ) -> None:
        """Construct an OnlineAudioProcessor."""

        self.hop_sz = frontend_conf.get("hop_length", 128)
        self.decoding_window = int(decoding_window * audio_sampling_rate / 1000)

        self.chunk_sz_bs = self.decoding_window // self.hop_sz
        self.offset = encoder_minimum_feats + encoder_subsampling_factor

        self.feats_shift = self.chunk_sz_bs - self.offset
        self.minimum_feats_sz = encoder_minimum_feats + self.offset

        assert self.chunk_sz_bs >= encoder_minimum_feats, (
            "Specified decoding window length will yield %d feats. "
            "Minimum number of feats required is %d."
            % (self.chunk_sz_bs, encoder_minimum_feats)
        )

        self.feature_extractor = feature_extractor
        self.normalization_module = normalization_module

    def reset_cache(self, device: torch.device) -> None:
        """Reset cache parameters.

        Args:
            device: Device to pin samples_length attribute on.

        """
        self.samples = None
        self.samples_length = torch.tensor([0], device=device)
        self.decoding_step = 0

    def compute_features(self, new_samples: torch.Tensor, is_final: bool) -> None:
        """Add samples from audio source.

        Args:
            new_samples: Speech data. (S)
            is_final: Whether speech corresponds to the final chunk of data.

        Returns:
            feats: Features sequence. (1, T, D_feats)
            feats_length: Features length sequence. (1,)

        """
        new_samples = new_samples.unsqueeze(0)

        if self.samples is None:
            self.samples = new_samples
        else:
            self.samples = torch.cat((self.samples, new_samples), dim=1)
        self.samples_length += self.samples.size(1)

        feats, feats_length = self.feature_extractor(self.samples, self.samples_length)

        if self.normalization_module is not None:
            feats, feats_length = self.normalization_module(feats, feats_length)

        start_id = self.decoding_step * self.feats_shift
        self.decoding_step += 1

        if is_final:
            feats = feats[:, start_id:, :]
            feats_size = feats.size(1)

            if feats_size < self.minimum_feats_sz:
                feats = torch.nn.functional.pad(
                    feats,
                    (0, 0, 0, (self.minimum_feats_sz - feats_size), 0, 0),
                    mode="constant",
                    value=math.log(1e-10),
                )
        else:
            feats = feats[:, start_id : (start_id + self.chunk_sz_bs), :]

        feats_length = feats.new_full([1], dtype=torch.long, fill_value=feats.size(1))

        return feats, feats_length
