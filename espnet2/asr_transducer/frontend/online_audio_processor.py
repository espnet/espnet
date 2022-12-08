"""Online processor for Transducer models chunk-by-chunk streaming decoding."""

import math
from typing import Dict, Tuple

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

        self.n_fft = frontend_conf.get("n_fft", 512)
        self.hop_sz = frontend_conf.get("hop_length", 128)
        self.win_sz = frontend_conf.get("win_length", self.n_fft)

        self.decoding_window = int(decoding_window * audio_sampling_rate / 1000)

        self.chunk_sz_bs = self.decoding_window // self.hop_sz
        self.offset = 3 * encoder_subsampling_factor

        self.feats_shift = self.chunk_sz_bs - self.offset
        self.minimum_feats_sz = encoder_minimum_feats + self.offset

        assert self.chunk_sz_bs >= self.minimum_feats_sz, (
            "Specified decoding window length will yield %d feats. "
            "Minimum number of feats required is %d."
            % (self.chunk_sz_bs, self.minimum_feats_sz)
        )

        self.feature_extractor = feature_extractor
        self.normalization_module = normalization_module

    def reset_cache(self, device: torch.device) -> None:
        """Reset cache parameters.

        Args:
            device: Device to pin samples_length attribute on.

        """
        self.samples = None
        self.feats = None
        self.decoding_step = 0

        self.device = device

    def get_current_samples(
        self, samples: torch.Tensor, is_final: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get samples for feature computation.

        Args:
            samples: Speech data. (S)
            is_final: Whether speech corresponds to the final chunk of data.

        Returns:
            samples: New speech data. (1, decoding_window)
            samples_length: New speech length. (1,)

        """
        if self.samples is not None:
            new_samples = torch.cat([self.samples, samples], dim=0)
        else:
            new_samples = samples

        samples_sz = new_samples.size(0)

        if is_final:
            waveform_buffer = None

            if samples_sz < self.decoding_window:
                samples = torch.nn.functional.pad(
                    new_samples,
                    (0, self.decoding_window - samples_sz),
                    mode="constant",
                    value=0.0,
                )
            else:
                samples = new_samples
        else:
            n_frames = (samples_sz - (self.win_sz - self.hop_sz)) // self.hop_sz
            n_residual = (samples_sz - (self.win_sz - self.hop_sz)) % self.hop_sz

            samples = new_samples.narrow(
                0, 0, (self.win_sz - self.hop_sz) + n_frames * self.hop_sz
            )

            waveform_buffer = new_samples.narrow(
                0,
                samples_sz - (self.win_sz - self.hop_sz) - n_residual,
                (self.win_sz - self.hop_sz) + n_residual,
            ).clone()

        self.samples = waveform_buffer

        samples = samples.unsqueeze(0).to(device=self.device)

        lengths = samples.new_full([1], dtype=torch.long, fill_value=samples.size(1))

        return samples, lengths

    def get_current_feats(
        self, feats: torch.Tensor, is_final: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get features for current decoding window.

        Args:
            feats: Computed features sequence. (1, F, D_feats)
            feats_length: Computed features length sequence. (1,)

        Returns:
            feats: Decoding window features sequence. (1, chunk_sz_bs, D_feats)
            feats_length: Decoding window features length sequence. (1,)

        """
        if self.samples is None:
            feats = feats.narrow(
                1,
                0,
                feats.size(1) - math.ceil(math.ceil(self.win_sz / self.hop_sz) / 2),
            )
        else:
            feats = feats.narrow(
                1,
                math.ceil(math.ceil(self.win_sz / self.hop_sz) / 2),
                feats.size(1) - 2 * math.ceil(math.ceil(self.win_sz / self.hop_sz) / 2),
            )

        if self.feats is not None:
            feats = torch.cat((self.feats, feats), dim=1)

        self.feats = feats

        start_id = self.decoding_step * self.feats_shift
        self.decoding_step += 1

        if is_final:
            feats = feats[:, start_id:, :]
        else:
            feats = feats[:, start_id : (start_id + self.chunk_sz_bs), :]

        feats_length = feats.new_full([1], dtype=torch.long, fill_value=feats.size(1))

        return feats, feats_length

    def compute_features(self, samples: torch.Tensor, is_final: bool) -> None:
        """Compute features from input samples.

        Args:
            samples: Speech data. (S)
            is_final: Whether speech corresponds to the final chunk of data.

        Returns:
            feats: Features sequence. (1, chunk_sz_bs, D_feats)
            feats_length: Features length sequence. (1,)

        """
        samples, samples_length = self.get_current_samples(samples, is_final)

        feats, feats_length = self.feature_extractor(samples, samples_length)

        if self.normalization_module is not None:
            feats, _ = self.normalization_module(feats, feats_length)

        feats, feats_length = self.get_current_feats(feats, is_final)

        return feats, feats_length
