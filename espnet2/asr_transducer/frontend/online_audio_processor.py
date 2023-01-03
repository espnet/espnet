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
        encoder_sub_factor: Encoder subsampling factor.
        frontend_conf: Frontend configuration.
        device: Device to pin module tensors on.
        audio_sampling_rate: Input sampling rate.

    """

    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        normalization_module: torch.nn.Module,
        decoding_window: int,
        encoder_sub_factor: int,
        frontend_conf: Dict,
        device: torch.device,
        audio_sampling_rate: int = 16000,
    ) -> None:
        """Construct an OnlineAudioProcessor."""

        self.n_fft = frontend_conf.get("n_fft", 512)
        self.hop_sz = frontend_conf.get("hop_length", 128)
        self.win_sz = frontend_conf.get("win_sz", self.n_fft)

        self.win_hop_sz = self.win_sz - self.hop_sz
        self.trim_val = math.ceil(math.ceil(self.win_sz / self.hop_sz) / 2)

        self.decoding_window = int(decoding_window * audio_sampling_rate / 1000)
        self.offset = lambda x: 2 * encoder_sub_factor + x % encoder_sub_factor

        self.feature_extractor = feature_extractor
        self.normalization_module = normalization_module

        self.device = device

        self.reset_cache()

    def reset_cache(self) -> None:
        """Reset cache parameters.

        Args:
            None

        Returns:
            None

        """
        self.samples = None
        self.feats = None

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
            n_frames = (samples_sz - self.win_hop_sz) // self.hop_sz
            n_residual = (samples_sz - self.win_hop_sz) % self.hop_sz

            samples = new_samples.narrow(0, 0, self.win_hop_sz + n_frames * self.hop_sz)

            waveform_buffer = new_samples.narrow(
                0,
                samples_sz - self.win_hop_sz - n_residual,
                self.win_hop_sz + n_residual,
            ).clone()

        self.samples = waveform_buffer

        samples = samples.unsqueeze(0).to(device=self.device)
        lengths = samples.new_full([1], dtype=torch.long, fill_value=samples.size(1))

        return samples, lengths

    def get_current_feats(
        self, feats: torch.Tensor, feats_length: torch.Tensor, is_final: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get features for current decoding window.

        Args:
            feats: Computed features sequence. (1, F, D_feats)
            feats_length: Computed features sequence length. (1,)
            is_final: Whether feats corresponds to the final chunk of data.

        Returns:
            feats: Decoding window features sequence. (1, chunk_sz_bs, D_feats)
            feats_length: Decoding window features length sequence. (1,)

        """
        if is_final:
            _feats = feats.narrow(1, self.trim_val, feats.size(1) - self.trim_val)
        else:
            if self.feats is None:
                _feats = feats.narrow(1, 0, feats.size(1) - self.trim_val)
            else:
                _feats = feats.narrow(
                    1, self.trim_val, feats.size(1) - 2 * self.trim_val
                )

        if self.feats is not None:
            feats = torch.cat((self.feats, _feats), dim=1)
        else:
            feats = torch.nn.functional.pad(
                _feats,
                (0, 0, self.offset(feats.size(1)), 0, 0, 0),
                mode="constant",
                value=0.0,
            )

        self.feats = feats[:, -self.offset(feats.size(1)) :, :]

        feats_length.fill_(feats.size(1))

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
            feats, feats_length = self.normalization_module(feats, feats_length)

        feats, feats_length = self.get_current_feats(feats, feats_length, is_final)

        return feats, feats_length
