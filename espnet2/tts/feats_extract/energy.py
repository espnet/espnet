# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Energy extractor."""

from typing import Any, Dict, Tuple, Union

import humanfriendly
import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.layers.stft import Stft
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet.nets.pytorch_backend.nets_utils import pad_list


class Energy(AbsFeatsExtract):
    """Energy extractor."""

    def __init__(
        self,
        fs: Union[int, str] = 22050,
        n_fft: int = 1024,
        win_length: int = None,
        hop_length: int = 256,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        use_token_averaged_energy: bool = True,
        reduction_factor: int = None,
    ):
        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.use_token_averaged_energy = use_token_averaged_energy
        if use_token_averaged_energy:
            assert reduction_factor >= 1
        self.reduction_factor = reduction_factor

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

    def output_size(self) -> int:
        return 1

    def get_parameters(self) -> Dict[str, Any]:
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            win_length=self.win_length,
            center=self.stft.center,
            normalized=self.stft.normalized,
            use_token_averaged_energy=self.use_token_averaged_energy,
            reduction_factor=self.reduction_factor,
        )

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor = None,
        feats_lengths: torch.Tensor = None,
        durations: torch.Tensor = None,
        durations_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # If not provide, we assume that the inputs have the same length
        if input_lengths is None:
            input_lengths = (
                input.new_ones(input.shape[0], dtype=torch.long) * input.shape[1]
            )

        # Domain-conversion: e.g. Stft: time -> time-freq
        input_stft, energy_lengths = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        assert input_stft.shape[-1] == 2, input_stft.shape

        # input_stft: (..., F, 2) -> (..., F)
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        # sum over frequency (B, N, F) -> (B, N)
        energy = torch.sqrt(torch.clamp(input_power.sum(dim=2), min=1.0e-10))

        # (Optional): Adjust length to match with the mel-spectrogram
        if feats_lengths is not None:
            energy = [
                self._adjust_num_frames(e[:el].view(-1), fl)
                for e, el, fl in zip(energy, energy_lengths, feats_lengths)
            ]
            energy_lengths = feats_lengths

        # (Optional): Average by duration to calculate token-wise energy
        if self.use_token_averaged_energy:
            durations = durations * self.reduction_factor
            energy = [
                self._average_by_duration(e[:el].view(-1), d)
                for e, el, d in zip(energy, energy_lengths, durations)
            ]
            energy_lengths = durations_lengths

        # Padding
        if isinstance(energy, list):
            energy = pad_list(energy, 0.0)

        # Return with the shape (B, T, 1)
        return energy.unsqueeze(-1), energy_lengths

    def _average_by_duration(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        assert 0 <= len(x) - d.sum() < self.reduction_factor
        d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
        x_avg = [
            x[start:end].mean() if len(x[start:end]) != 0 else x.new_tensor(0.0)
            for start, end in zip(d_cumsum[:-1], d_cumsum[1:])
        ]
        return torch.stack(x_avg)

    @staticmethod
    def _adjust_num_frames(x: torch.Tensor, num_frames: torch.Tensor) -> torch.Tensor:
        if num_frames > len(x):
            x = F.pad(x, (0, num_frames - len(x)))
        elif num_frames < len(x):
            x = x[:num_frames]
        return x
