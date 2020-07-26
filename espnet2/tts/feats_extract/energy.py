# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Energy extractor."""

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import humanfriendly
import torch
import torch.nn.functional as F

from typeguard import check_argument_types

from espnet2.layers.stft import Stft
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract


class Energy(AbsFeatsExtract):
    """Energy extractor."""

    def __init__(
        self,
        fs: Union[int, str] = 22050,
        n_fft: int = 1024,
        win_length: int = None,
        hop_length: int = 256,
        window: Optional[str] = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: bool = True,
        use_token_averaged_energy: bool = True,
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

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
        )

    def output_size(self) -> int:
        return 1

    def get_parameters(self) -> Dict[str, Any]:
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            n_shift=self.hop_length,
            window=self.window,
            win_length=self.win_length,
        )

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
        durations: torch.Tensor = None,
        durations_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        input_stft, energy_lengths = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # input_stft: (..., F, 2) -> (..., F)
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        energy = torch.sqrt(torch.clamp(input_power.sum(dim=1), min=1.0e-10))
        energy = energy.unsqueeze(-1)

        # (Optional): Average by duration
        if self.use_token_averaged_energy:
            energy = [
                self._average_by_duration(e, d) for e, d in zip(energy, durations)
            ]
            energy_lengths = durations_lengths
        return energy, energy_lengths

    @staticmethod
    def _average_by_duration(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        assert d.sum() == len(x)
        d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
        x_avg = [
            x[start:end].masked_select(x[start:end].ne(0.0)).mean(dim=0)
            if len(x[start:end].masked_select(x[start:end].ne(0.0))) != 0
            else x.new_tensor(0.0)
            for start, end in zip(d_cumsum[:-1], d_cumsum[1:])
        ]
        return torch.stack(x_avg).unsqueeze(-1)
