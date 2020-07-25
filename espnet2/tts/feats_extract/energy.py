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
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # input_stft: (..., F, 2) -> (..., F)
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        energy = torch.sqrt(torch.clamp(input_power.sum(dim=1), min=1.0e-10))
        return energy.unsqueeze(-1), feats_lens
