from typing import Tuple

import torch
from typeguard import typechecked


class Stft(torch.nn.Module):
    @typechecked
    def __init__(self,
                 n_fft: int = 512,
                 win_length: int = 512,
                 hop_length: int = 128,
                 center: bool = True,
                 pad_mode: str = 'reflect',
                 normalized: bool = False,
                 onesided: bool = True
                 ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided

    def forward(self, input: torch.Tensor, ilens: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        # output: (Batch, Freq, Frames, 2=real_imag)
        output = torch.stft(
            input, n_fft=self.n_fft, win_length=self.win_length,
            hop_length=self.hop_length, center=self.center,
            pad_mode=self.pad_mode, normalized=self.normalized,
            onesided=self.onesided)

        if self.center:
            pad = self.n_fft // 2
            ilens = ilens + 2 * pad
        olens = (ilens - self.win_length) // self.hop_length + 1

        return output, olens

    def istft(self, input):
        # TODO(kamo): torch audio?
        raise NotImplementedError
