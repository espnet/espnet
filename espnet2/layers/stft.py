from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.layers.inversible_interface import InversibleInterface


class Stft(torch.nn.Module, InversibleInterface):
    def __init__(self,
                 n_fft: int = 512,
                 win_length: int = 512,
                 hop_length: int = 128,
                 center: bool = True,
                 pad_mode: str = 'reflect',
                 normalized: bool = False,
                 onesided: bool = True
                 ):
        assert check_argument_types()
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided

    def extra_repr(self):
        return (f'n_fft={self.n_fft}, '
                f'win_length={self.win_length}, '
                f'hop_length={self.hop_length}, '
                f'center={self.center}, '
                f'pad_mode={self.pad_mode}, '
                f'normalized={self.normalized}, '
                f'onesided={self.onesided}')

    def forward(self, input: torch.Tensor, ilens: torch.Tensor = None) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # output: (Batch, Freq, Frames, 2=real_imag)
        output = torch.stft(
            input, n_fft=self.n_fft, win_length=self.win_length,
            hop_length=self.hop_length, center=self.center,
            pad_mode=self.pad_mode, normalized=self.normalized,
            onesided=self.onesided)

        if ilens is not None:
            if self.center:
                pad = self.n_fft // 2
                ilens = ilens + 2 * pad

            olens = (ilens - self.win_length) // self.hop_length + 1
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None
        return output, olens

    def inverse(self, input: torch.Tensor, ilens: torch.Tensor = None) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # TODO(kamo): torch audio?
        raise NotImplementedError
