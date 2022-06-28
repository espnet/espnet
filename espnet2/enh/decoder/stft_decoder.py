from distutils.version import LooseVersion

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.layers.stft import Stft

is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")


class STFTDecoder(AbsDecoder):
    """STFT decoder for speech enhancement and separation"""

    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window="hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

    def forward(self, input: ComplexTensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (ComplexTensor): spectrum [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
        """
        if not isinstance(input, ComplexTensor) and (
            is_torch_1_9_plus and not torch.is_complex(input)
        ):
            raise TypeError("Only support complex tensors for stft decoder")

        wav, wav_lens = self.stft.inverse(input, ilens)

        return wav, wav_lens
