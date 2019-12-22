from typing import Tuple
from typing import Union

import humanfriendly
import torch
from typeguard import check_argument_types

from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.utils.griffin_lim import Spectrogram2Waveform


class LogMelFbank(AbsFeatsExtract):
    """Conventional frontend structure for ASR

    Stft -> Power-spec -> Log-Mel-Fbank
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: Union[int, None] = 512,
        hop_length: int = 128,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
        htk: bool = False,
        norm=1
    ):
        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided
        )

        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
            norm=norm
        )

    def output_size(self) -> int:
        return self.n_mels

    def build_griffin_lim_vocoder(self, griffin_lim_iters) -> Spectrogram2Waveform:
        return Spectrogram2Waveform(
            fs=self.fs,
            griffin_lim_iters=griffin_lim_iters,
            n_fft=self.n_fft,
            n_shift=self.hop_length,
            n_mels=self.n_mels,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax,
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
        input_amp = torch.sqrt(input_power + 1.e-20)
        input_feats, _ = self.logmel(input_amp, feats_lens)
        return input_feats, feats_lens
