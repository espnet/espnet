from typing import Any, Dict, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.layers.stft import Stft
from espnet2.s2st.tgt_feats_extract.abs_tgt_feats_extract import AbsTgtFeatsExtract


class LogSpectrogram(AbsTgtFeatsExtract):
    """Conventional frontend structure for ASR

    Stft -> log-amplitude-spec
    """

    @typechecked
    def __init__(
        self,
        n_fft: int = 1024,
        win_length: int = None,
        hop_length: int = 256,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
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
            normalized=normalized,
            onesided=onesided,
        )
        self.n_fft = n_fft

    def output_size(self) -> int:
        return self.n_fft // 2 + 1

    def get_parameters(self) -> Dict[str, Any]:
        """Return the parameters required by Vocoder"""
        return dict(
            n_fft=self.n_fft,
            n_shift=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Stft: time -> time-freq
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # NOTE(kamo): We use different definition for log-spec between TTS and ASR
        #   TTS: log_10(abs(stft))
        #   ASR: log_e(power(stft))

        # STFT -> Power spectrum
        # input_stft: (..., F, 2) -> (..., F)
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        log_amp = 0.5 * torch.log10(torch.clamp(input_power, min=1.0e-10))
        return log_amp, feats_lens

    def spectrogram(self) -> bool:
        return True
