# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Default frontend module for TTS."""

from typing import Union

from typeguard import check_argument_types

from espnet2.asr.frontend.default import DefaultFrontend as ASRDefaultFrontend


class DefaultFrontend(ASRDefaultFrontend):
    """Conventional frontend structure for TTS.

    Speech -> STFT -> Power-spec -> Mel-Fbank

    """

    def __init__(
        self,
        fs: int,
        n_fft: int = 1024,
        n_shift: int = 256,
        win_length: Union[int, None] = None,
        n_mels: int = 80,
        fmin: Union[int, None] = 80,
        fmax: Union[int, None] = 7600,
    ):
        """Initialize frontend."""
        assert check_argument_types()
        super().__init__(
            fs=fs,
            n_fft=n_fft,
            stft_conf={
                "hop_length": n_shift,
                "win_length": win_length
            },
            frontend_conf={},
            logmel_fbank_conf={
                "n_mels": n_mels,
                "fmin": fmin,
                "fmax": fmax
            },
        )
