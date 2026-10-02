"""Neural-vocoder mel front-end as an espnet2 ``AbsFeatsExtract``.

Wraps F5-TTS's ``MelSpec`` so it can be selected as a ``feats_extract`` choice in
the (espnet2-compatible) TTS task. Using F5's own mel — rather than
``LogMelFbank`` — keeps training features bit-compatible with the neural vocoder
used at inference. ``mel_spec_type`` selects which vocoder family the mel targets:
``"vocos"`` or ``"bigvgan"`` (hence the vocoder-agnostic name).

Output layout is ``[B, T, n_mels]`` (time-first), matching what ``ESPnetTTSModel``
passes to the ``tts`` model.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from espnet2.tts.f5.modules import MelSpec
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract


class VocoderMelSpec(AbsFeatsExtract):
    """F5-TTS mel spectrogram (vocos/bigvgan target) as an AbsFeatsExtract."""

    def __init__(
        self,
        fs: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 100,
        mel_spec_type: str = "vocos",
    ):
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.mel_spec_type = mel_spec_type

        self.mel = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mels,
            target_sample_rate=fs,
            mel_spec_type=mel_spec_type,
        )

    def output_size(self) -> int:
        return self.n_mels

    def get_parameters(self) -> Dict[str, Any]:
        """Parameters a vocoder needs to reconstruct waveforms from these feats."""
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            n_shift=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            mel_spec_type=self.mel_spec_type,
        )

    def forward(
        self, input: torch.Tensor, input_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """input ``[B, T_wav]`` -> (feats ``[B, T, n_mels]``, feats_lengths ``[B]``).

        ``feats_lengths`` uses the standard center=True STFT frame count
        ``T_wav // hop + 1`` — the same formula espnet2's ``Stft`` uses
        (``(ilens + 2*(n_fft//2) - n_fft)//hop + 1``).
        """
        # MelSpec returns [B, n_mels, T]; F5's MelSpectrogram uses center=True.
        feats = self.mel(input).transpose(1, 2).contiguous()  # [B, T, n_mels]

        if input_lengths is None:
            feats_lengths = input.new_full(
                (feats.shape[0],), feats.shape[1], dtype=torch.long
            )
        else:
            feats_lengths = (
                input_lengths.div(self.hop_length, rounding_mode="floor") + 1
            ).clamp(max=feats.shape[1])
        return feats, feats_lengths
