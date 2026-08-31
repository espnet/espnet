"""Neural-vocoder mel front-end as an ``AbsFeatsExtract``.

Wraps F5-TTS's ``MelSpec`` so :class:`~espnet3.systems.tts.f5_tts.f5tts.F5TTS`
can use it as its ``feats_extract``. Using F5's own mel, rather than
``LogMelFbank``, keeps training features bit-compatible with the neural vocoder
used at inference. ``mel_spec_type`` selects which vocoder family the mel
targets: ``"vocos"`` or ``"bigvgan"`` (hence the vocoder-agnostic name).

Output layout is ``[B, T, n_mels]`` (time-first), the layout the flow-matching
stack expects.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet3.systems.tts.f5_tts.modules import MelSpec


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
        """Configure the mel front end.

        Args:
            fs: Sampling rate of the input waveform.
            n_fft: FFT size.
            hop_length: STFT hop in samples.
            win_length: STFT window length in samples.
            n_mels: Number of mel bins, which becomes the model's mel
                dimension.
            mel_spec_type: Vocoder family this mel targets, ``"vocos"`` or
                ``"bigvgan"``.

        Example:
            .. code-block:: yaml

                feats_extract_conf:
                  fs: 24000
                  n_fft: 1024
                  hop_length: 256
                  win_length: 1024
                  n_mels: 100
                  mel_spec_type: vocos

        Note:
            The mel must match the vocoder used at inference, so these values
            are fixed by the vocoder rather than freely tunable. ``n_mels``
            becomes the model's mel dimension via :attr:`output_size`.
        """
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

    @property
    def output_size(self) -> int:
        """The mel dimension the model generates.

        Returns:
            ``n_mels``.

        Note:
            :class:`~espnet3.systems.tts.f5_tts.f5tts.F5TTS` reads this to size
            its backbone, which is why the recipe states the mel dimension only
            once, in ``feats_extract_conf``.
        """
        return self.n_mels

    def get_parameters(self) -> Dict[str, Any]:
        """Parameters a vocoder needs to reconstruct waveforms from these feats.

        Returns:
            Dict with ``fs``, ``n_fft``, ``n_shift``, ``win_length``, ``n_mels``
            and ``mel_spec_type``.

        Example:
            .. code-block:: python

                >>> sorted(VocoderMelSpec().get_parameters())
                ['fs', 'mel_spec_type', 'n_fft', 'n_mels', 'n_shift', 'win_length']

        Note:
            The hop is reported as ``n_shift``, espnet2's name for it, not as
            ``hop_length``. Downstream espnet2 vocoder tooling expects that key.
        """
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
        """Turn waveform ``[B, T_wav]`` into mel ``[B, T, n_mels]`` and lengths.

        ``feats_lengths`` uses the standard center=True STFT frame count
        ``T_wav // hop + 1``, the same formula espnet2's ``Stft`` uses
        (``(ilens + 2*(n_fft//2) - n_fft)//hop + 1``).

        Args:
            input: Waveform batch ``[B, T_wav]``.
            input_lengths: Valid sample count per utterance, ``[B]``. When
                omitted every utterance is treated as full length.

        Returns:
            Tuple of mel ``[B, T, n_mels]`` and frame counts ``[B]``.

        Example:
            .. code-block:: python

                >>> feats, lengths = VocoderMelSpec()(torch.randn(2, 24000))
                >>> feats.shape, lengths.tolist()
                (torch.Size([2, 94, 100]), [94, 94])

        Note:
            Output is time-first ``[B, T, n_mels]``, which is what the model
            consumes, not the channel-first layout F5's own ``MelSpec``
            returns.
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
