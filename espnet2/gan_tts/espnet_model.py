# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based text-to-speech ESPnet model."""

from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.gan_tts.abs_gan_tts import AbsGANTTS
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch < 1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


class ESPnetGANTTSModel(AbsGANESPnetModel):
    """ESPnet model for GAN-based text-to-speech task."""

    def __init__(
        self,
        feats_extract: Optional[AbsFeatsExtract],
        normalize: Optional[AbsNormalize and InversibleInterface],
        pitch_extract: Optional[AbsFeatsExtract],
        pitch_normalize: Optional[AbsNormalize and InversibleInterface],
        energy_extract: Optional[AbsFeatsExtract],
        energy_normalize: Optional[AbsNormalize and InversibleInterface],
        tts: AbsGANTTS,
    ):
        """Initialize ESPnetGANTTSModel module."""
        assert check_argument_types()
        super().__init__()
        self.feats_extract = feats_extract
        self.normalize = normalize
        self.pitch_extract = pitch_extract
        self.pitch_normalize = pitch_normalize
        self.energy_extract = energy_extract
        self.energy_normalize = energy_normalize
        self.tts = tts
        assert hasattr(
            tts, "generator"
        ), "generator module must be registered as tts.generator"
        assert hasattr(
            tts, "discriminator"
        ), "discriminator module must be registered as tts.discriminator"

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        durations_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        forward_generator: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Return generator or discriminator loss with dict format.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            duration (Optional[Tensor]): Duration tensor.
            duration_lengths (Optional[Tensor]): Duration length tensor (B,).
            pitch (Optional[Tensor]): Pitch tensor.
            pitch_lengths (Optional[Tensor]): Pitch length tensor (B,).
            energy (Optional[Tensor]): Energy tensor.
            energy_lengths (Optional[Tensor]): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker ID tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).
            forward_generator (bool): Whether to forward generator.
            kwargs: "utt_id" is among the input.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        with autocast(False):
            # Extract features
            feats = None
            if self.feats_extract is not None:
                feats, feats_lengths = self.feats_extract(speech, speech_lengths,)
            if self.pitch_extract is not None and pitch is None:
                pitch, pitch_lengths = self.pitch_extract(
                    speech,
                    speech_lengths,
                    feats_lengths=feats_lengths,
                    durations=durations,
                    durations_lengths=durations_lengths,
                )
            if self.energy_extract is not None and energy is None:
                energy, energy_lengths = self.energy_extract(
                    speech,
                    speech_lengths,
                    feats_lengths=feats_lengths,
                    durations=durations,
                    durations_lengths=durations_lengths,
                )

            # Normalize
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)
            if self.pitch_normalize is not None:
                pitch, pitch_lengths = self.pitch_normalize(pitch, pitch_lengths)
            if self.energy_normalize is not None:
                energy, energy_lengths = self.energy_normalize(energy, energy_lengths)

        # Make batch for tts inputs
        batch = dict(
            text=text, text_lengths=text_lengths, forward_generator=forward_generator,
        )

        # Update batch for additional auxiliary inputs
        if feats is not None:
            batch.update(feats=feats, feats_lengths=feats_lengths)
        if self.tts.require_raw_speech:
            batch.update(speech=speech, speech_lengths=speech_lengths)
        if durations is not None:
            batch.update(durations=durations, durations_lengths=durations_lengths)
        if self.pitch_extract is not None and pitch is not None:
            batch.update(pitch=pitch, pitch_lengths=pitch_lengths)
        if self.energy_extract is not None and energy is not None:
            batch.update(energy=energy, energy_lengths=energy_lengths)
        if spembs is not None:
            batch.update(spembs=spembs)
        if sids is not None:
            batch.update(sids=sids)
        if lids is not None:
            batch.update(lids=lids)

        return self.tts(**batch)

    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        durations_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Calculate features and return them as a dict.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B, 1).
            durations (Optional[Tensor): Duration tensor.
            durations_lengths (Optional[Tensor): Duration length tensor (B,).
            pitch (Optional[Tensor): Pitch tensor.
            pitch_lengths (Optional[Tensor): Pitch length tensor (B,).
            energy (Optional[Tensor): Energy tensor.
            energy_lengths (Optional[Tensor): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker index tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).

        Returns:
            Dict[str, Tensor]: Dict of features.

        """
        feats = None
        if self.feats_extract is not None:
            feats, feats_lengths = self.feats_extract(speech, speech_lengths,)
        if self.pitch_extract is not None:
            pitch, pitch_lengths = self.pitch_extract(
                speech,
                speech_lengths,
                feats_lengths=feats_lengths,
                durations=durations,
                durations_lengths=durations_lengths,
            )
        if self.energy_extract is not None:
            energy, energy_lengths = self.energy_extract(
                speech,
                speech_lengths,
                feats_lengths=feats_lengths,
                durations=durations,
                durations_lengths=durations_lengths,
            )

        # store in dict
        feats_dict = {}
        if feats is not None:
            feats_dict.update(feats=feats, feats_lengths=feats_lengths)
        if pitch is not None:
            feats_dict.update(pitch=pitch, pitch_lengths=pitch_lengths)
        if energy is not None:
            feats_dict.update(energy=energy, energy_lengths=energy_lengths)

        return feats_dict
