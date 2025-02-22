# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Text-to-speech ESPnet model."""

from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.tts2.abs_tts2 import AbsTTS2
from espnet2.tts2.feats_extract.abs_feats_extract import AbsFeatsExtractDiscrete
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


class ESPnetTTS2Model(AbsESPnetModel):
    """ESPnet model for text-to-speech task."""

    @typechecked
    def __init__(
        self,
        discrete_feats_extract: AbsFeatsExtractDiscrete,
        pitch_extract: Optional[AbsFeatsExtract],
        energy_extract: Optional[AbsFeatsExtract],
        pitch_normalize: Optional[AbsNormalize and InversibleInterface],
        energy_normalize: Optional[AbsNormalize and InversibleInterface],
        tts: AbsTTS2,
    ):
        """Initialize ESPnetTTSModel module."""
        super().__init__()
        self.discrete_feats_extract = discrete_feats_extract
        self.pitch_extract = pitch_extract
        self.energy_extract = energy_extract
        self.pitch_normalize = pitch_normalize
        self.energy_normalize = energy_normalize
        self.tts = tts

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        discrete_speech: torch.Tensor,
        discrete_speech_lengths: torch.Tensor,
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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Caclualte outputs and return the loss tensor.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            discrete_speech (Tensor): Discrete speech tensor (B, T_token).
            discrete_speech_lengths (Tensor): Discrete speech length tensor (B,).
            duration (Optional[Tensor]): Duration tensor.
            duration_lengths (Optional[Tensor]): Duration length tensor (B,).
            pitch (Optional[Tensor]): Pitch tensor.
            pitch_lengths (Optional[Tensor]): Pitch length tensor (B,).
            energy (Optional[Tensor]): Energy tensor.
            energy_lengths (Optional[Tensor]): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker ID tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).
            kwargs: "utt_id" is among the input.

        Returns:
            Tensor: Loss scalar tensor.
            Dict[str, float]: Statistics to be monitored.
            Tensor: Weight tensor to summarize losses.

        """
        with autocast(False):
            # Extract features
            discrete_feats, discrete_feats_lengths = self.discrete_feats_extract(
                discrete_speech, discrete_speech_lengths
            )
            feats, feats_lengths = speech, speech_lengths  # noqa

            # Extract auxiliary features
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
            if self.pitch_normalize is not None:
                pitch, pitch_lengths = self.pitch_normalize(pitch, pitch_lengths)
            if self.energy_normalize is not None:
                energy, energy_lengths = self.energy_normalize(energy, energy_lengths)

        # Make batch for tts inputs
        batch = dict(
            text=text,
            text_lengths=text_lengths,
            discrete_feats=discrete_feats,
            discrete_feats_lengths=discrete_feats_lengths,
        )

        # Update batch for additional auxiliary inputs
        if spembs is not None:
            batch.update(spembs=spembs)
        if sids is not None:
            batch.update(sids=sids)
        if lids is not None:
            batch.update(lids=lids)
        if durations is not None:
            batch.update(durations=durations, durations_lengths=durations_lengths)
        if self.pitch_extract is not None and pitch is not None:
            batch.update(pitch=pitch, pitch_lengths=pitch_lengths)
        if self.energy_extract is not None and energy is not None:
            batch.update(energy=energy, energy_lengths=energy_lengths)
        if self.tts.require_raw_speech:
            batch.update(speech=speech, speech_lengths=speech_lengths)

        return self.tts(**batch)

    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        discrete_speech: torch.Tensor,
        discrete_speech_lengths: torch.Tensor,
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
        """Caclualte features and return them as a dict.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            discrete_speech (Tensor): Discrete speech tensor (B, T_token).
            discrete_speech_lengths (Tensor): Discrete speech length tensor (B,).
            durations (Optional[Tensor): Duration tensor.
            durations_lengths (Optional[Tensor): Duration length tensor (B,).
            pitch (Optional[Tensor): Pitch tensor.
            pitch_lengths (Optional[Tensor): Pitch length tensor (B,).
            energy (Optional[Tensor): Energy tensor.
            energy_lengths (Optional[Tensor): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker ID tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).

        Returns:
            Dict[str, Tensor]: Dict of features.

        """
        # feature extraction
        discrete_feats, discrete_feats_lengths = self.discrete_feats_extract(
            discrete_speech, discrete_speech_lengths
        )
        feats, feats_lengths = speech, speech_lengths  # noqa
        if self.pitch_extract is not None:
            pitch, pitch_lengths = self.pitch_extract(
                speech,
                speech_lengths,
                feats_lengths=discrete_feats_lengths,  # noqa
                durations=durations,
                durations_lengths=durations_lengths,
            )
        if self.energy_extract is not None:
            energy, energy_lengths = self.energy_extract(
                speech,
                speech_lengths,
                feats_lengths=discrete_feats_lengths,  # noqa
                durations=durations,
                durations_lengths=durations_lengths,
            )

        # store in dict
        feats_dict = dict(
            discrete_feats=discrete_feats,  # noqa
            discrete_feats_lengths=discrete_feats_lengths,  # noqa
        )
        if pitch is not None:
            feats_dict.update(pitch=pitch, pitch_lengths=pitch_lengths)
        if energy is not None:
            feats_dict.update(energy=energy, energy_lengths=energy_lengths)

        return feats_dict

    def inference(
        self,
        text: torch.Tensor,
        speech: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        **decode_config,
    ) -> Dict[str, torch.Tensor]:
        """Caclualte features and return them as a dict.

        Args:
            text (Tensor): Text index tensor (T_text).
            speech (Tensor): Speech waveform tensor (T_wav).
            spembs (Optional[Tensor]): Speaker embedding tensor (D,).
            sids (Optional[Tensor]): Speaker ID tensor (1,).
            lids (Optional[Tensor]): Language ID tensor (1,).
            durations (Optional[Tensor): Duration tensor.
            pitch (Optional[Tensor): Pitch tensor.
            energy (Optional[Tensor): Energy tensor.

        Returns:
            Dict[str, Tensor]: Dict of outputs.

        """
        input_dict = dict(text=text)
        if decode_config["use_teacher_forcing"] or getattr(self.tts, "use_gst", False):
            if speech is None:
                raise RuntimeError("missing required argument: 'speech'")
            feats = speech
            input_dict.update(feats=feats)
            if self.tts.require_raw_speech:
                input_dict.update(speech=speech)

        if decode_config["use_teacher_forcing"]:
            if durations is not None:
                input_dict.update(durations=durations)

            if self.pitch_extract is not None:
                pitch = self.pitch_extract(
                    speech[None],
                    feats_lengths=torch.LongTensor([len(feats)]),
                    durations=durations[None],
                )[0][0]
            if self.pitch_normalize is not None:
                pitch = self.pitch_normalize(pitch[None])[0][0]
            if pitch is not None:
                input_dict.update(pitch=pitch)

            if self.energy_extract is not None:
                energy = self.energy_extract(
                    speech[None],
                    feats_lengths=torch.LongTensor([len(feats)]),
                    durations=durations[None],
                )[0][0]
            if self.energy_normalize is not None:
                energy = self.energy_normalize(energy[None])[0][0]
            if energy is not None:
                input_dict.update(energy=energy)

        if spembs is not None:
            input_dict.update(spembs=spembs)
        if sids is not None:
            input_dict.update(sids=sids)
        if lids is not None:
            input_dict.update(lids=lids)

        output_dict = self.tts.inference(**input_dict, **decode_config)

        # Predict the discrete tokens. Currently only apply argmax for selction
        output_dict["feat_gen"] = output_dict["feat_gen"].argmax(1)

        return output_dict
