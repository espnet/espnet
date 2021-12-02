# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Voice Conversion ESPnet model."""

from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import Optional
from typing import Tuple

import torch

from typeguard import check_argument_types

from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.vc.abs_vc import AbsVC
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


class ESPnetVCModel(AbsESPnetModel):
    """ESPnet model for voice conversion task."""

    def __init__(
        self,
        feats_extract: Optional[AbsFeatsExtract],
        input_feats_extract: Optional[AbsFeatsExtract],
        pitch_extract: Optional[AbsFeatsExtract],
        energy_extract: Optional[AbsFeatsExtract],
        normalize: Optional[AbsNormalize and InversibleInterface],
        input_normalize: Optional[AbsNormalize and InversibleInterface],
        pitch_normalize: Optional[AbsNormalize and InversibleInterface],
        energy_normalize: Optional[AbsNormalize and InversibleInterface],
        vc: AbsVC,
    ):
        """Initialize ESPnetVCModel module."""
        assert check_argument_types()
        super().__init__()
        self.feats_extract = feats_extract
        self.input_feats_extract = input_feats_extract
        self.pitch_extract = pitch_extract
        self.energy_extract = energy_extract
        self.normalize = normalize
        self.input_normalize = input_normalize
        self.pitch_normalize = pitch_normalize
        self.energy_normalize = energy_normalize
        self.vc = vc

    def forward(
        self,
        in_speech: torch.Tensor,
        in_speech_lengths: torch.Tensor,
        out_speech: torch.Tensor,
        out_speech_lengths: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        durations_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate outputs and return the loss tensor.

        Args:
            in_speech (Tensor): Speech waveform tensor (B, T_wav).
            in_speech_lengths (Tensor): Speech length tensor (B,).
            out_speech (Tensor): Speech waveform tensor (B, T_wav).
            out_speech_lengths (Tensor): Speech length tensor (B,).
            duration (Optional[Tensor]): Duration tensor.
            duration_lengths (Optional[Tensor]): Duration length tensor (B,).
            pitch (Optional[Tensor]): Pitch tensor.
            pitch_lengths (Optional[Tensor]): Pitch length tensor (B,).
            energy (Optional[Tensor]): Energy tensor.
            energy_lengths (Optional[Tensor]): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker ID tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).

        Returns:
            Tensor: Loss scalar tensor.
            Dict[str, float]: Statistics to be monitored.
            Tensor: Weight tensor to summarize losses.

        """
        with autocast(False):
            # Extract output features
            if self.feats_extract is not None:
                feats, feats_lengths = self.feats_extract(out_speech, out_speech_lengths)
            else:
                # Use precalculated feats (feats_type != raw case)
                feats, feats_lengths = out_speech, out_speech_lengths
            # Extract input features
            if self.input_feats_extract is not None:
                in_feats, in_feats_lengths = self.input_feats_extract(in_speech, in_speech_lengths)
            else:
                # Use precalculated feats (feats_type != raw case)
                in_feats, in_feats_lengths = in_speech, in_speech_lengths

            # Extract auxiliary features
            if self.pitch_extract is not None and pitch is None:
                pitch, pitch_lengths = self.pitch_extract(
                    out_speech,
                    out_speech_lengths,
                    feats_lengths=feats_lengths,
                    durations=durations,
                    durations_lengths=durations_lengths,
                )
            if self.energy_extract is not None and energy is None:
                energy, energy_lengths = self.energy_extract(
                    out_speech,
                    out_speech_lengths,
                    feats_lengths=feats_lengths,
                    durations=durations,
                    durations_lengths=durations_lengths,
                )

            # Normalize
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)
            if self.input_normalize is not None:
                in_feats, in_feats_lengths = self.input_normalize(in_feats, in_feats_lengths)
            if self.pitch_normalize is not None:
                pitch, pitch_lengths = self.pitch_normalize(pitch, pitch_lengths)
            if self.energy_normalize is not None:
                energy, energy_lengths = self.energy_normalize(energy, energy_lengths)

        # Make batch for vc inputs
        batch = dict(
            xs=in_feats,
            ilens=in_feats_lengths,
            ys=feats,
            olens=feats_lengths,
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
        if self.vc.require_raw_speech:
            batch.update(speech=out_speech, speech_lengths=out_speech_lengths)

        return self.vc(**batch)

    def collect_feats(
        self,
        in_speech: torch.Tensor,
        in_speech_lengths: torch.Tensor,
        out_speech: torch.Tensor,
        out_speech_lengths: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        durations_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Calculate features and return them as a dict.

        Args:
            in_speech (Tensor): Speech waveform tensor (B, T_wav).
            in_speech_lengths (Tensor): Speech length tensor (B,).
            out_speech (Tensor): Speech waveform tensor (B, T_wav).
            out_speech_lengths (Tensor): Speech length tensor (B,).
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
        # output feature extraction
        if self.feats_extract is not None:
            feats, feats_lengths = self.feats_extract(out_speech, out_speech_lengths)
        else:
            # Use precalculated feats (feats_type != raw case)
            feats, feats_lengths = out_speech, out_speech_lengths
        # input feature extraction
        if self.input_feats_extract is not None:
            in_feats, in_feats_lengths = self.input_feats_extract(in_speech, in_speech_lengths)
        else:
            # Use precalculated feats (feats_type != raw case)
            in_feats, in_feats_lengths = in_speech, in_speech_lengths
        if self.pitch_extract is not None:
            pitch, pitch_lengths = self.pitch_extract(
                out_speech,
                out_speech_lengths,
                feats_lengths=feats_lengths,
                durations=durations,
                durations_lengths=durations_lengths,
            )
        if self.energy_extract is not None:
            energy, energy_lengths = self.energy_extract(
                out_speech,
                out_speech_lengths,
                feats_lengths=feats_lengths,
                durations=durations,
                durations_lengths=durations_lengths,
            )

        # store in dict
        feats_dict = dict(feats=feats, feats_lengths=feats_lengths)
        if pitch is not None:
            feats_dict.update(pitch=pitch, pitch_lengths=pitch_lengths)
        if energy is not None:
            feats_dict.update(energy=energy, energy_lengths=energy_lengths)

        return feats_dict

    def inference(
        self,
        in_speech: torch.Tensor,
        out_speech: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        **decode_config,
    ) -> Dict[str, torch.Tensor]:
        """Calculate features and return them as a dict.

        Args:
            in_speech (Tensor): Speech waveform tensor (T_wav).
            out_speech (Tensor): Speech waveform tensor (T_wav).
            spembs (Optional[Tensor]): Speaker embedding tensor (D,).
            sids (Optional[Tensor]): Speaker ID tensor (1,).
            lids (Optional[Tensor]): Language ID tensor (1,).
            durations (Optional[Tensor): Duration tensor.
            pitch (Optional[Tensor): Pitch tensor.
            energy (Optional[Tensor): Energy tensor.

        Returns:
            Dict[str, Tensor]: Dict of outputs.

        """
        if self.input_feats_extract is not None:
            x = self.input_feats_extract(in_speech[None])[0][0]
        else:
            # Use precalculated feats (feats_type != raw case)
            x = in_speech
        if self.input_normalize is not None:
            x = self.input_normalize(x[None])[0][0]
        input_dict = dict(x=x)
        if decode_config["use_teacher_forcing"] or getattr(self.vc, "use_gst", False):
            if out_speech is None:
                raise RuntimeError("missing required argument: 'speech'")
            if self.feats_extract is not None:
                feats = self.feats_extract(out_speech[None])[0][0]
            else:
                # Use precalculated feats (feats_type != raw case)
                feats = out_speech
            if self.normalize is not None:
                feats = self.normalize(feats[None])[0][0]
            input_dict.update(feats=feats)
            if self.vc.require_raw_speech:
                input_dict.update(speech=out_speech)

        if decode_config["use_teacher_forcing"]:
            if durations is not None:
                input_dict.update(durations=durations)

            if self.pitch_extract is not None:
                pitch = self.pitch_extract(
                    out_speech[None],
                    feats_lengths=torch.LongTensor([len(feats)]),
                    durations=durations[None],
                )[0][0]
            if self.pitch_normalize is not None:
                pitch = self.pitch_normalize(pitch[None])[0][0]
            if pitch is not None:
                input_dict.update(pitch=pitch)

            if self.energy_extract is not None:
                energy = self.energy_extract(
                    out_speech[None],
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

        output_dict = self.vc.inference(**input_dict, **decode_config)

        if self.normalize is not None and output_dict.get("feat_gen") is not None:
            # NOTE: normalize.inverse is in-place operation
            feat_gen_denorm = self.normalize.inverse(
                output_dict["feat_gen"].clone()[None]
            )[0][0]
            output_dict.update(feat_gen_denorm=feat_gen_denorm)

        return output_dict
