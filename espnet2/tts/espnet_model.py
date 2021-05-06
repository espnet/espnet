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
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetTTSModel(AbsESPnetModel):
    def __init__(
        self,
        feats_extract: Optional[AbsFeatsExtract],
        pitch_extract: Optional[AbsFeatsExtract],
        energy_extract: Optional[AbsFeatsExtract],
        normalize: Optional[AbsNormalize and InversibleInterface],
        pitch_normalize: Optional[AbsNormalize and InversibleInterface],
        energy_normalize: Optional[AbsNormalize and InversibleInterface],
        tts: AbsTTS,
    ):
        assert check_argument_types()
        super().__init__()
        self.feats_extract = feats_extract
        self.pitch_extract = pitch_extract
        self.energy_extract = energy_extract
        self.normalize = normalize
        self.pitch_normalize = pitch_normalize
        self.energy_normalize = energy_normalize
        self.tts = tts

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        durations: torch.Tensor = None,
        durations_lengths: torch.Tensor = None,
        pitch: torch.Tensor = None,
        pitch_lengths: torch.Tensor = None,
        energy: torch.Tensor = None,
        energy_lengths: torch.Tensor = None,
        spembs: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        with autocast(False):
            # Extract features
            if self.feats_extract is not None:
                feats, feats_lengths = self.feats_extract(speech, speech_lengths)
            else:
                feats, feats_lengths = speech, speech_lengths

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
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)
            if self.pitch_normalize is not None:
                pitch, pitch_lengths = self.pitch_normalize(pitch, pitch_lengths)
            if self.energy_normalize is not None:
                energy, energy_lengths = self.energy_normalize(energy, energy_lengths)

        # Update kwargs for additional auxiliary inputs
        if spembs is not None:
            kwargs.update(spembs=spembs)
        if durations is not None:
            kwargs.update(durations=durations, durations_lengths=durations_lengths)
        if self.pitch_extract is not None and pitch is not None:
            kwargs.update(pitch=pitch, pitch_lengths=pitch_lengths)
        if self.energy_extract is not None and energy is not None:
            kwargs.update(energy=energy, energy_lengths=energy_lengths)

        return self.tts(
            text=text,
            text_lengths=text_lengths,
            speech=feats,
            speech_lengths=feats_lengths,
            **kwargs,
        )

    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        durations: torch.Tensor = None,
        durations_lengths: torch.Tensor = None,
        pitch: torch.Tensor = None,
        pitch_lengths: torch.Tensor = None,
        energy: torch.Tensor = None,
        energy_lengths: torch.Tensor = None,
        spembs: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        if self.feats_extract is not None:
            feats, feats_lengths = self.feats_extract(speech, speech_lengths)
        else:
            feats, feats_lengths = speech, speech_lengths
        feats_dict = {"feats": feats, "feats_lengths": feats_lengths}

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
        if pitch is not None:
            feats_dict.update(pitch=pitch, pitch_lengths=pitch_lengths)
        if energy is not None:
            feats_dict.update(energy=energy, energy_lengths=energy_lengths)

        return feats_dict

    def inference(
        self,
        text: torch.Tensor,
        speech: torch.Tensor = None,
        spembs: torch.Tensor = None,
        durations: torch.Tensor = None,
        pitch: torch.Tensor = None,
        energy: torch.Tensor = None,
        **decode_config,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        kwargs = {}
        if decode_config["use_teacher_forcing"] or getattr(self.tts, "use_gst", False):
            if speech is None:
                raise RuntimeError("missing required argument: 'speech'")
            if self.feats_extract is not None:
                feats = self.feats_extract(speech[None])[0][0]
            else:
                feats = speech
            if self.normalize is not None:
                feats = self.normalize(feats[None])[0][0]
            kwargs["speech"] = feats

        if decode_config["use_teacher_forcing"]:
            if durations is not None:
                kwargs["durations"] = durations

            if self.pitch_extract is not None:
                pitch = self.pitch_extract(
                    speech[None],
                    feats_lengths=torch.LongTensor([len(feats)]),
                    durations=durations[None],
                )[0][0]
            if self.pitch_normalize is not None:
                pitch = self.pitch_normalize(pitch[None])[0][0]
            if pitch is not None:
                kwargs["pitch"] = pitch

            if self.energy_extract is not None:
                energy = self.energy_extract(
                    speech[None],
                    feats_lengths=torch.LongTensor([len(feats)]),
                    durations=durations[None],
                )[0][0]
            if self.energy_normalize is not None:
                energy = self.energy_normalize(energy[None])[0][0]
            if energy is not None:
                kwargs["energy"] = energy

        if spembs is not None:
            kwargs["spembs"] = spembs

        outs, probs, att_ws = self.tts.inference(text=text, **kwargs, **decode_config)

        if self.normalize is not None:
            # NOTE: normalize.inverse is in-place operation
            outs_denorm = self.normalize.inverse(outs.clone()[None])[0][0]
        else:
            outs_denorm = outs
        return outs, outs_denorm, probs, att_ws
