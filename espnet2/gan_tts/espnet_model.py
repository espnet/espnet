# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based TTS model."""

from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import Optional
from typing import Tuple

import torch

from typeguard import check_argument_types

from espnet2.gan_tts.abs_gan_tts import AbsGANTTS
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetGANTTSModel(AbsGANESPnetModel):
    def __init__(
        self,
        feats_extract: Optional[AbsFeatsExtract],
        normalize: Optional[AbsNormalize and InversibleInterface],
        tts: AbsGANTTS,
    ):
        assert check_argument_types()
        super().__init__()
        self.feats_extract = feats_extract
        self.normalize = normalize
        self.tts = tts
        assert hasattr(tts, "generator")
        assert hasattr(tts, "discriminator")

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        forward_generator: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        with autocast(False):
            # Extract features
            if self.feats_extract is not None:
                feats, feats_lengths = self.feats_extract(speech, speech_lengths)
            else:
                feats, feats_lengths = speech, speech_lengths

            # Normalize
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Update kwargs for additional auxiliary inputs
        if sids is not None:
            kwargs.update(sids=sids)
        if spembs is not None:
            kwargs.update(spembs=spembs)

        if self.tts.require_raw_speech:
            kwargs.update(feats=feats)
            kwargs.update(feats_lengths=feats_lengths)
            kwargs.update(speech=speech)
            kwargs.update(speech_lengths=speech_lengths)
        else:
            kwargs.update(speech=feats)
            kwargs.update(speech_lengths=feats_lengths)

        return self.tts(
            text=text,
            text_lengths=text_lengths,
            forward_generator=forward_generator,
            **kwargs,
        )

    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        sids: torch.Tensor = None,
        spembs: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        output_dict = {"speech": speech, "speech_lengths": speech_lengths}
        if self.feats_extract is not None:
            feats, feats_lengths = self.feats_extract(speech, speech_lengths)
            output_dict = {"feats": feats, "feats_lengths": feats_lengths}

        return output_dict

    def inference(
        self,
        text: torch.Tensor,
        speech: torch.Tensor = None,
        sids: torch.Tensor = None,
        spembs: torch.Tensor = None,
        durations: torch.Tensor = None,
        **decode_config,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        kwargs = {}
        if decode_config["use_teacher_forcing"] or getattr(self.tts, "use_gst", False):
            if self.feats_extract is not None:
                feats = self.feats_extract(speech[None])[0][0]
            else:
                feats = speech
            if self.normalize is not None:
                feats = self.normalize(feats[None])[0][0]
            if self.tts.require_raw_speech:
                kwargs["feats"] = feats
            else:
                kwargs["speech"] = feats

        if decode_config["use_teacher_forcing"]:
            if durations is not None:
                kwargs["durations"] = durations

        if spembs is not None:
            kwargs["spembs"] = spembs

        if sids is not None:
            kwargs["sids"] = sids

        outs, probs, att_ws = self.tts.inference(text=text, **kwargs, **decode_config)

        if self.normalize is not None:
            # NOTE: normalize.inverse is in-place operation
            outs_denorm = self.normalize.inverse(outs.clone()[None])[0][0]
        else:
            outs_denorm = outs

        return outs, outs_denorm, probs, att_ws
