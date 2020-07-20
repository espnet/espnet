from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.fastspeech import FastSpeech
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.tts.tacotron2 import Tacotron2
from espnet2.tts.transformer import Transformer


class ESPnetTTSModel(AbsESPnetModel):
    def __init__(
        self,
        feats_extract: Optional[AbsFeatsExtract],
        normalize: Optional[AbsNormalize and InversibleInterface],
        tts: AbsTTS,
    ):
        assert check_argument_types()
        super().__init__()
        self.feats_extract = feats_extract
        self.normalize = normalize
        self.tts = tts

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spembs: torch.Tensor = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)

        if self.normalize is not None:
            feats, feats_lengths = self.normalize(feats, feats_lengths)

        return self.tts(
            text=text,
            text_lengths=text_lengths,
            speech=feats,
            speech_lengths=feats_lengths,
            spembs=spembs,
            **kwargs,
        )

    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spembs: torch.Tensor = None,
        spcs: torch.Tensor = None,
        spcs_lengths: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.feats_extract is not None:
            feats, feats_lengths = self.feats_extract(speech, speech_lengths)
        else:
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def inference(
        self,
        text: torch.Tensor,
        spembs: torch.Tensor = None,
        speech: torch.Tensor = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_teacher_forcing: bool = False,
        use_att_constraint: bool = False,
        backward_window: int = 1,
        forward_window: int = 3,
        speed_control_alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        kwargs = {}

        if isinstance(self.tts, (Tacotron2, Transformer)):
            kwargs.update(
                {
                    "threshold": threshold,
                    "maxlenratio": maxlenratio,
                    "minlenratio": minlenratio,
                    "use_teacher_forcing": use_teacher_forcing,
                }
            )
        if isinstance(self.tts, Tacotron2):
            kwargs.update(
                {
                    "use_att_constraint": use_att_constraint,
                    "forward_window": forward_window,
                    "backward_window": backward_window,
                }
            )
        if isinstance(self.tts, FastSpeech):
            kwargs.update({"alpha": speed_control_alpha})

        if use_teacher_forcing or getattr(self.tts, "use_gst", False):
            if speech is None:
                raise RuntimeError("missing required argument: 'speech'")
            if self.feats_extract is not None:
                speech = self.feats_extract(speech[None])[0][0]
            if self.normalize is not None:
                speech = self.normalize(speech[None])[0][0]
            kwargs["speech"] = speech
        outs, probs, att_ws = self.tts.inference(text=text, spembs=spembs, **kwargs)

        if self.normalize is not None:
            # NOTE: normalize.inverse is in-place operation
            outs_denorm = self.normalize.inverse(outs[None])[0][0]
        else:
            outs_denorm = outs
        return outs, outs_denorm, probs, att_ws
