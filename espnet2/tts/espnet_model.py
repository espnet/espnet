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
        spcs: torch.Tensor = None,
        spcs_lengths: torch.Tensor = None,
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
            spcs=spcs,
            spcs_lengths=spcs_lengths,
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
