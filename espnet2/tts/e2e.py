from typing import Tuple, Dict, Optional

import torch
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.train.abs_e2e import AbsE2E
from espnet2.tts.abs_model import AbsTTS


class TTSE2E(AbsE2E):
    def __init__(self,
                 feats_extract: Optional[AbsFrontend],
                 normalize: Optional[AbsNormalize and InversibleInterface],
                 tts: AbsTTS,
                 ):
        assert check_argument_types()
        super().__init__()
        self.feats_extract = feats_extract
        self.normalize = normalize
        self.tts = tts

    def forward(self,
                text: torch.Tensor,
                text_lengths: torch.Tensor,
                feats: torch.Tensor,
                feats_lengths: torch.Tensor,
                spembs: torch.Tensor = None,
                spembs_lengths: torch.Tensor = None,
                spcs: torch.Tensor = None,
                spcs_lengths: torch.Tensor = None) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        assert text.size(1) >= text_lengths.max(), \
            (text.size(), text_lengths.max())
        if self.feats_extract is not None:
            feats, feats_lengths = self.feats_extract(feats, feats_lengths)

        if self.normalize is not None:
            feats, feats_lengths = self.normalize(feats, feats_lengths)

        return self.tts(
            input=text,
            input_lengths=text_lengths,
            output=feats,
            output_lengths=feats_lengths,
            spembs=spembs,
            spcs=spcs,
            spcs_lengths=spcs_lengths)
