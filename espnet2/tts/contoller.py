from typing import Tuple, Dict, Optional

import torch
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.normalize.abs_normalization import AbsNormalization
from espnet2.train.abs_model_controller import AbsModelController
from espnet2.tts.abs_model import AbsTTS


class TTSModelController(AbsModelController):
    def __init__(self,
                 feats_extractor: Optional[AbsFrontend],
                 normalize: Optional[AbsNormalization],
                 tts: AbsTTS,
                 ):
        assert check_argument_types()
        super().__init__()
        self.feats_extractor = feats_extractor
        self.normalize = normalize
        self.tts = tts

    def forward(self,
                input: torch.Tensor,
                input_lengths: torch.Tensor,
                output: torch.Tensor,
                output_lengths: torch.Tensor,
                spembs: torch.Tensor = None,
                spembs_lengths: torch.Tensor = None,
                spcs: torch.Tensor = None,
                spcs_lengths: torch.Tensor = None) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        assert input.size(1) >= input_lengths.max(), \
            (input.size(), input_lengths.max())
        if self.feats_extractor is not None:
            feats, feats_lens = self.feats_extractor(output, output_lengths)
        else:
            feats, feats_lens = output, output_lengths

        if self.normalize is not None:
            feats, feats_lens = self.normalize(feats, feats_lens)

        return self.tts(
            input=input,
            input_lengths=input_lengths,
            output=feats,
            output_lengths=feats_lens,
            spembs=spembs,
            spcs=spcs,
            spcs_lengths=spcs_lengths)
