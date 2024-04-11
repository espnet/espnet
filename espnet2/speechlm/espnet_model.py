from typing import Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet2.speechlm.predictor.abs_predictor import AbsPredictor
from espnet2.speechlm.postprocessor.abs_postprocessor import AbsPostProcessor
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


class ESPnetSpeechLMModel(AbsESPnetModel):
    def __init__(
        self,
        corelm: AbsCoreLM,
        predictor: AbsPredictor,
        postprocessor: Optional[AbsPostProcessor],
        token_list: List,
        codec_token_in_use: int,
        share_emb: bool = True,
        ignore_id: int = 0,
        extract_feats_in_collect_stats: bool = False,
    ):
        assert check_argument_types()
        super().__init__()

        self.corelm = corelm
        self.predictor = predictor
        self.post_processor = postprocessor
        self.token_list = token_list

        self.codec_token_in_use = codec_token_in_use
        self.ignore_id = ignore_id
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        self.emb = torch.nn.Embedding(len(token_list), 512)

    def forward(
        self,
        decoder_sequence: torch.Tensor,
        decoder_sequence_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        print(decoder_sequence)
        pass

        return loss, stats, weight

    def collect_feats(self, **kwargs):
        raise NotImplementedError
