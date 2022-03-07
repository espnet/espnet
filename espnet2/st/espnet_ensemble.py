import argparse
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.st.decoder.ensemble_decoder import EnsembleDecoder
from espnet2.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetSTEnsemble(AbsESPnetModel):
    """Ensemble ST ESPnet model (a wrapper)"""

    def __init__(
        self,
        models: List[AbsESPnetModel],
        configs: List[argparse.Namespace],
    ):
        assert check_argument_types()
        assert len(models) > 0, "At least one model should presents for ensemble"
        super().__init__()
        self.model_num = len(models)
        self.single_model = models[0]
        self.sos = self.single_model.sos
        self.eos = self.single_model.eos
        self.models = torch.nn.ModuleList(models)
        self.configs = configs
        decoders = []
        for model in self.models:
            decoders.append(model.decoder)
        self.decoder = EnsembleDecoder(decoders)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Dummy forward"""
        pass

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Dummy collect feats"""
        pass

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Frontend + Encoder. Note that this method is used by st_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        encoder_out = []
        encoder_out_lens = []

        for model in self.models:
            single_encoder_out, single_encoder_out_lens = model.encode(
                speech, speech_lengths
            )
            encoder_out.append(single_encoder_out)
            encoder_out_lens.append(single_encoder_out_lens)

        return encoder_out, encoder_out_lens
