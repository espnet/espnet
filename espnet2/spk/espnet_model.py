# Copyright 2023 Jee-weon Jung
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from contextlib import contextmanager
from itertools import permutations
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from packaging.version import parse as V
from typeguard import check_argument_types

# TODO: remove unused modules
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.diar.attractor.abs_attractor import AbsAttractor
from espnet2.diar.decoder.abs_decoder import AbsDecoder
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.nets_utils import to_device

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetSpeakerModel(AbsESPnetModel):
    """
    Speaker embedding extraction model.
    Core model for diverse speaker-related tasks (e.g., verification, open-set
    identification, diarization)
    """

    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: AbsEncoder,
        aggregator
    ):
        assert check_argument_types()

        super().__init__()

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        spk_labels: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Feed-forward through encoder layers and aggregate into utterance-level
        feature.

        Args:
            speech: (Batch, samples)
            speech_lengths: (Batch,)
            spk_labels: (Batch, )
        """
        assert speech.shape[0] == spk_labels.shape[0], (speech.shape, spk_labels.shape)
        batch_size = speech.shape[0]

        # 1. extract low-level feats (e.g., mel-spectrogram or MFCC)
        feats = self.extract_feats(speech)

        frame_level_feats = self.encode_frame(speech)

        # 2. aggregation into utterance-level

