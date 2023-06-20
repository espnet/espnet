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
from espnet2.spk.encoder.rawnet3_encoder import RawNet3Encoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.spk.pooling.abs_pooling import AbsPooling
from espnet2.spk.projector.abs_projector import AbsProjector
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet2.spk.loss import *

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
        encoder: Optional[AbsEncoder],
        pooling: Optional[AbsPooling],
        projector,
        loss,
    ):
        assert check_argument_types()

        super().__init__()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.pooling = pooling
        self.projector = projector
        self.loss = loss

    def forward(
        self,
        speech: torch.Tensor,
        #speech_lengths: torch.Tensor = None,
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
        # Will do nothing for raw waveform-based models (e.g., RawNets)
        feats = self.extract_feats(speech)

        frame_level_feats = self.encode_frame(speech)

        # 2. aggregation into utterance-level
        utt_level_feat = self.pooling(frame_level_feats)

        # 3. (optionally) go through further projection(s)
        spk_embd = self.project_spk_embd(utt_level_feat)

        # 4. calculate loss
        loss = self.loss(spk_embd, spk_labels)

        stats = dict(
            loss=loss.detach()
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight


    def extract_feats(self, speech: torch.Tensor) -> torch.Tensor:
        batch_size = speech.shape[0]

        # 1. extract feats
        if self.frontend is not None:
            feats, _ = self.frontend(speech, None)
        else:
            feats = speech

        # 2. apply augmentations
        if self.specaug is not None and self.training:
            feats, _ = self.specaug(feats, None)

        # 3. normalize
        if self.normalize is not None:
            feats, _ = self.normalize(feats, None)

        return feats

    def encode_frame(self, feats: torch.Tensor) -> torch.Tensor:
        frame_level_feats = self.encoder(feats)

        return frame_level_feats

    def aggregate(self, frame_level_feats: torch.Tensor) -> torch.Tensor:
        utt_level_feat = self.aggregator(frame_level_feats)

        return utt_level_feat

    def project_spk_embd(self, utt_level_feat: torch.Tensor) -> torch.Tensor:
        if self.projector is not None:
            spk_embd = self.projector(utt_level_feat)
        else:
            spk_embd = utt_level_feat

        return spk_embd

    def collect_feats(
        self,
        speech: torch.Tensor,
        spk_labels: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats = self._extract_feats(speech)
        return {"feats": feats}
