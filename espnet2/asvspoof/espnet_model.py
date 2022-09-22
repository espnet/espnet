# Copyright 2022 Jiatong Shi (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from contextlib import contextmanager
from itertools import permutations
from typing import Dict, Optional, Tuple

import numpy as np
import logging
import torch
import torch.nn.functional as F
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asvspoof.decoder.abs_decoder import AbsDecoder
from espnet2.asvspoof.loss.abs_loss import AbsASVSpoofLoss
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


class ESPnetASVSpoofModel(AbsESPnetModel):
    """ASV Spoofing model
    A simple ASV Spoofing model
    """

    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: AbsEncoder,
        preencoder: Optional[AbsPreEncoder],
        decoder: AbsDecoder,
        losses: Dict[str, AbsASVSpoofLoss],
    ):
        assert check_argument_types()

        super().__init__()

        self.preencoder = preencoder
        self.encoder = encoder
        self.normalize = normalize
        self.frontend = frontend
        self.specaug = specaug
        self.decoder = decoder
        self.losses = losses

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        label: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss
        Args:
            speech: (Batch, samples)
            spk_labels: (Batch, )
            kwargs: "utt_id" is among the input.
        """
        assert speech.shape[0] == label.shape[0], (speech.shape, label.shape)
        batch_size = speech.shape[0]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # 2. Decoder (baiscally a predction layer after encoder_out)
        pred = self.decoder(encoder_out, encoder_out_lens)

        if "oc_softmax_loss" in self.losses:
            loss = (
                self.losses["oc_softmax_loss"](label, encoder_out) * self.losses["oc_softmax_loss"].weight
            )
            pred = self.losses["am_softmax_loss"].score(encoder_out)
        elif "am_softmax_loss" in self.losses:
            loss = (
                self.losses["am_softmax_loss"](label, encoder_out) * self.losses["am_softmax_loss"].weight
            )
            pred = self.losses["am_softmax_loss"].score(encoder_out)
        else:
            loss = (
                self.losses["binary_loss"](pred, label) * self.losses["binary_loss"].weight
            )
        acc = torch.sum(((pred.view(-1) > 0.5) == (label.view(-1) > 0.5))) / batch_size

        stats = dict(
            loss=loss.detach(),
            acc=acc.detach(),
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
            bottleneck_feats: (Batch, Length, ...): used for enh + diar
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

            # Pre-encoder, e.g. used for raw input data
            if self.preencoder is not None:
                feats, feats_lengths = self.preencoder(feats, feats_lengths)

            # 4. Forward encoder
            # feats: (Batch, Length, Dim)
            # -> encoder_out: (Batch, Length2, Dim)
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(), 
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = speech.shape[0]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else torch.ones(batch_size).int() * speech.shape[1]
        )

        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths