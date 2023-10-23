#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.nn import functional as F
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.ssl.loss.abs_loss import AbsLoss
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.hubert.hubert_loss import HubertPretrainLoss
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator

class ESPnetSSLModel(AbsESPnetModel):
    """A generic SSL model"""

    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        masker: AbsMasker,
        losses: List[AbsLoss],
        ignore_id: int = -1,
        vocab_size: int = None,
        token_list: Union[Tuple[str, ...], List[str]] = None,
    ):

        assert check_argument_types()

        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        self.losses = losses
        self.error_calculator = None

        self.nan_loss_count = 0.0

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor = None,
        text_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape)

        batch_size = speech.shape[0]

        # want to modify so encoded is [outs]. Where out is the output of a specific layer
        # this is important for w2v-bert 
        encoded, mask_info, feature_penalty = self.encode(
            speech, speech_lengths
        )

        total_loss = 0.
        total_stats = {}
        for loss_fn in self.losses:
            
            loss, stats = loss_fn(encoded, text, mask_info)
            total_loss += loss
            total_stats.update(stats)

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        total_loss, total_stats, weight = force_gatherable((total_loss, total_stats, batch_size), loss.device)
        return total_loss, total_stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ):
        """Frontend + Encoder"""

        """Returns (encoded, feat_penalty)"""

        # 1 Normalize Waveform
        if self.encoder.normalize_feats:
            speech = F.layer_norm(speech, speech.shape)
        
        # 2. Extract feats
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)

        # 3. Data augmentation
        if self.specaug is not None and self.training:
            feats, feats_lengths = self.specaug(feats, feats_lengths)

        # 4. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.normalize is not None:
            feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 5. Masking
        if self.masker is not None:
            encoder_in, mask_info = self.masker(feats, feats_lengths)

        # 6. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out = self.encoder(feats, feats_lengths, y_pad, y_pad_length)

        # for now returning as array
        # need to modify so self.encoder returns list of layer representations
        return [encoder_out], mask_info, feat_penalty

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

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