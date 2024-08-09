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
from packaging.version import parse as V
from torch.nn import functional as F
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.ssl.loss.abs_loss import AbsLoss
from espnet2.ssl.mask.abs_mask import AbsMasker
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class ESPnetSSLModel(AbsESPnetModel):
    """A generic SSL model"""

    @typechecked
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
        feature_grad_mult: Optional[float] = 0.1,
        extract_feats_in_collect_stats: bool = True,
    ):

        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        self.losses = torch.nn.ModuleList(losses)
        self.error_calculator = None
        self.masker = masker

        self.nan_loss_count = 0.0
        self.feature_grad_mult = feature_grad_mult

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor = None,
        text_lengths: torch.Tensor = None,
        use_mask: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        assert speech.shape[0] == speech_lengths.shape[0], (
            speech.shape,
            speech_lengths.shape,
        )

        batch_size = speech.shape[0]

        encoded, mask_info, feature_penalty = self.encode(speech, speech_lengths)

        # TO DO: support multiple loss fns
        total_loss, total_stats = self.losses[0](
            encoded, text, mask_info, feature_penalty
        )
        total_stats["loss"] = total_loss.detach().item()

        del encoded, mask_info, feature_penalty

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        total_loss, total_stats, weight = force_gatherable(
            (total_loss, total_stats, batch_size), total_loss.device
        )

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
        y_pad: torch.Tensor = None,
        y_pad_length: torch.Tensor = None,
        use_mask=True,
        use_final_output: bool = True,
    ):
        """Frontend + Encoder"""

        """Returns (encoded, feat_penalty)"""

        # 1. Extract feats
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)

        if (
            self.train
            and self.feature_grad_mult is not None
            and self.feature_grad_mult < 1.0
        ):
            feats = GradMultiply.apply(feats, self.feature_grad_mult)
        features_pen = feats.float().pow(2).mean()

        # 2. Data augmentation
        if self.specaug is not None and self.training:
            feats, feats_lengths = self.specaug(feats, feats_lengths)

        # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.normalize is not None:
            feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)
        # 4. Masking
        mask_info = None
        pad_masks = None
        if use_mask and self.masker is not None:
            pad_masks = make_pad_mask(feats_lengths).to(feats.device)
            encoder_in, mask_info = self.masker(feats, pad_masks)

        # 5. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, out_lens, _ = self.encoder(
            feats, feats_lengths, masks=pad_masks, return_all_hs=True
        )

        if use_final_output:
            encoder_out = encoder_out[1][:-1] + [encoder_out[0]]
        else:
            encoder_out = encoder_out[1]
        del feats, feats_lengths, pad_masks

        return encoder_out, mask_info, features_pen

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
        del speech, speech_lengths
        return feats, feats_lengths


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None
