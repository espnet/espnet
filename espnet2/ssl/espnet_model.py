#!/usr/bin/env python3
# -*- coding: utf-8 -*-

<<<<<<< HEAD
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
=======
# Copyright 2025 William Chen
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from typeguard import typechecked

>>>>>>> e72ffd9f44396fd97121dc5d38893d66b18756cc
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
<<<<<<< HEAD
from espnet2.ssl.loss.abs_loss import AbsLoss
from espnet2.ssl.mask.abs_mask import AbsMasker
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
=======
from espnet2.ssl.loss.abs_loss import AbsSSLLoss
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
>>>>>>> e72ffd9f44396fd97121dc5d38893d66b18756cc
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class ESPnetSSLModel(AbsESPnetModel):
<<<<<<< HEAD
    """A generic SSL model"""
=======
    """An encoder-only SSL model.

    We currently/will support the
    following SSL objectives:
     - HuBERT
     - Data2Vec (in development)
     - DinoSR (in development)
     - wav2vec 2.0 (TODO)
     - w2v-BERT (TODO)
     - BEST-RQ (TODO)
     - Flow Matching (TODO)

     Models can be trained with
     multiple objectives by adding
     multiple entries under loss_conf
     in the training configuration.
    """
>>>>>>> e72ffd9f44396fd97121dc5d38893d66b18756cc

    @typechecked
    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
<<<<<<< HEAD
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
=======
        losses: List[AbsSSLLoss],
        util_attributes: Set[str],
        required_inputs: Set[str],
        util_modules: torch.nn.ModuleDict,
        token_list: Union[Tuple[str, ...], List[str]] = None,
        extract_feats_in_collect_stats: bool = True,
        **kwargs,
    ):

        super().__init__()
>>>>>>> e72ffd9f44396fd97121dc5d38893d66b18756cc
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        self.losses = torch.nn.ModuleList(losses)
<<<<<<< HEAD
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
=======

        self.util_attributes = util_attributes
        self.required_inputs = required_inputs
        self.util_modules = util_modules

        # track the current iteration
        # this is used for calculating decay in EMA
        self.register_buffer("global_step", torch.tensor([0]))
>>>>>>> e72ffd9f44396fd97121dc5d38893d66b18756cc

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

<<<<<<< HEAD
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
=======
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor = None,
        text_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        assert speech.shape[0] == speech_lengths.shape[0], (
            speech.shape,
            speech_lengths.shape,
        )

        if text is not None:
            assert text.shape[0] == text_lengths.shape[0], (
                text.shape,
                text_lengths.shape,
            )

        batch_size = speech.shape[0]

        total_loss = 0.0
        stats = {}

        # we move the ema model to GPU here to avoid issues with NCCL
        # when copying the ema params when initializing ddp
        if "ema" in self.util_modules:
            self.util_modules["ema"].update_device_and_type(speech.device, speech.dtype)

        # update decay parameters only after the first gradient update
        # note that this assumes accum_grad == 1
        if self.training and self.global_step > 0:
            for attr in self.util_attributes:
                if hasattr(self.util_modules[attr], "step"):
                    if attr == "ema":
                        self.util_modules[attr].step(self.global_step, self.encoder)
                        stats["ema_decay"] = self.util_modules[attr].get_decay() * 1000
                    else:
                        self.util_modules[attr].step(self.global_step)
            for loss_func in self.losses:
                if hasattr(loss_func, "step"):
                    loss_func.step(self.global_step)

        data = self.encode(speech, speech_lengths, text, text_lengths)
        data["text"] = text
        data["text_lengths"] = text_lengths

        for loss_func in self.losses:
            loss_input = {k: data[k] for k in loss_func.required_inputs}
            loss, stats_i = loss_func(**loss_input)
            total_loss += loss
            stats.update(stats_i)

        stats["loss"] = total_loss.detach().item()

        # Use loss function to override gradient
        # scaling factor. This is because we only
        # take the loss for a subset of elements
        if "sample_size" in stats:
            batch_size = stats["sample_size"]

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        total_loss, stats, weight = force_gatherable(
            (total_loss, stats, batch_size), total_loss.device
        )

        if self.training:
            self.global_step = self.global_step + 1

        return total_loss, stats, weight
>>>>>>> e72ffd9f44396fd97121dc5d38893d66b18756cc

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
<<<<<<< HEAD
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
=======

        return feats, feats_lengths

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor = None,
        text_lengths: torch.Tensor = None,
        use_final_output: bool = True,
    ) -> Dict:

        data = {}

        # 1. Frontend
        speech, speech_lengths = self._extract_feats(speech, speech_lengths)

        if "frontend" in self.required_inputs:
            data["frontend"] = speech
            data["frontend_lengths"] = speech_lengths

        # 2. Data Augmentation
        if self.specaug is not None and self.training:
            speech, speech_lengths = self.specaug(speech, speech_lengths)

        # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.normalize is not None:
            speech, speech_lengths = self.normalize(speech, speech_lengths)

        # 4. Pre-encoder
        if self.preencoder is not None:
            speech, speech_lengths = self.preencoder(speech, speech_lengths)
            if "preencoder" in self.required_inputs:
                data["preencoder"] = speech
                data["preencoder_lengths"] = speech_lengths

        pad_masks = make_pad_mask(speech_lengths).to(speech.device)

        # 5. Objective-specific pre-processing

        # EMA - data2vec / DinoSR
        if "ema" in self.util_attributes:
            with torch.no_grad():
                ema_output, ema_output_lengths, _ = self.util_modules["ema"](
                    speech, speech_lengths, masks=pad_masks, return_all_hs=True
                )
            data["ema_output"] = ema_output[1]
            data["ema_output_lengths"] = ema_output_lengths

        # Masking
        if "block_mask" in self.util_attributes or "mask" in self.util_attributes:
            # Prioritize block masking if both are available
            if "block_mask" in self.util_attributes:
                speech, mask_info = self.util_modules["block_mask"](speech, pad_masks)
            elif "mask" in self.util_attributes:
                speech, mask_info = self.util_modules["mask"](speech, pad_masks)
            data["mask_info"] = mask_info

        # Flow Matching
        # TODO (william): NOT IMPLEMENTED YET
        if "flow_preprocess" in self.util_attributes:
            speech, target = None  # TODO
            data["flow_target"] = target

        # 6. Encoder
        speech, speech_lengths, _ = self.encoder(
            speech, speech_lengths, masks=pad_masks, return_all_hs=True
        )

        # for encoder architectures that have a final norm
        if use_final_output:
            speech = speech[1][:-1] + [speech[0]]
        else:
            speech = speech[1]

        data["encoder_output"] = speech
        data["encoder_output_lengths"] = speech_lengths

        return data

    def inference_encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        use_mask: bool = False,
        use_final_output: bool = True,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # 1. Frontend
        speech, speech_lengths = self._extract_feats(speech, speech_lengths)

        # 2. Data Augmentation
        if self.specaug is not None and self.training and use_mask:
            speech, speech_lengths = self.specaug(speech, speech_lengths)

        # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.normalize is not None:
            speech, speech_lengths = self.normalize(speech, speech_lengths)

        # 4. Pre-encoder
        if self.preencoder is not None:
            speech, speech_lengths = self.preencoder(speech, speech_lengths)

        # Masking
        pad_masks = make_pad_mask(speech_lengths).to(speech.device)
        if self.training and use_mask:
            if "block_mask" in self.util_attributes or "mask" in self.util_attributes:
                # Prioritize block masking if both are available
                if "block_mask" in self.util_attributes:
                    speech, mask_info = self.util_modules["block_mask"](
                        speech, pad_masks
                    )
                elif "mask" in self.util_attributes:
                    speech, mask_info = self.util_modules["mask"](speech, pad_masks)
        # 6. Encoder
        speech, speech_lengths, _ = self.encoder(
            speech, speech_lengths, masks=pad_masks, return_all_hs=True
        )

        return speech[0], speech[1], speech_lengths
>>>>>>> e72ffd9f44396fd97121dc5d38893d66b18756cc
