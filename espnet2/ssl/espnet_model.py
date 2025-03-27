#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 William Chen
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.ssl.loss.abs_loss import AbsSSLLoss
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class ESPnetSSLModel(AbsESPnetModel):
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

    @typechecked
    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        losses: List[AbsSSLLoss],
        util_attributes: Set[str],
        required_inputs: Set[str],
        util_modules: torch.nn.ModuleDict,
        token_list: Union[Tuple[str, ...], List[str]] = None,
        extract_feats_in_collect_stats: bool = True,
        **kwargs,
    ):

        super().__init__()
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        self.losses = torch.nn.ModuleList(losses)

        self.util_attributes = util_attributes
        self.required_inputs = required_inputs
        self.util_modules = util_modules

        # track the current iteration
        # this is used for calculating decay in EMA
        self.register_buffer("global_step", torch.tensor([0]))

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
