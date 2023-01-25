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
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.hubert.hubert_loss import HubertPretrainLoss
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class TorchAudioHubertPretrainModel(AbsESPnetModel):
    """TorchAudio Hubert Pretrain model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        ignore_id: int = -1,
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
        self.error_calculator = None

        self.nan_loss_count = 0.0

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        logit_m, logit_u, feature_penalty = self.encode(
            speech, speech_lengths, text, text_lengths
        )

        # 2a. Hubert criterion
        loss = self._calc_hubert_loss(
            logit_m,
            logit_u,
            feature_penalty,
        )

        if not torch.isinf(loss) and not torch.isnan(loss):
            pass
            # logging.warning(f"loss, {loss.item() / logit_m.size(0)}")
        else:
            self.nan_loss_count += 1
            logging.warning(f"nan_loss_count, {self.nan_loss_count}")

        # log accuracies of masked and unmasked frames
        correct_m, count_m = self._compute_correct(logit_m)
        correct_u, count_u = self._compute_correct(logit_u)

        stats = dict(
            loss=loss.detach(),
            correct_m=correct_m,
            count_m=count_m,
            acc_m=correct_m / count_m,
            correct_u=correct_u,
            count_u=count_u,
            acc_u=correct_u / count_u,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

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
        y_pad: torch.Tensor,
        y_pad_length: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            y_pad: (Batch, Length, ...)
            y_pad_length: (Batch, )
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
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out = self.encoder(feats, feats_lengths, y_pad, y_pad_length)

        return encoder_out

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

    def _compute_correct(
        self,
        logits,
    ):
        if logits.numel() == 0:
            corr, count = 0, 0
        else:
            assert logits.dim() > 1, logits.shape
            max = logits.argmax(-1) == 0
            min = logits.argmin(-1) == 0
            both = max & min
            corr = max.long().sum().item() - both.long().sum().item()
            count = max.numel()
        return corr, count

    def _calc_hubert_loss(
        self,
        logit_m: Optional[torch.Tensor],
        logit_u: Optional[torch.Tensor],
        feature_penalty: torch.Tensor,
        masked_weight: float = 1.0,
        unmasked_weight: float = 0.0,
        feature_weight: float = 10.0,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute the cross-entropy loss on HuBERT masked and non-masked logits.

        Args:
            logit_m (Tensor or None): The masked logit Tensor of dimension
                `(masked_frames, final_dim)`.
            logit_u (Tensor or None): The non-masked logit Tensor of dimension
                `(unmasked_frames, final_dim)`.
            feature_penalty (Tensor): The feature mean value for additional penalty
                loss.
            masked_weight (float, optional): The weight for masked cross-entropy loss
                (Default: ``1.0``).
            unmasked_weight (float, optional): The weight for non-masked cross-entropy
                loss (Default: ``0.0``).
            feature_weight (float, optional): The weight for feature penalty loss
                (Default: ``10.0``).
            reduction (str, optional): The reduction method for cross-entropy loss
                (Default: ``"sum"``).
        Ref:
            torchaudio: examples/hubert/loss/hubert_loss.py
        """
        loss = feature_penalty * feature_weight * logit_m.shape[0]
        if logit_m is not None:
            target_m = torch.zeros(
                logit_m.shape[0], dtype=torch.long, device=logit_m.device
            )
            loss_m = torch.nn.functional.cross_entropy(
                logit_m, target_m, reduction=reduction
            )
            loss += loss_m * masked_weight
        if logit_u is not None:
            target_u = torch.zeros(
                logit_u.shape[0], dtype=torch.long, device=logit_m.device
            )
            loss_u = torch.nn.functional.cross_entropy(
                logit_u, target_u, reduction=reduction
            )
            loss += loss_u * unmasked_weight
        return loss


class HubertPretrainModel(AbsESPnetModel):
    """Hubert Pretrain model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = False,
        report_wer: bool = False,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        pred_masked_weight: float = 1.0,
        pred_nomask_weight: float = 0.0,
        loss_weights: float = 0.0,
    ):
        assert check_argument_types()

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        self.criterion_hubert = HubertPretrainLoss(
            pred_masked_weight,
            pred_nomask_weight,
            loss_weights,
        )
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out = self.encode(speech, speech_lengths, text, text_lengths)

        # 2a. Hubert criterion
        loss, acc_mask, acc_unmask = self._calc_hubert_loss(
            encoder_out,
        )

        stats = dict(
            loss=loss.detach(),
            acc_mask=acc_mask,
            acc_unmask=acc_unmask,
            acc=acc_mask,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

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
        y_pad: torch.Tensor,
        y_pad_length: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            y_pad: (Batch, Length, ...)
            y_pad_length: (Batch, )
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
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out = self.encoder(feats, feats_lengths, y_pad, y_pad_length)

        if hasattr(self.encoder, "encoder"):
            logp_m_list = self.encoder.encoder.get_logits(encoder_out, True)
            assert self.pred_masked_weight == 0 or len(logp_m_list) > 0

            logp_u_list = self.encoder.encoder.get_logits(encoder_out, False)
            assert self.pred_nomask_weight == 0 or len(logp_u_list) > 0

        return encoder_out

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

    def compute_correct(
        self,
        logits,
    ):
        if logits.numel() == 0:
            corr, count = 0, 0
        else:
            assert logits.dim() > 1, logits.shape
            max = logits.argmax(-1) == 0
            min = logits.argmin(-1) == 0
            both = max & min
            corr = max.long().sum().item() - both.long().sum().item()
            count = max.numel()
        return corr, count

    def _calc_hubert_loss(
        self,
        encoder_out: Dict[str, torch.Tensor],
    ):

        # 1. Compute hubert loss
        loss, logp_m_list, logp_u_list = self.criterion_hubert(
            self.encoder.encoder, encoder_out
        )

        corr_masked, count_masked = 0, 0
        corr_unmask, count_unmask = 0, 0
        with torch.no_grad():
            for i, logp_m in enumerate(logp_m_list):
                corr_m, count_m = self.compute_correct(logp_m)
                corr_masked += corr_m
                count_masked += count_m
            for i, logp_u in enumerate(logp_u_list):
                corr_u, count_u = self.compute_correct(logp_u)
                corr_unmask += corr_u
                count_unmask += count_u

        acc_m = corr_masked / (count_masked + 1e-10)
        acc_u = corr_unmask / (count_unmask + 1e-10)

        return loss, acc_m, acc_u
