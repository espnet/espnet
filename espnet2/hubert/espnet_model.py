#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert

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
        self.criterion_att = HubertPretrainLoss(
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
            return 0, 0
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

        # 1. Compute attention loss
        loss_att, logp_m_list, logp_u_list = self.criterion_att(
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

        acc_att_m = corr_masked / (count_masked + 1e-10)
        acc_att_u = corr_unmask / (count_unmask + 1e-10)

        return loss_att, acc_att_m, acc_att_u
