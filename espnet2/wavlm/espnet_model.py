#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# WavLM self-supervised pretraining model.
#     Paper: https://arxiv.org/abs/2110.13900
#     Code: https://github.com/microsoft/unilm/tree/master/wavlm

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.amp import autocast
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.legacy.nets.e2e_asr_common import ErrorCalculator
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.wavlm.utterance_mixing import UtteranceMixing


class WavLMPretrainModel(AbsESPnetModel):
    """WavLM Pretrain model.

    This is a thin wrapper that delegates the self-supervised pretraining
    objective to the encoder. It is designed to interface with a
    :class:`HuggingFaceTransformersEncoder` (or any ``AbsEncoder``) that wraps a
    WavLM pretraining model and computes the loss internally. The encoder is
    expected to return a dict containing at least a ``"loss"`` key; any extra
    scalar entries are surfaced as training statistics.
    """

    @typechecked
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
        utterance_mixing: bool = False,
        utterance_mixing_conf: Optional[Dict] = None,
        **kwargs,
    ):

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

        # WavLM speech denoising: mix each primary utterance with in-batch
        # secondary segments while keeping the clean-primary cluster targets.
        # NOTE: the recipe normally does this earlier, in HuBERTCollateFn
        # (`collate_fn_conf.mix_speech: true`), which can also draw the
        # interfering segment from a noise corpus. Enable only one of the two.
        if utterance_mixing:
            self.utterance_mixing = UtteranceMixing(**(utterance_mixing_conf or {}))
        else:
            self.utterance_mixing = None

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

        # WavLM utterance mixing / speech denoising (train only). Corrupts the
        # input waveform; the k-means targets remain those of the clean primary.
        if self.training and self.utterance_mixing is not None:
            speech, speech_lengths = self.utterance_mixing(speech, speech_lengths)

        # 1. Encoder (the encoder masks its own inputs and, given the k-means
        #    cluster targets, computes the masked-prediction loss internally)
        encoder_out = self.encode(speech, speech_lengths, text, text_lengths)

        # 2. WavLM criterion: loss is delegated to the encoder
        loss, stats = self._calc_wavlm_loss(encoder_out)

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
    ) -> Dict[str, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            y_pad: (Batch, Length) per-frame k-means cluster targets
            y_pad_length: (Batch, )
        """
        with autocast("cuda", enabled=False):
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
        # feats: (Batch, Length, Dim), y_pad: cluster targets
        # -> encoder_out: dict containing at least a "loss" key
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

    def _calc_wavlm_loss(
        self,
        encoder_out: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # The encoder (e.g. HuggingFaceTransformersEncoder wrapping WavLM)
        # computes the self-supervised loss and returns it in a dict.
        assert isinstance(encoder_out, dict) and "loss" in encoder_out, (
            "The encoder is expected to return a dict containing a 'loss' key "
            "for WavLM pretraining; got: {}".format(type(encoder_out))
        )
        loss = encoder_out["loss"]

        stats = dict(loss=loss.detach())

        # Surface any additional scalar statistics reported by the encoder
        # (e.g. accuracy of the masked prediction objective).
        for key, value in encoder_out.items():
            if key == "loss" or not isinstance(value, torch.Tensor):
                continue
            if value.numel() == 1:
                stats[key] = value.detach()

        return loss, stats
