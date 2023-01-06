import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator as ASRErrorCalculator
from espnet.nets.e2e_mt_common import ErrorCalculator as MTErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetS2STModel(AbsESPnetModel):
    """ESPnet speech-to-speech translation model"""

    def __init__(
        self,
        s2st_type: str,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        src_normalize: Optional[AbsNormalize],
        tgt_normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        asr_decoder: Optional[AbsDecoder],
        st_decoder: Optional[AbsDecoder],
        synthesizer: Optional[AbsSynthesizer],
        asr_ctc: Optional[CTC],
        st_ctc: Optional[CTC],
        losses: Dict[str, torch.nn.Module],
        vocab_size: Optional[int],
        token_list: Optional[Union[Tuple[str, ...], List[str]]],
        src_vocab_size: Optional[int],
        src_token_list: Optional[Union[Tuple[str, ...], List[str]]],
        ignore_id: int = -1,
        report_cer: bool = True,
        report_wer: bool = True,
        report_bleu: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
    ):
        assert check_argument_types()

        super().__init__()
        self.sos = vocab_size - 1 if vocab_size else None
        self.eos = vocab_size - 1 if vocab_size else None
        self.src_sos = src_vocab_size - 1 if src_vocab_size else None
        self.src_eos = src_vocab_size - 1 if src_vocab_size else None
        self.vocab_size = vocab_size
        self.src_vocab_size = src_vocab_size
        self.ignore_id = ignore_id
        self.token_list = token_list.copy()
        self.s2st_type = s2st_type

        self.frontend = frontend
        self.specaug = specaug
        self.src_normalize = src_normalize
        self.tgt_normalize = tgt_normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder
        self.asr_decoder = asr_decoder
        self.st_decoder = st_decoder
        self.synthesizer = synthesizer
        self.asr_ctc = asr_ctc
        self.st_ctc = st_ctc
        self.losses = losses

        # MT error calculator
        if mt_decoder and vocab_size and report_bleu:
            self.mt_error_calculator = MTErrorCalculator(
                token_list, sym_space, sym_blank, report_bleu
            )
        else:
            self.mt_error_calculator = None

        # ASR error calculator
        if asr_decoder and src_vocab_size and (report_cer or report_wer):
            assert (
                src_token_list is not None
            ), "Missing src_token_list, cannot add asr module to st model"
            self.asr_error_calculator = ASRErrorCalculator(
                src_token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.asr_error_calculator = None

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        src_speech: torch.Tensor,
        src_speech_lengths: torch.Tensor,
        tgt_speech: torch.Tensor,
        tgt_speech_lengths: torch.Tensor,
        tgt_text: Optional[torch.Tensor] = None,
        tgt_text_lengths: Optional[torch.Tensor] = None,
        src_text: Optional[torch.Tensor] = None,
        src_text_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,  # TODO(Jiatong)
        sids: Optional[torch.Tensor] = None,  # TODO(Jiatong)
        lids: Optional[torch.Tensor] = None,  # TODO(Jiatong)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        # TODO(jiatong): add comments etc.

        assert (
            src_speech.shape[0]
            == src_speech_lengths.shape[0]
            == tgt_speech.shape[0]
            == tgt_speech_lengths.shape[0]
        ), (
            src_speech.shape,
            src_speech_lengths.shape,
            tgt_speech.shape,
            tgt_speech_lengths.shape,
        )

        # additional checks with valid tgt_text and src_text
        if tgt_text is not None:
            assert tgt_text_lengths.dim() == 1, tgt_text_lengths.shape
            assert (
                src_speech.shape[0]
                == src_text.shape[0]
                == src_text_lengths.shape[0]
                == tgt_text.shape[0]
                == tgt_text_lengths.shape[0]
            ), (
                src_speech.shape,
                src_text.shape,
                src_text_lengths.shape,
                tgt_text.shape,
                tgt_text_lengths.shape,
            )

        batch_size = src_speech.shape[0]
        # for data-parallel
        src_speech = src_speech[:, : src_speech_lengths.max()]
        if src_text is not None:
            src_text = src_text[:, : src_text_lengths.max()]
        if tgt_text is not None:
            tgt_text = tgt_text[:, : tgt_text_lengths.max()]
        tgt_speech = tgt_speech[:, : tgt_speech_lengths.max()]

        # 0. Target feature extract
        # Note(jiatong): only for spec for now
        tgt_feats, tgt_feats_lengths = self._extract_feats(
            tgt_speech.tgt_speech_lengths
        )
        # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.tgt_normalize is not None:
            tgt_feats, tgt_feats_lengths = self.tgt_normalize(feats, feats_lengths)

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(
            src_speech, src_speech_lengths, tgt_speech, tgt_speech_lengths
        )

        loss_record = []
        if self.s2st_type == "translatotron":
            # use a shared encoder with three decoders (i.e., asr, st, s2st)
            # reference https://arxiv.org/pdf/1904.06037.pdf

            # src_ctc
            if self.src_ctc is not None and "src_ctc" in self.losses:
                src_ctc_loss, cer_src_ctc = self._calc_ctc_loss(
                    encoder_out,
                    encoder_out_lens,
                    src_text,
                    src_text_lengths,
                    ctc_type="src",
                )
                loss_record.append(src_ctc_loss * self.losses["src_ctc"].weight)
            else:
                src_ctc_loss, cer_src_ctc = None, None

            # asr decoder
            if self.asr_decoder is not None and "src_attn" in self.losses:
                (
                    src_attn_loss,
                    acc_src_attn,
                    cer_src_attn,
                    wer_src_attn,
                ) = self._calc_asr_att_loss(
                    encoder_out, encoder_out_lens, src_text, src_text_lengths
                )
                loss_record.append(src_attn_loss * self.losses["src_attn_loss"].weight)
            else:
                src_attn_loss, acc_src_attn, cer_src_attn, wer_src_attn = (
                    None,
                    None,
                    None,
                    None,
                )

            # st decoder
            if self.st_decoder is not None and "tgt_attn" in self.losses:
                tgt_attn_loss, acc_tgt_attn, bleu_tgt_attn = self._calc_st_att_loss(
                    encoder_out, encoder_out_lens, tgt_text, tgt_text_lengths
                )
                loss_record.append(tgt_attn_loss * self.losses["tgt_attn_loss"].weight)
            else:
                tgt_attn_loss, acc_tgt_attn, bleu_tgt_attn = None, None, None

            # synthesizer
            assert (
                "synthesis" in self.losses
            ), "must have synthesis loss in the losses for S2ST"
            (
                after_outs,
                before_outs,
                logits,
                att_ws,
                stop_labels,
                updated_tgt_feats_lengths,
            ) = self.synthesizer(
                encoder_out,
                encoder_out_lens,
                tgt_feats,
                tgt_feats_lengths,
                spembs,
                sids,
                lids,
            )

            syn_loss, l1_loss, mse_loss, bce_loss = self.losses["synthesis"](
                after_outs,
                before_outs,
                logits,
                tgt_feats,
                stop_labels,
                updated_tgt_feats_lengths,
            )
            loss_record.append(syn_loss * self.losses["synthesis"].weight)

            loss = sum(loss_record)

            stats = dict(
                loss=loss.item(),
                src_ctc_loss=src_ctc_loss.item(),
                cer_src_ctc=cer_src_ctc,
                src_attn_loss=src_attn_loss.item(),
                acc_src_attn=acc_src_attn,
                cer_src_attn=cer_src_attn,
                wer_src_attn=wer_src_attn,
                tgt_attn_loss=tgt_attn_loss.item(),
                acc_tgt_attn=acc_tgt_attn,
                bleu_tgt_attn=bleu_tgt_attn,
                syn_loss=syn_loss.item(),
                syn_l1_loss=l1_loss.item(),
                syn_mse_loss=mse_loss.item(),
                syn_bce_loss=bce_loss.item(),
            )

        elif self.s2st_type == "translatotron2":
            # use a sinlge decoder for synthesis
            # reference https://arxiv.org/pdf/2107.08661v5.pdf

            # src_ctc
            if self.src_ctc is not None and "src_ctc" in self.losses:
                src_ctc_loss, cer_src_ctc = self._calc_ctc_loss(
                    encoder_out,
                    encoder_out_lens,
                    src_text,
                    src_text_lengths,
                    ctc_type="src",
                )
                loss_record.append(src_ctc_loss * self.losses["src_ctc"].weight)
            else:
                src_ctc_loss, cer_src_ctc = None, None

            # st decoder
            if self.st_decoder is not None and "tgt_attn" in self.losses:
                (
                    tgt_attn_loss,
                    acc_tgt_attn,
                    bleu_tgt_attn,
                    decoder_out,
                ) = self._calc_st_att_loss(
                    encoder_out,
                    encoder_out_lens,
                    tgt_text,
                    tgt_text_lengths,
                    return_hidden=True,
                )
                loss_record.append(tgt_attn_loss * self.losses["tgt_attn_loss"].weight)
            else:
                tgt_attn_loss, acc_tgt_attn, bleu_tgt_attn, decoder_out = (
                    None,
                    None,
                    None,
                    None,
                )

            # TODO(jiatong): to re-organize codebase

        elif self.s2st_type == "discrete_unit":
            # discrete unit-based synthesis

            # TODO(jiatong): add calculation
            pass

        else:
            raise ValueError(
                "Not supported s2st type {}, available type include ('translatotron', 'translatotron2', 'discrete_unit')"
            )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        src_speech: torch.Tensor,
        src_speech_lengths: torch.Tensor,
        tgt_speech: torch.Tensor,
        tgt_speech_lengths: torch.Tensor,
        tgt_text: Optional[torch.Tensor] = None,
        tgt_text_lengths: Optional[torch.Tensor] = None,
        src_text: Optional[torch.Tensor] = None,
        src_text_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            src_feats, src_feats_lengths = self._extract_feats(
                src_speech, src_speech_lengths
            )

            # Note(jiatong): need change for discret units
            tgt_feats, tgt_feats_lengths = self._extract_feats(
                tgt_speech, tgt_speech_lengths
            )
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            src_feats, src_feats_lengths = src_speech, src_speech_lengths
            tgt_feats, tgt_feats_lengths = tgt_speech, tgt_speech_lengths
        return {
            "src_feats": src_feats,
            "tgt_feats": tgt_feats,
            "src_feats_lengths": src_feats_lengths,
            "tgt_feats_lengths": tgt_feats_lengths,
        }

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by st_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.src_normalize is not None:
                feats, feats_lengths = self.src_normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

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

    def _calc_st_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        return_hidden: bool = False,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.st_decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_st(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.mt_error_calculator is None:
            bleu_att = None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            bleu_att = self.mt_error_calculator(ys_hat.cpu(), ys_pad.cpu())

        if return_hidden:
            return loss_att, acc_att, bleu_att, decoder_out
        else:
            return loss_att, acc_att, bleu_att

    def _calc_asr_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(
            ys_pad, self.src_sos, self.src_eos, self.ignore_id
        )
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.asr_decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_asr(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.src_vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.asr_error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.asr_error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        ctc_type: str,
    ):
        if ctc_type == "src":
            ctc = self.src_ctc
        elif ctc_type == "tgt":
            ctc = self.tgt_ctc
        else:
            raise RuntimeError(
                "Cannot recognize the ctc type (need to be either 'src' or 'tgt', but found ".format(
                    ctc_type
                )
            )
        # Calc CTC loss
        loss_ctc = ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.asr_error_calculator is not None:
            ys_hat = ctc.argmax(encoder_out).data
            cer_ctc = self.asr_error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc
