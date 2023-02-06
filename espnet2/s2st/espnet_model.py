import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.s2st.aux_attention.abs_aux_attention import AbsS2STAuxAttention
from espnet2.s2st.losses.abs_loss import AbsS2STLoss
from espnet2.s2st.synthesizer.abs_synthesizer import AbsSynthesizer
from espnet2.s2st.tgt_feats_extract.abs_tgt_feats_extract import AbsTgtFeatsExtract
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator as ASRErrorCalculator
from espnet.nets.e2e_mt_common import ErrorCalculator as MTErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos

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
        tgt_feats_extract: Optional[AbsTgtFeatsExtract],
        specaug: Optional[AbsSpecAug],
        src_normalize: Optional[AbsNormalize],
        tgt_normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        asr_decoder: Optional[AbsDecoder],
        st_decoder: Optional[AbsDecoder],
        aux_attention: Optional[AbsS2STAuxAttention],
        synthesizer: Optional[AbsSynthesizer],
        asr_ctc: Optional[CTC],
        st_ctc: Optional[CTC],
        losses: Dict[str, AbsS2STLoss],
        tgt_vocab_size: Optional[int],
        tgt_token_list: Optional[Union[Tuple[str, ...], List[str]]],
        src_vocab_size: Optional[int],
        src_token_list: Optional[Union[Tuple[str, ...], List[str]]],
        unit_vocab_size: Optional[int],
        unit_token_list: Optional[Union[Tuple[str, ...], List[str]]],
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
        self.sos = tgt_vocab_size - 1 if tgt_vocab_size else None
        self.eos = tgt_vocab_size - 1 if tgt_vocab_size else None
        self.src_sos = src_vocab_size - 1 if src_vocab_size else None
        self.src_eos = src_vocab_size - 1 if src_vocab_size else None
        self.tgt_vocab_size = tgt_vocab_size
        self.src_vocab_size = src_vocab_size
        self.unit_vocab_size = unit_vocab_size
        self.ignore_id = ignore_id
        self.tgt_token_list = tgt_token_list.copy()
        self.src_token_list = src_token_list.copy()
        self.unit_token_list = unit_token_list.copy()
        self.s2st_type = s2st_type

        self.frontend = frontend
        self.tgt_feats_extract = tgt_feats_extract
        self.specaug = specaug
        self.src_normalize = src_normalize
        self.tgt_normalize = tgt_normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder
        self.asr_decoder = asr_decoder
        self.st_decoder = st_decoder
        self.aux_attention = aux_attention
        self.synthesizer = synthesizer
        self.asr_ctc = asr_ctc
        self.st_ctc = st_ctc
        self.losses = torch.nn.ModuleDict(losses)

        # ST error calculator
        if st_decoder and tgt_vocab_size and report_bleu:
            self.mt_error_calculator = MTErrorCalculator(
                tgt_token_list, sym_space, sym_blank, report_bleu
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

        if self.s2st_type == "discrete_unit":
            assert isinstance(self.encoder, ConformerEncoder) or isinstance(
                self.encoder, TransformerEncoder
            ), "only support conformer or transformer-based encoder because the model needs to return all hiddens, which is not supported by other encoders"

        # synthesizer
        assert (
            "synthesis" in self.losses
        ), "must have synthesis loss in the losses for S2ST"

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
        **kwargs,
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
        # NOTE(jiatong): only for teaching-forcing in spectrogram
        if self.tgt_feats_extract is not None:
            tgt_feats, tgt_feats_lengths = self._extract_feats(
                tgt_speech, tgt_speech_lengths, target=True
            )
            # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.tgt_normalize is not None:
                tgt_feats, tgt_feats_lengths = self.tgt_normalize(
                    tgt_feats, tgt_feats_lengths
                )
        else:
            # NOTE(jiatong): for discrete unit case
            tgt_feats, tgt_feats_lengths = tgt_speech, tgt_speech_lengths

        # 1. Encoder
        if self.s2st_type == "discrete_unit":
            (encoder_out, inter_encoder_out), encoder_out_lens = self.encode(
                src_speech, src_speech_lengths, return_all_hiddens=True
            )
        else:
            encoder_out, encoder_out_lens = self.encode(src_speech, src_speech_lengths)

        loss_record = []

        ########################
        # Translaotron Forward #
        ########################
        if self.s2st_type == "translatotron":
            # use a shared encoder with three decoders (i.e., asr, st, s2st)
            # reference https://arxiv.org/pdf/1904.06037.pdf

            # asr_ctc
            if self.asr_ctc is not None and "asr_ctc" in self.losses:
                asr_ctc_loss, cer_asr_ctc = self._calc_ctc_loss(
                    encoder_out,
                    encoder_out_lens,
                    src_text,
                    src_text_lengths,
                    ctc_type="asr",
                )
                loss_record.append(asr_ctc_loss * self.losses["asr_ctc"].weight)
            else:
                asr_ctc_loss, cer_asr_ctc = None, None

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
                loss_record.append(src_attn_loss * self.losses["src_attn"].weight)
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
                loss_record.append(tgt_attn_loss * self.losses["tgt_attn"].weight)
            else:
                tgt_attn_loss, acc_tgt_attn, bleu_tgt_attn = None, None, None

            # NOTE(jiatong): the tgt_feats is also updated based on the reduction_factor
            (
                after_outs,
                before_outs,
                logits,
                att_ws,
                updated_tgt_feats,
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
                updated_tgt_feats,
                stop_labels,
                updated_tgt_feats_lengths,
            )
            loss_record.append(syn_loss * self.losses["synthesis"].weight)

            # NOTE(jiatong): guided attention will be not used in multi-head attention
            if (
                "syn_guided_attn" in self.losses
                and self.synthesizer.atype != "multihead"
            ):
                # NOTE(kan-bayashi): length of output for auto-regressive
                # input will be changed when r > 1
                if self.synthesizer.reduction_factor > 1:
                    updated_tgt_feats_lengths_in = updated_tgt_feats_lengths.new(
                        [
                            olen // self.reduction_factor
                            for olen in updated_tgt_feats_lengths
                        ]
                    )
                else:
                    updated_tgt_feats_lengths_in = updated_tgt_feats_lengths
                syn_guided_attn_loss = self.losses["syn_guided_attn"](
                    att_ws=att_ws,
                    ilens=encoder_out_lens,
                    olens_in=updated_tgt_feats_lengths_in,
                )
                loss_record.append(
                    syn_guided_attn_loss * self.losses["syn_guided_attn"].weight
                )
            else:
                syn_guided_attn_loss = None

            loss = sum(loss_record)

            stats = dict(
                loss=loss.item(),
                asr_ctc_loss=asr_ctc_loss.item() if asr_ctc_loss is not None else None,
                cer_asr_ctc=cer_asr_ctc,
                src_attn_loss=src_attn_loss.item()
                if src_attn_loss is not None
                else None,
                acc_src_attn=acc_src_attn,
                cer_src_attn=cer_src_attn,
                wer_src_attn=wer_src_attn,
                tgt_attn_loss=tgt_attn_loss.item()
                if tgt_attn_loss is not None
                else None,
                acc_tgt_attn=acc_tgt_attn,
                bleu_tgt_attn=bleu_tgt_attn,
                syn_loss=syn_loss.item() if syn_loss is not None else None,
                syn_guided_attn_loss=syn_guided_attn_loss.item()
                if syn_guided_attn_loss is not None
                else None,
                syn_l1_loss=l1_loss.item(),
                syn_mse_loss=mse_loss.item(),
                syn_bce_loss=bce_loss.item(),
            )

        #########################
        # Translaotron2 Forward #
        #########################
        elif self.s2st_type == "translatotron2":
            # use a sinlge decoder for synthesis
            # reference https://arxiv.org/pdf/2107.08661v5.pdf

            # asr_ctc
            if self.asr_ctc is not None and "asr_ctc" in self.losses:
                asr_ctc_loss, cer_asr_ctc = self._calc_ctc_loss(
                    encoder_out,
                    encoder_out_lens,
                    src_text,
                    src_text_lengths,
                    ctc_type="asr",
                )
                loss_record.append(asr_ctc_loss * self.losses["asr_ctc"].weight)
            else:
                asr_ctc_loss, cer_asr_ctc = None, None

            # st decoder
            if self.st_decoder is not None and "tgt_attn" in self.losses:
                (
                    tgt_attn_loss,
                    acc_tgt_attn,
                    bleu_tgt_attn,
                    decoder_out,
                    _,
                ) = self._calc_st_att_loss(
                    encoder_out,
                    encoder_out_lens,
                    tgt_text,
                    tgt_text_lengths,
                    return_last_hidden=True,
                )
                loss_record.append(tgt_attn_loss * self.losses["tgt_attn"].weight)
            else:
                tgt_attn_loss, acc_tgt_attn, bleu_tgt_attn, decoder_out = (
                    None,
                    None,
                    None,
                    None,
                )

            assert (
                self.aux_attention is not None
            ), "must have aux attention in translatotron loss"

            # NOTE(jiatong): tgt_text_lengths + 1 for <eos>
            encoder_out_mask = (
                make_pad_mask(encoder_out_lens).to(encoder_out.device).unsqueeze(1)
            )
            attention_out = self.aux_attention(
                decoder_out, encoder_out, encoder_out, mask=encoder_out_mask
            )
            decoder_out = torch.cat((decoder_out, attention_out), dim=-1)

            # NOTE(jiatong): the tgt_feats is also updated based on the reduction_factor
            # TODO(jiatong): use non-attentive tacotron-based synthesizer
            (
                after_outs,
                before_outs,
                logits,
                att_ws,
                updated_tgt_feats,
                stop_labels,
                updated_tgt_feats_lengths,
            ) = self.synthesizer(
                decoder_out,
                tgt_text_lengths + 1,  # NOTE(jiatong): +1 for <eos>
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
                updated_tgt_feats,
                stop_labels,
                updated_tgt_feats_lengths,
            )
            # loss_record.append(syn_loss * self.losses["synthesis"].weight)

            loss = sum(loss_record)

            stats = dict(
                loss=loss.item(),
                asr_ctc_loss=asr_ctc_loss.item() if asr_ctc_loss is not None else None,
                cer_asr_ctc=cer_asr_ctc,
                tgt_attn_loss=tgt_attn_loss.item()
                if tgt_attn_loss is not None
                else None,
                acc_tgt_attn=acc_tgt_attn,
                bleu_tgt_attn=bleu_tgt_attn,
                syn_loss=syn_loss.item() if syn_loss is not None else None,
                syn_l1_loss=l1_loss.item() if l1_loss is not None else None,
                syn_mse_loss=mse_loss.item() if mse_loss is not None else None,
                syn_bce_loss=bce_loss.item() if bce_loss is not None else None,
            )

        elif self.s2st_type == "discrete_unit":
            # discrete unit-based synthesis
            # Reference: https://arxiv.org/pdf/2107.05604.pdf

            encoder_layer_for_asr = len(inter_encoder_out) // 2
            encoder_layer_for_st = len(inter_encoder_out) * 2 // 3

            # asr_ctc
            if self.asr_ctc is not None and "asr_ctc" in self.losses:
                asr_ctc_loss, cer_asr_ctc = self._calc_ctc_loss(
                    inter_encoder_out[encoder_layer_for_asr],
                    encoder_out_lens,
                    src_text,
                    src_text_lengths,
                    ctc_type="asr",
                )
                loss_record.append(asr_ctc_loss * self.losses["asr_ctc"].weight)
            else:
                asr_ctc_loss, cer_asr_ctc = None, None

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
                loss_record.append(src_attn_loss * self.losses["src_attn"].weight)
            else:
                src_attn_loss, acc_src_attn, cer_src_attn, wer_src_attn = (
                    None,
                    None,
                    None,
                    None,
                )

            # st decoder
            if self.st_decoder is not None and "tgt_attn" in self.losses:
                (tgt_attn_loss, acc_tgt_attn, bleu_tgt_attn,) = self._calc_st_att_loss(
                    inter_encoder_out[encoder_layer_for_st],
                    encoder_out_lens,
                    tgt_text,
                    tgt_text_lengths,
                )
                loss_record.append(tgt_attn_loss * self.losses["tgt_attn"].weight)
            else:
                tgt_attn_loss, acc_tgt_attn, bleu_tgt_attn, decoder_out = (
                    None,
                    None,
                    None,
                    None,
                )

            # synthesizer
            (
                unit_attn_loss,
                acc_unit_attn,
                syn_hidden,
                syn_hidden_lengths,
            ) = self._calc_unit_att_loss(
                encoder_out,
                encoder_out_lens,
                tgt_speech,
                tgt_speech_lengths,
                return_all_hiddens=True,
            )

            loss_record.append(unit_attn_loss * self.losses["synthesis"].weight)

            unit_decoder_layer_for_st = len(syn_hidden) // 2

            # st_ctc
            if self.st_ctc is not None and "st_ctc" in self.losses:
                st_ctc_loss, cer_st_ctc = self._calc_ctc_loss(
                    syn_hidden[unit_decoder_layer_for_st],
                    tgt_speech_lengths + 1,
                    tgt_text,
                    tgt_text_lengths,
                    ctc_type="st",
                )
                loss_record.append(st_ctc_loss * self.losses["st_ctc"].weight)
            else:
                st_ctc_loss, cer_st_ctc = None, None

            loss = sum(loss_record)

            stats = dict(
                loss=loss.item(),
                asr_ctc_loss=asr_ctc_loss.item() if asr_ctc_loss is not None else None,
                cer_asr_ctc=cer_asr_ctc,
                src_attn_loss=src_attn_loss.item()
                if src_attn_loss is not None
                else None,
                acc_src_attn=acc_src_attn,
                cer_src_attn=cer_src_attn,
                wer_src_attn=wer_src_attn,
                tgt_attn_loss=tgt_attn_loss.item()
                if tgt_attn_loss is not None
                else None,
                acc_tgt_attn=acc_tgt_attn,
                bleu_tgt_attn=bleu_tgt_attn,
                st_ctc_loss=st_ctc_loss.item() if st_ctc_loss is not None else None,
                cer_st_ctc=cer_st_ctc,
                unit_attn_loss=unit_attn_loss.item()
                if unit_attn_loss is not None
                else None,
                acc_unit_attn=acc_unit_attn
                if acc_unit_attn is not None
                else None,
            )

        elif self.s2st_type == "unity":
            # unity
            # Reference: https://arxiv.org/pdf/2212.08055.pdf

            # asr_ctc
            if self.asr_ctc is not None and "asr_ctc" in self.losses:
                asr_ctc_loss, cer_asr_ctc = self._calc_ctc_loss(
                    encoder_out,
                    encoder_out_lens,
                    src_text,
                    src_text_lengths,
                    ctc_type="asr",
                )
                loss_record.append(asr_ctc_loss * self.losses["asr_ctc"].weight)
            else:
                asr_ctc_loss, cer_asr_ctc = None, None

            # st decoder
            assert (
                self.st_decoder is not None and "tgt_attn" in self.losses
            ), "st_decoder is necessary for unity-based model"
            (
                tgt_attn_loss,
                acc_tgt_attn,
                bleu_tgt_attn,
                decoder_out,
                _,
            ) = self._calc_st_att_loss(
                encoder_out,
                encoder_out_lens,
                tgt_text,
                tgt_text_lengths,
                return_last_hidden=True,
            )
            loss_record.append(tgt_attn_loss * self.losses["tgt_attn"].weight)

            # synthesizer
            unit_attn_loss, acc_unit_attn = self._calc_unit_att_loss(
                decoder_out, tgt_text_lengths + 1, tgt_speech, tgt_speech_lengths
            )

            stats = dict(
                loss=loss.item(),
                asr_ctc_loss=asr_ctc_loss.item() if asr_ctc_loss is not None else None,
                cer_asr_ctc=cer_asr_ctc,
                tgt_attn_loss=tgt_attn_loss.item()
                if tgt_attn_loss is not None
                else None,
                acc_tgt_attn=acc_tgt_attn,
                bleu_tgt_attn=bleu_tgt_attn,
                st_ctc_loss=st_ctc_loss.item() if st_ctc_loss is not None else None,
                cer_st_ctc=cer_st_ctc,
                unit_attn_loss=unit_attn_loss.item()
                if unit_attn_loss is not None
                else None,
                acc_unit_attn=acc_unit_attn.item()
                if acc_unit_attn is not None
                else None,
            )

        else:
            raise ValueError(
                "Not supported s2st type {}, available type include ('translatotron', 'translatotron2', 'discrete_unit')"
            )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def inference(
        self,
        src_speech: torch.Tensor,
        src_speech_lengths: Optional[torch.Tensor] = None,
        tgt_speech: Optional[torch.Tensor] = None,
        tgt_speech_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,  # TODO(Jiatong)
        sids: Optional[torch.Tensor] = None,  # TODO(Jiatong)
        lids: Optional[torch.Tensor] = None,  # TODO(Jiatong)
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_att_constraint: bool = False,
        backward_window: int = 1,
        forward_window: int = 3,
        use_teacher_forcing: bool = False,
    ) -> Dict[str, torch.Tensor]:

        assert check_argument_types()

        # 0. Target feature extract
        # NOTE(jiatong): only for teaching-forcing in spectrogram
        if tgt_speech is not None and self.tgt_feats_extract is not None:
            tgt_feats, tgt_feats_lengths = self._extract_feats(
                tgt_speech, tgt_speech_lengths, target=True
            )
            # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.tgt_normalize is not None:
                tgt_feats, tgt_feats_lengths = self.tgt_normalize(
                    tgt_feats, tgt_feats_lengths
                )
        else:
            # NOTE(jiatong): for discrete unit case
            tgt_feats, tgt_feats_lengths = tgt_speech, tgt_speech_lengths

        # 1. Encoder
        encoder_out, _ = self.encode(src_speech, src_speech_lengths)

        # 2. Decoder
        if self.s2st_type == "translatotron":
            assert encoder_out.size(0) == 1
            output_dict = self.synthesizer.inference(
                encoder_out[0],
                tgt_feats[0],
                spembs,
                sids,
                lids,
                threshold,
                minlenratio,
                maxlenratio,
                use_att_constraint,
                backward_window,
                forward_window,
                use_teacher_forcing,
            )
        elif self.s2st_type == "translatotron2":
            assert encoder_out.size(0) == 1

            output_dict = self.synthesizer.inference(
                encoder_out[0],
                tgt_feats[0],
                spembs,
                sids,
                lids,
                threshold,
                minlenratio,
                maxlenratio,
                use_att_constraint,
                backward_window,
                forward_window,
                use_teacher_forcing,
            )
        elif self.s2st_type == "discrete_unit":
            assert encoder_out.size(0) == 1
            # TODO (use beam search decoder)
        else:
            raise ValueError(
                "Not supported s2st type {}, available type include ('translatotron', 'translatotron2', 'discrete_unit')"
            )

        if self.tgt_normalize is not None and output_dict.get("feat_gen") is not None:
            # NOTE: normalize.inverse is in-place operation
            feat_gen_denorm = self.tgt_normalize.inverse(
                output_dict["feat_gen"].clone()[None]
            )[0][0]
            output_dict.update(feat_gen_denorm=feat_gen_denorm)
        return output_dict

    def collect_feats(
        self,
        src_speech: torch.Tensor,
        src_speech_lengths: torch.Tensor,
        tgt_speech: torch.Tensor,
        tgt_speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            src_feats, src_feats_lengths = self._extract_feats(
                src_speech, src_speech_lengths
            )
            return_dict = {
                "src_feats": src_feats,
                "src_feats_lengths": src_feats_lengths,
            }

            if self.tgt_feats_extract is not None:
                tgt_feats, tgt_feats_lengths = self._extract_feats(
                    tgt_speech, tgt_speech_lengths, target=True
                )
                return_dict.update(
                    tgt_feats=tgt_feats,
                    tgt_feats_lengths=tgt_feats_lengths,
                )
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            return_dict = {
                "src_feats": src_speech,
                "tgt_feats": tgt_speech,
                "src_feats_lengths": src_speech_lengths,
                "tgt_feats_lengths": tgt_speech_lengths,
            }
        
        return return_dict

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, return_all_hiddens: bool = False,
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
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths, return_all_hiddens=return_all_hiddens)
        if return_all_hiddens:
            encoder_out, inter_encoder_out = encoder_out

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

        return (encoder_out, inter_encoder_out), encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, target: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if speech_lengths is not None:
            assert speech_lengths.dim() == 1, speech_lengths.shape

            # for data-parallel
            speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            if target:
                feats, feats_lengths = self.tgt_feats_extract(speech, speech_lengths)
            else:
                feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def _calc_unit_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        return_last_hidden: bool = False,
        return_all_hiddens: bool = False,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_outs, decoder_out_lengths, = self.synthesizer(
            encoder_out,
            encoder_out_lens,
            ys_in_pad,
            ys_in_lens,
            spembs,
            sids,
            lids,
            return_last_hidden=return_last_hidden,
            return_all_hiddens=return_all_hiddens,
        )

        if return_last_hidden or return_all_hiddens:
            (decoder_out, decoder_hidden) = decoder_outs
        else:
            decoder_out = decoder_outs

        # 2. Compute attention loss
        loss_att = self.losses["synthesis"](decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.unit_vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        if return_last_hidden or return_all_hiddens:
            return loss_att, acc_att, decoder_hidden, decoder_out_lengths
        else:
            return loss_att, acc_att

    def _calc_st_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        return_last_hidden: bool = False,
        return_all_hiddens: bool = False,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        assert not return_last_hidden and not return_all_hiddens, \
             "cannot return both last hiddens or all hiddens"

        # 1. Forward decoder
        decoder_outs, decoder_out_lengths, = self.st_decoder(
            encoder_out,
            encoder_out_lens,
            ys_in_pad,
            ys_in_lens,
            return_last_hidden=return_last_hidden,
            return_all_hiddens=return_all_hiddens,
        )

        if return_last_hidden or return_all_hiddens:
            (decoder_out, decoder_hidden) = decoder_outs
        else:
            decoder_out = decoder_outs

        # 2. Compute attention loss
        loss_att = self.losses["tgt_attn"](decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.tgt_vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.mt_error_calculator is None:
            bleu_att = None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            bleu_att = self.mt_error_calculator(ys_hat.cpu(), ys_pad.cpu())

        if return_last_hidden:
            return loss_att, acc_att, bleu_att, decoder_hidden, decoder_out_lengths
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
        loss_att = self.losses["src_attn"](decoder_out, ys_out_pad)
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
        if ctc_type == "asr":
            ctc = self.asr_ctc
        elif ctc_type == "st":
            ctc = self.st_ctc
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

    @property
    def require_vocoder(self):
        """Return whether or not vocoder is required."""
        return True
