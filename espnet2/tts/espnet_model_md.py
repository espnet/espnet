from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator as ASRErrorCalculator
from espnet.nets.e2e_mt_common import ErrorCalculator as MTErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.nets_utils import pad_list
from sklearn.metrics import f1_score
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
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetTTSMDModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        asr_encoder: AbsEncoder,
        asr_decoder: AbsDecoder,
        ctc: CTC,
        feats_extract: Optional[AbsFeatsExtract],
        pitch_extract: Optional[AbsFeatsExtract],
        energy_extract: Optional[AbsFeatsExtract],
        normalize: Optional[AbsNormalize and InversibleInterface],
        pitch_normalize: Optional[AbsNormalize and InversibleInterface],
        energy_normalize: Optional[AbsNormalize and InversibleInterface],
        tts: AbsTTS,
        asr_weight: float = 0.5,
        mt_weight: float = 0.0,
        mtlalpha: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        report_bleu: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        speech_attn: bool = False,
    ):
        assert check_argument_types()
        assert 0.0 < asr_weight < 1.0, "asr_weight should be (0.0, 1.0)"
        assert 0.0 <= mt_weight < 1.0, "mt_weight should be [0.0, 1.0)"
        assert 0.0 <= mtlalpha < 1.0, "mtlalpha should be [0.0, 1.0)"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.feats_extract = feats_extract
        self.pitch_extract = pitch_extract
        self.energy_extract = energy_extract
        self.normalize = normalize
        self.pitch_normalize = pitch_normalize
        self.energy_normalize = energy_normalize
        self.tts = tts
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.asr_weight = asr_weight
        self.mt_weight = mt_weight
        self.mtlalpha = mtlalpha
        self.token_list = token_list.copy()
        self.speech_attn = speech_attn
        self.specaug = None
        self.preencoder = None
        self.postencoder = None

        self.frontend = frontend
        self.asr_encoder = asr_encoder
        self.asr_decoder = asr_decoder

        # if self.CRF_loss:
        #     self.criterion_st = CRF(self.vocab_size,batch_first=True)
        #     # we should check the normalization that we provide in this.
        # else:
        #     self.criterion_st = LabelSmoothingLoss(
        #         size=vocab_size,
        #         padding_idx=ignore_id,
        #         smoothing=lsm_weight,
        #         normalize_length=token_normalized_loss,
        #     )

        self.criterion_asr = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # submodule for ASR task
        assert (asr_decoder is not None), "ASR decoder needs to be present for MD"
        # assert (
        #     src_token_list is not None
        # ), "Missing src_token_list, cannot add asr module to st model"
        if self.mtlalpha > 0.0:
            self.ctc = ctc
        self.asr_decoder = asr_decoder

        # MT error calculator
        if report_bleu:
            self.mt_error_calculator = MTErrorCalculator(
                token_list, sym_space, sym_blank, report_bleu
            )
        else:
            self.mt_error_calculator = None

        # ASR error calculator
        if report_cer or report_wer:
            self.asr_error_calculator = ASRErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.asr_error_calculator = None

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        # src_text: Optional[torch.Tensor],
        # src_text_lengths: Optional[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
            text: (Batch, Length)
            text_lengths: (Batch,)
            src_text: (Batch, length)
            src_text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        print(text_lengths.shape)
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        # additional checks with valid src_text
        # assert src_text is not None, "missing source text for asr sub-task of ST"
        # assert src_text_lengths.dim() == 1, src_text_lengths.shape
        # assert text.shape[0] == src_text.shape[0] == src_text_lengths.shape[0], (
        #     text.shape,
        #     src_text.shape,
        #     src_text_lengths.shape,
        # )

        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]
        # if src_text is not None:
        #     src_text = src_text[:, : src_text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # 2a. ASR Decoder
        (
            loss_asr_att,
            acc_asr_att,
            cer_asr_att,
            wer_asr_att,
            hs_dec_asr,
        ) = self._calc_asr_att_loss(encoder_out, encoder_out_lens, text, text_lengths)

        # 2b. CTC branch
        if self.mtlalpha > 0:
            loss_asr_ctc, cer_asr_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
        else:
            loss_asr_ctc, cer_asr_ctc = 0, None

        # 3a. MT Encoder
        dec_asr_lengths = text_lengths + 1
        # if self.encoder_mt is not None:
        #     encoder_mt_out, encoder_mt_out_lens, _ = self.encoder_mt(hs_dec_asr, dec_asr_lengths)
        # else:
        #     encoder_mt_out, encoder_mt_out_lens = hs_dec_asr, dec_asr_lengths

        # if self.speech_attn:
        #     speech_out = encoder_out
        #     speech_lens = encoder_out_lens
        # else:
        #     speech_out = None
        #     speech_lens = None
        # # 2a. Attention-decoder branch (ST)
        # loss_st_att, acc_st_att, bleu_st_att = self._calc_mt_att_loss(
        #     encoder_mt_out, encoder_mt_out_lens, text, text_lengths, speech_out, speech_lens
        # )
        with autocast(False):
            # Extract features
            if self.feats_extract is not None:
                feats, feats_lengths = self.feats_extract(speech, speech_lengths)
            else:
                # Use precalculated feats (feats_type != raw case)
                feats, feats_lengths = speech, speech_lengths

            # Extract auxiliary features
            if self.pitch_extract is not None and pitch is None:
                pitch, pitch_lengths = self.pitch_extract(
                    speech,
                    speech_lengths,
                    feats_lengths=feats_lengths,
                    durations=durations,
                    durations_lengths=durations_lengths,
                )
            if self.energy_extract is not None and energy is None:
                energy, energy_lengths = self.energy_extract(
                    speech,
                    speech_lengths,
                    feats_lengths=feats_lengths,
                    durations=durations,
                    durations_lengths=durations_lengths,
                )

            # Normalize
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)
            if self.pitch_normalize is not None:
                pitch, pitch_lengths = self.pitch_normalize(pitch, pitch_lengths)
            if self.energy_normalize is not None:
                energy, energy_lengths = self.energy_normalize(energy, energy_lengths)
        batch = dict(
            text=hs_dec_asr,
            text_lengths=dec_asr_lengths,
            feats=feats,
            feats_lengths=feats_lengths,
        )
        tts_loss, tts_stats, tts_weight =self.tts(**batch)
        
        # 3. Loss computation
        asr_ctc_weight = self.mtlalpha
        if asr_ctc_weight == 0.0:
            loss_asr = loss_asr_att
        else:
            loss_asr = (
                asr_ctc_weight * loss_asr_ctc + (1 - asr_ctc_weight) * loss_asr_att
            )
        loss = (
            (1 - self.asr_weight) * tts_loss
            + self.asr_weight * loss_asr
        )

        # print(tts_stats)
        # import pdb;pdb.set_trace()
        stats = dict(
            loss=loss.detach(),
            loss_asr=loss_asr.detach() if type(loss_asr) is not float else loss_asr,
            loss_tts=tts_loss.detach(),
            acc_asr=acc_asr_att,
            cer_ctc=cer_asr_ctc,
            cer=cer_asr_att,
            wer=wer_asr_att,
            tts_l1_loss=tts_stats['l1_loss'],
            tts_l2_loss=tts_stats['l2_loss'],
            tts_bce_loss=tts_stats['bce_loss'],
            tts_enc_dec_attn_loss=tts_stats['enc_dec_attn_loss'],
            tts_encoder_alpha=tts_stats['encoder_alpha'],
            tts_decoder_alpha=tts_stats['decoder_alpha'],
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
        src_text: Optional[torch.Tensor],
        src_text_lengths: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

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
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.asr_encoder(feats, feats_lengths)

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

    def _calc_mt_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        speech: Optional[torch.Tensor],
        speech_lens: Optional[torch.Tensor],
    ):

        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        if self.token_decoder is not None:
            if self.speech_attn:
                decoder_out = self.token_decoder(encoder_out, encoder_out_lens, speech=speech, speech_lens=speech_lens)
            else:
                decoder_out = self.token_decoder(encoder_out, encoder_out_lens)
        else:
            # 1. Forward decoder
            if self.speech_attn:
                decoder_out, _ = self.decoder(
                    encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, speech, speech_lens
                )
            else:
                decoder_out, _ = self.decoder(
                    encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
                )

        ignore = ys_out_pad != self.ignore_id  # (B,)
        # 2. Compute attention loss
        if self.CRF_loss:
            #We should verify if what's going inside is acurate. 
            loss_att = -1 * self.criterion_st(decoder_out, ys_out_pad, mask=ignore,reduction="mean")
            y_pred= self.criterion_st.decode(decoder_out, mask=ignore)
            y_pred = [torch.tensor(x, dtype=ys_out_pad.dtype, device=ys_out_pad.device) for x in y_pred]
            y_pred_pad=pad_list(y_pred,self.ignore_id)
            numerator = torch.sum(
                y_pred_pad.masked_select(ignore) == ys_out_pad.masked_select(ignore)
            )
            denominator = torch.sum(ignore)
            acc_att = float(numerator) / float(denominator)
            y_pred_pad = y_pred_pad * ignore
        else:
            loss_att = self.criterion_st(decoder_out, ys_out_pad)
            acc_att = th_accuracy(
                decoder_out.view(-1, self.vocab_size),
                ys_out_pad,
                ignore_label=self.ignore_id,
            )
            y_pred_pad = decoder_out.argmax(2) * ignore

        if self.token_decoder is not None:
            ys_out = (ys_out_pad * ignore).view(-1).tolist()
            bleu_att = f1_score(ys_out,y_pred_pad.view(-1).tolist(),labels=range(-1,self.vocab_size),average="micro")
        elif self.training or self.mt_error_calculator is None:
            bleu_att = None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            bleu_att = self.mt_error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, bleu_att

    def _calc_asr_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _, hs_dec_asr = self.asr_decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, return_hidden=True
        )

        # 2. Compute attention loss
        loss_att = self.criterion_asr(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.asr_error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.asr_error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att, hs_dec_asr

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.asr_error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.asr_error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc
