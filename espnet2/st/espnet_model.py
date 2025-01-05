import logging
import random
from contextlib import contextmanager
from itertools import groupby
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from torch.nn.utils.rnn import pad_sequence
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr_transducer.utils import get_transducer_task_io
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


class ESPnetSTModel(AbsESPnetModel):
    """
    CTC-attention hybrid Encoder-Decoder model.

    This model combines CTC (Connectionist Temporal Classification) and
    attention mechanisms for sequence-to-sequence tasks, particularly
    in speech translation. It utilizes various components including
    frontends, encoders, decoders, and loss functions tailored for
    automatic speech recognition (ASR) and machine translation (MT).

    Attributes:
        vocab_size (int): Size of the target vocabulary.
        token_list (List[str]): List of tokens in the target language.
        frontend (Optional[AbsFrontend]): Frontend for feature extraction.
        specaug (Optional[AbsSpecAug]): SpecAugment for data augmentation.
        normalize (Optional[AbsNormalize]): Normalization layer.
        preencoder (Optional[AbsPreEncoder]): Pre-encoder for raw input.
        encoder (AbsEncoder): Main encoder for the input sequence.
        hier_encoder (Optional[AbsEncoder]): Hierarchical encoder.
        md_encoder (Optional[AbsEncoder]): Multi-decoder encoder.
        extra_mt_encoder (Optional[AbsEncoder]): Additional encoder for MT.
        postencoder (Optional[AbsPostEncoder]): Post-encoder for output processing.
        decoder (AbsDecoder): Main decoder for generating output sequences.
        extra_asr_decoder (Optional[AbsDecoder]): Additional ASR decoder.
        extra_mt_decoder (Optional[AbsDecoder]): Additional MT decoder.
        ctc (Optional[CTC]): CTC loss for ASR task.
        st_ctc (Optional[CTC]): CTC loss for ST task.
        st_joint_network (Optional[torch.nn.Module]): Joint network for ST.
        src_vocab_size (Optional[int]): Size of the source vocabulary.
        src_token_list (Optional[List[str]]): List of tokens in the source language.
        asr_weight (float): Weight for ASR loss.
        mt_weight (float): Weight for MT loss.
        mtlalpha (float): Weight for multi-task learning.
        st_mtlalpha (float): Weight for ST multi-task learning.
        ignore_id (int): ID to ignore during loss calculation.
        tgt_ignore_id (int): Target ignore ID.
        lsm_weight (float): Label smoothing weight.
        length_normalized_loss (bool): Whether to normalize loss by length.
        report_cer (bool): Whether to report character error rate.
        report_wer (bool): Whether to report word error rate.
        report_bleu (bool): Whether to report BLEU score.
        sym_space (str): Symbol for space in target language.
        sym_blank (str): Symbol for blank in target language.
        tgt_sym_space (str): Symbol for space in target language.
        tgt_sym_blank (str): Symbol for blank in target language.
        extract_feats_in_collect_stats (bool): Flag to extract features.
        ctc_sample_rate (float): Sampling rate for CTC.
        tgt_sym_sos (str): Start of sequence symbol for target.
        tgt_sym_eos (str): End of sequence symbol for target.
        lang_token_id (Optional[torch.Tensor]): Language token ID.

    Args:
        vocab_size (int): Size of the target vocabulary.
        token_list (Union[Tuple[str, ...], List[str]]): List of tokens.
        frontend (Optional[AbsFrontend]): Frontend for feature extraction.
        specaug (Optional[AbsSpecAug]): SpecAugment for data augmentation.
        normalize (Optional[AbsNormalize]): Normalization layer.
        preencoder (Optional[AbsPreEncoder]): Pre-encoder for raw input.
        encoder (AbsEncoder): Main encoder for the input sequence.
        hier_encoder (Optional[AbsEncoder]): Hierarchical encoder.
        md_encoder (Optional[AbsEncoder]): Multi-decoder encoder.
        extra_mt_encoder (Optional[AbsEncoder]): Additional encoder for MT.
        postencoder (Optional[AbsPostEncoder]): Post-encoder for output processing.
        decoder (AbsDecoder): Main decoder for generating output sequences.
        extra_asr_decoder (Optional[AbsDecoder]): Additional ASR decoder.
        extra_mt_decoder (Optional[AbsDecoder]): Additional MT decoder.
        ctc (Optional[CTC]): CTC loss for ASR task.
        st_ctc (Optional[CTC]): CTC loss for ST task.
        st_joint_network (Optional[torch.nn.Module]): Joint network for ST.
        src_vocab_size (Optional[int]): Size of the source vocabulary.
        src_token_list (Optional[Union[Tuple[str, ...], List[str]]]): List of tokens
            in the source language.
        asr_weight (float): Weight for ASR loss.
        mt_weight (float): Weight for MT loss.
        mtlalpha (float): Weight for multi-task learning.
        st_mtlalpha (float): Weight for ST multi-task learning.
        ignore_id (int): ID to ignore during loss calculation.
        tgt_ignore_id (int): Target ignore ID.
        lsm_weight (float): Label smoothing weight.
        length_normalized_loss (bool): Whether to normalize loss by length.
        report_cer (bool): Whether to report character error rate.
        report_wer (bool): Whether to report word error rate.
        report_bleu (bool): Whether to report BLEU score.
        sym_space (str): Symbol for space in target language.
        sym_blank (str): Symbol for blank in target language.
        tgt_sym_space (str): Symbol for space in target language.
        tgt_sym_blank (str): Symbol for blank in target language.
        extract_feats_in_collect_stats (bool): Flag to extract features.
        ctc_sample_rate (float): Sampling rate for CTC.
        tgt_sym_sos (str): Start of sequence symbol for target.
        tgt_sym_eos (str): End of sequence symbol for target.
        lang_token_id (int): Language token ID.

    Returns:
        None

    Examples:
        # Initialize the model
        model = ESPnetSTModel(
            vocab_size=1000,
            token_list=["<blank>", "<sos>", "<eos>", "hello", "world"],
            frontend=None,
            specaug=None,
            normalize=None,
            preencoder=None,
            encoder=my_encoder,
            hier_encoder=None,
            md_encoder=None,
            extra_mt_encoder=None,
            postencoder=None,
            decoder=my_decoder,
            extra_asr_decoder=None,
            extra_mt_decoder=None,
            ctc=my_ctc,
            st_ctc=my_st_ctc,
            st_joint_network=None,
            src_vocab_size=500,
            src_token_list=["<blank>", "<sos>", "<eos>", "bonjour", "monde"],
            asr_weight=0.5,
            mt_weight=0.5,
            mtlalpha=0.5,
            st_mtlalpha=0.5,
            ignore_id=-1,
            tgt_ignore_id=-1,
            lsm_weight=0.1,
            length_normalized_loss=False,
            report_cer=True,
            report_wer=True,
            report_bleu=True,
            sym_space="<space>",
            sym_blank="<blank>",
            tgt_sym_space="<space>",
            tgt_sym_blank="<blank>",
            extract_feats_in_collect_stats=True,
            ctc_sample_rate=0.0,
            tgt_sym_sos="<sos/eos>",
            tgt_sym_eos="<sos/eos>",
            lang_token_id=-1,
        )
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
        hier_encoder: Optional[AbsEncoder],
        md_encoder: Optional[AbsEncoder],
        extra_mt_encoder: Optional[AbsEncoder],
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        extra_asr_decoder: Optional[AbsDecoder],
        extra_mt_decoder: Optional[AbsDecoder],
        ctc: Optional[CTC],
        st_ctc: Optional[CTC],
        st_joint_network: Optional[torch.nn.Module],
        src_vocab_size: Optional[int],
        src_token_list: Optional[Union[Tuple[str, ...], List[str]]],
        asr_weight: float = 0.0,
        mt_weight: float = 0.0,
        mtlalpha: float = 0.0,
        st_mtlalpha: float = 0.0,
        ignore_id: int = -1,
        tgt_ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        report_bleu: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        tgt_sym_space: str = "<space>",
        tgt_sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        ctc_sample_rate: float = 0.0,
        tgt_sym_sos: str = "<sos/eos>",
        tgt_sym_eos: str = "<sos/eos>",
        lang_token_id: int = -1,
    ):
        assert 0.0 <= asr_weight < 1.0, "asr_weight should be [0.0, 1.0)"
        assert 0.0 <= mt_weight < 1.0, "mt_weight should be [0.0, 1.0)"
        assert 0.0 <= mtlalpha <= 1.0, "mtlalpha should be [0.0, 1.0]"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        if tgt_sym_sos in token_list:
            self.sos = token_list.index(tgt_sym_sos)
        else:
            self.sos = vocab_size - 1
        if tgt_sym_eos in token_list:
            self.eos = token_list.index(tgt_sym_eos)
        else:
            self.eos = vocab_size - 1
        self.src_sos = src_vocab_size - 1 if src_vocab_size else None
        self.src_eos = src_vocab_size - 1 if src_vocab_size else None
        self.vocab_size = vocab_size
        self.src_vocab_size = src_vocab_size
        self.ignore_id = ignore_id
        self.tgt_ignore_id = tgt_ignore_id
        self.asr_weight = asr_weight
        self.mt_weight = mt_weight
        self.mtlalpha = mtlalpha
        self.st_mtlalpha = st_mtlalpha
        self.token_list = token_list.copy()
        self.src_token_list = src_token_list.copy()
        self.ctc_sample_rate = ctc_sample_rate

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.hier_encoder = hier_encoder
        self.encoder = encoder
        if self.st_mtlalpha < 1.0:
            self.decoder = decoder
        elif decoder is not None:
            logging.warning(
                "Not using decoder because "
                "st_mtlalpha is set as {} (== 1.00)".format(st_mtlalpha),
            )
        self.md_encoder = md_encoder

        self.st_use_transducer_decoder = st_joint_network is not None

        if self.st_use_transducer_decoder:
            from warprnnt_pytorch import RNNTLoss

            self.st_joint_network = st_joint_network
            if tgt_sym_blank in token_list:
                self.blank_id = token_list.index(tgt_sym_blank)
            else:
                # OpenAI Whisper model doesn't <blank> token
                self.blank_id = 0
            self.st_criterion_transducer = RNNTLoss(
                blank=self.blank_id,
                fastemit_lambda=0.0,
            )
        else:
            self.criterion_st = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=tgt_ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

        if self.st_mtlalpha > 0.0:
            self.st_ctc = st_ctc

        self.criterion_asr = LabelSmoothingLoss(
            size=src_vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # submodule for ASR task
        if self.asr_weight > 0:
            assert (
                src_token_list is not None
            ), "Missing src_token_list, cannot add asr module to st model"
            if self.mtlalpha > 0.0:
                self.ctc = ctc
            if self.mtlalpha < 1.0:
                self.extra_asr_decoder = extra_asr_decoder
            elif extra_asr_decoder is not None:
                logging.warning(
                    "Not using extra_asr_decoder because "
                    "mtlalpha is set as {} (== 1.0)".format(mtlalpha),
                )

        # submodule for MT task
        # TODO(brian): this should be deprecated
        if self.mt_weight > 0:
            self.extra_mt_decoder = extra_mt_decoder
            self.extra_mt_encoder = extra_mt_encoder
        elif extra_mt_decoder is not None:
            logging.warning(
                "Not using extra_mt_decoder because "
                "mt_weight is set as {} (== 0)".format(mt_weight),
            )

        # MT error calculator
        if report_bleu:
            self.mt_error_calculator = MTErrorCalculator(
                token_list, tgt_sym_space, tgt_sym_blank, report_bleu
            )
        else:
            self.mt_error_calculator = None

        # ASR error calculator
        if self.asr_weight > 0 and (report_cer or report_wer):
            assert (
                src_token_list is not None
            ), "Missing src_token_list, cannot add asr module to st model"
            self.asr_error_calculator = ASRErrorCalculator(
                src_token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.asr_error_calculator = None

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        self.use_multidecoder = self.md_encoder is not None
        if hasattr(self, "decoder"):
            self.use_speech_attn = getattr(self.decoder, "use_speech_attn", False)
        else:
            self.use_speech_attn = None

        # TODO(jiatong): add multilingual related functions

        if lang_token_id != -1:
            self.lang_token_id = torch.tensor([[lang_token_id]])
        else:
            self.lang_token_id = None

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        src_text: Optional[torch.Tensor] = None,
        src_text_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Frontend + Encoder + Decoder + Calc loss.

        This method processes the input speech and text data through the model's
        frontend, encoder, and decoder, and computes the corresponding losses for
        the tasks (ASR, ST, MT). It performs checks to ensure the input data
        dimensions are consistent and handles optional source text for ASR.

        Args:
            speech: Tensor of shape (Batch, Length, ...) representing the input
                speech data.
            speech_lengths: Tensor of shape (Batch,) representing the lengths of
                the input speech sequences.
            text: Tensor of shape (Batch, Length) representing the target text
                sequences.
            text_lengths: Tensor of shape (Batch,) representing the lengths of
                the target text sequences.
            src_text: Optional; Tensor of shape (Batch, length) representing the
                source text sequences for ASR. Defaults to None.
            src_text_lengths: Optional; Tensor of shape (Batch,) representing the
                lengths of the source text sequences. Defaults to None.
            kwargs: Additional keyword arguments; "utt_id" can be included among
                the input.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
                - loss: The total computed loss for the input batch.
                - stats: A dictionary containing various loss metrics and
                  accuracy statistics.
                - weight: The batch size for normalization purposes.

        Raises:
            AssertionError: If the dimensions of the input tensors do not match
            the expected shapes.

        Examples:
            >>> model = ESPnetSTModel(...)
            >>> speech = torch.randn(16, 16000)  # Example input
            >>> speech_lengths = torch.tensor([16000] * 16)
            >>> text = torch.randint(0, 100, (16, 20))  # Example target text
            >>> text_lengths = torch.tensor([20] * 16)
            >>> loss, stats, weight = model.forward(speech, speech_lengths, text, text_lengths)

        Note:
            Ensure that the input tensors are correctly padded and that the lengths
            provided are accurate to avoid dimension mismatch errors.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        # additional checks with valid src_text
        if src_text is not None:
            assert src_text_lengths.dim() == 1, src_text_lengths.shape
            assert text.shape[0] == src_text.shape[0] == src_text_lengths.shape[0], (
                text.shape,
                src_text.shape,
                src_text_lengths.shape,
            )

        batch_size = speech.shape[0]

        text[text == -1] = self.tgt_ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]
        if src_text is not None:
            src_text = src_text[:, : src_text_lengths.max()]

        # lang id for mbart
        if hasattr(self, "lang_token_id") and self.lang_token_id is not None:
            text = torch.cat(
                [
                    self.lang_token_id.repeat(text.size(0), 1).to(text.device),
                    text,
                ],
                dim=1,
            )
            text_lengths += 1

        # 1. Encoder
        if self.hier_encoder is not None or (
            self.postencoder is not None and self.postencoder.return_int_enc
        ):
            (
                st_encoder_out,
                st_encoder_out_lens,
                asr_encoder_out,
                asr_encoder_out_lens,
            ) = self.encode(speech, speech_lengths, return_int_enc=True)
        else:
            st_encoder_out, st_encoder_out_lens = self.encode(speech, speech_lengths)
            asr_encoder_out, asr_encoder_out_lens = st_encoder_out, st_encoder_out_lens

        # 2a. CTC branch
        if self.asr_weight > 0:
            assert src_text is not None, "missing source text for asr sub-task of ST"

        if self.asr_weight > 0 and self.mtlalpha > 0:
            loss_asr_ctc, cer_asr_ctc = self._calc_asr_ctc_loss(
                asr_encoder_out, asr_encoder_out_lens, src_text, src_text_lengths
            )
        else:
            loss_asr_ctc, cer_asr_ctc = 0.0, None

        if self.st_mtlalpha > 0:
            if self.postencoder is not None and self.postencoder.return_int_enc:
                # run ST CTC without post-encoder downsampling (no hier enc)
                loss_st_ctc, bleu_st_ctc = self._calc_mt_ctc_loss(
                    asr_encoder_out, asr_encoder_out_lens, text, text_lengths
                )
            else:
                loss_st_ctc, bleu_st_ctc = self._calc_mt_ctc_loss(
                    st_encoder_out, st_encoder_out_lens, text, text_lengths
                )
        else:
            loss_st_ctc, bleu_st_ctc = 0.0, None

        # 2b. Attention-decoder branch (extra ASR)
        if self.asr_weight > 0 and self.mtlalpha < 1.0:
            (
                loss_asr_att,
                acc_asr_att,
                cer_asr_att,
                wer_asr_att,
                hs_dec_asr,
            ) = self._calc_asr_att_loss(
                asr_encoder_out,
                asr_encoder_out_lens,
                src_text,
                src_text_lengths,
                self.use_multidecoder,
            )
        else:
            loss_asr_att, acc_asr_att, cer_asr_att, wer_asr_att = 0.0, None, None, None

        # 2c. Attention-decoder branch (extra MT)
        if self.mt_weight > 0:
            mt_encoder_out, mt_encoder_out_lens = self.extra_mt_encoder(
                src_text, src_text_lengths
            )
            loss_mt_att, acc_mt_att, bleu_mt_att = self._calc_mt_att_loss(
                mt_encoder_out,
                mt_encoder_out_lens,
                text,
                text_lengths,
                None,
                None,
                st=False,  # uses same decoder as ST
            )
            # loss_mt_att, acc_mt_att = self._calc_mt_att_loss(
            #     st_encoder_out, st_encoder_out_lens, text, text_lengths, st=False
            # )
        else:
            loss_mt_att, acc_mt_att = 0.0, None

        # 2d. Multi-Decoder encoder
        if self.use_speech_attn:
            speech_out = st_encoder_out
            speech_lens = st_encoder_out_lens
        else:
            speech_out = None
            speech_lens = None

        if self.use_multidecoder:
            dec_asr_lengths = src_text_lengths + 1
            st_encoder_out, st_encoder_out_lens, _ = self.md_encoder(
                hs_dec_asr, dec_asr_lengths
            )

        st_ctc_weight = self.st_mtlalpha
        if st_ctc_weight < 1.0:
            if self.st_use_transducer_decoder:
                # 2e. Transducer decoder branch
                (
                    loss_st_trans,
                    _,
                    _,
                ) = self._calc_st_transducer_loss(
                    st_encoder_out,
                    st_encoder_out_lens,
                    text,
                )

                if st_ctc_weight == 1.0:
                    loss_st = loss_st_ctc
                elif st_ctc_weight == 0.0:
                    loss_st = loss_st_trans
                else:
                    loss_st = (
                        st_ctc_weight * loss_st_ctc
                        + (1 - st_ctc_weight) * loss_st_trans
                    )
                loss_st_att = 0.0
                acc_st_att = None
                bleu_st_att = None
            else:
                # 2e. Attention-decoder branch (ST)
                loss_st_att, acc_st_att, bleu_st_att = self._calc_mt_att_loss(
                    st_encoder_out,
                    st_encoder_out_lens,
                    text,
                    text_lengths,
                    speech_out,
                    speech_lens,
                    st=True,
                )

                if st_ctc_weight == 1.0:
                    loss_st = loss_st_ctc
                elif st_ctc_weight == 0.0:
                    loss_st = loss_st_att
                else:
                    loss_st = (
                        st_ctc_weight * loss_st_ctc + (1 - st_ctc_weight) * loss_st_att
                    )
                loss_st_trans = 0.0
        else:
            loss_st = loss_st_ctc
            loss_st_att = 0.0
            acc_st_att = None
            bleu_st_att = None
            loss_st_trans = 0.0

        # 3. Loss computation
        asr_ctc_weight = self.mtlalpha

        if asr_ctc_weight == 1.0:
            loss_asr = loss_asr_ctc
        elif asr_ctc_weight == 0.0:
            loss_asr = loss_asr_att
        else:
            loss_asr = (
                asr_ctc_weight * loss_asr_ctc + (1 - asr_ctc_weight) * loss_asr_att
            )
        loss_mt = self.mt_weight * loss_mt_att
        loss = (
            (1 - self.asr_weight - self.mt_weight) * loss_st
            + self.asr_weight * loss_asr
            + self.mt_weight * loss_mt
        )

        stats = dict(
            loss=loss.detach(),
            loss_asr=loss_asr.detach() if type(loss_asr) is not float else loss_asr,
            loss_mt=loss_mt.detach() if type(loss_mt) is not float else loss_mt,
            loss_st_ctc=(
                loss_st_ctc.detach() if type(loss_st_ctc) is not float else loss_st_ctc
            ),
            loss_st_trans=(
                loss_st_trans.detach()
                if type(loss_st_trans) is not float
                else loss_st_trans
            ),
            loss_st_att=(
                loss_st_att.detach() if type(loss_st_att) is not float else loss_st_att
            ),
            loss_st=loss_st.detach(),
            acc_asr=acc_asr_att,
            acc_mt=acc_mt_att,
            acc=acc_st_att,
            cer_ctc=cer_asr_ctc,
            cer=cer_asr_att,
            wer=wer_asr_att,
            bleu=bleu_st_att,
            bleu_ctc=bleu_st_ctc,
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
        src_text: Optional[torch.Tensor] = None,
        src_text_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
                Collect features from the input speech data.

        This method extracts features from the given speech tensor and its lengths,
        returning them in a dictionary format. It can be used in various stages of
        the model, including training and evaluation.

        Args:
            speech: A tensor of shape (Batch, Length, ...), representing the input
                speech data.
            speech_lengths: A tensor of shape (Batch,), indicating the lengths of
                each input sequence in the batch.
            text: A tensor of shape (Batch, Length), representing the target text data.
            text_lengths: A tensor of shape (Batch,), indicating the lengths of each
                target sequence in the batch.
            src_text: (Optional) A tensor of shape (Batch, Length) for source text data.
            src_text_lengths: (Optional) A tensor of shape (Batch,) for lengths of
                source text sequences.
            kwargs: Additional keyword arguments for future use.

        Returns:
            A dictionary containing:
                - "feats": A tensor of extracted features.
                - "feats_lengths": A tensor of lengths corresponding to the extracted
                  features.

        Examples:
            >>> model = ESPnetSTModel(...)
            >>> speech_data = torch.randn(2, 16000)  # Example speech tensor
            >>> speech_lengths = torch.tensor([16000, 16000])
            >>> text_data = torch.randint(0, 100, (2, 20))  # Example text tensor
            >>> text_lengths = torch.tensor([20, 20])
            >>> feats = model.collect_feats(speech_data, speech_lengths, text_data, text_lengths)
            >>> print(feats['feats'].shape)  # Output shape of the extracted features

        Note:
            Ensure that the input tensors are correctly shaped and that lengths
            match the batch size. This method relies on the _extract_feats
            method to perform the actual feature extraction.
        """
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        return_int_enc: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Encodes input speech data using a frontend and an encoder.

        This method performs the following steps:
        1. Extract features from the input speech using the frontend.
        2. Apply data augmentation (if specified) during training.
        3. Normalize the features if a normalization method is provided.
        4. Pass the features through the pre-encoder (if specified).
        5. Forward the features through the main encoder.
        6. Optionally pass the encoder output through a hierarchical encoder and/or a
           post-encoder.

        Args:
            speech: A tensor containing the input speech data of shape
                (Batch, Length, ...).
            speech_lengths: A tensor containing the lengths of the input speech
                sequences of shape (Batch,).
            return_int_enc: A boolean indicating whether to return the internal
                encoder output.

        Returns:
            A tuple containing:
                - encoder_out: The output from the encoder of shape
                  (Batch, Length2, Dim2).
                - encoder_out_lens: The lengths of the encoder outputs of shape
                  (Batch,).
                - int_encoder_out: The internal encoder output (only if
                  return_int_enc is True).
                - int_encoder_out_lens: The lengths of the internal encoder output
                  (only if return_int_enc is True).

        Note:
            This method is primarily used by the `st_inference.py` script for
            speech translation.

        Examples:
            >>> model = ESPnetSTModel(...)
            >>> speech_tensor = torch.randn(32, 16000)  # 32 samples, 1 second each
            >>> speech_lengths = torch.tensor([16000] * 32)  # All samples are 1 second
            >>> encoder_out, encoder_out_lens = model.encode(speech_tensor, speech_lengths)
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
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        if return_int_enc:
            int_encoder_out, int_encoder_out_lens = encoder_out, encoder_out_lens

        if self.hier_encoder is not None:
            encoder_out, encoder_out_lens, _ = self.hier_encoder(
                encoder_out, encoder_out_lens
            )

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

        if return_int_enc:
            return encoder_out, encoder_out_lens, int_encoder_out, int_encoder_out_lens
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
        st: bool = True,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(
            ys_pad, self.sos, self.eos, self.tgt_ignore_id
        )
        ys_in_lens = ys_pad_lens + 1
        # 1. Forward decoder
        if st:
            if self.use_speech_attn:
                decoder_out, _ = self.decoder(
                    encoder_out,
                    encoder_out_lens,
                    ys_in_pad,
                    ys_in_lens,
                    speech,
                    speech_lens,
                )
            else:
                decoder_out, _ = self.decoder(
                    encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
                )
        else:
            decoder_out, _ = self.decoder(
                encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
            )

        # 2. Compute attention loss
        loss_att = self.criterion_st(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.tgt_ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.mt_error_calculator is None:
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
        return_hs: bool = False,
    ):
        # Use CTC output as AR decoder target; useful for multi-decoder training
        skip_loss = False
        if self.training and self.ctc_sample_rate > 0:
            if random.uniform(0, 1) < self.ctc_sample_rate:
                ys_hat = self.ctc.argmax(encoder_out).data
                ys_hat = [[x[0] for x in groupby(ys)] for ys in ys_hat]
                ys_hat = [[x for x in filter(lambda x: x != 0, ys)] for ys in ys_hat]
                for i, ys in enumerate(ys_hat):
                    if len(ys) == 0:
                        ys_hat[i] = [x for x in ys_pad[i] if x != -1]
                ys_pad_lens = torch.tensor(
                    [len(x) for x in ys_hat], device=encoder_out.device
                )
                ys_pad = [torch.tensor(ys, device=encoder_out.device) for ys in ys_pad]
                ys_pad = pad_sequence(ys_pad, batch_first=True, padding_value=-1)

                # skip the loss
                skip_loss = True

        ys_in_pad, ys_out_pad = add_sos_eos(
            ys_pad, self.src_sos, self.src_eos, self.ignore_id
        )
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        if return_hs:
            decoder_out, _, hs_dec_asr = self.extra_asr_decoder(
                encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, return_hs=True
            )
        else:
            hs_dec_asr = None
            decoder_out, _ = self.extra_asr_decoder(
                encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
            )

        if skip_loss:
            return 0.0, None, None, None, hs_dec_asr

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

        return loss_att, acc_att, cer_att, wer_att, hs_dec_asr

    def _calc_asr_ctc_loss(
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

    def _calc_mt_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.st_ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        bleu_ctc = None
        if not self.training and self.mt_error_calculator is not None:
            ys_hat = self.st_ctc.argmax(encoder_out).data
            bleu_ctc = self.mt_error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, bleu_ctc

    def _calc_st_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.tgt_ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        joint_out = self.st_joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )

        loss_transducer = self.st_criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
        )

        cer_transducer, wer_transducer = None, None
        # TODO(brian): add error_calculator_trans

        return loss_transducer, cer_transducer, wer_transducer
