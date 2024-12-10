import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import typechecked

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
    """
    ESPnet speech-to-speech translation model.

    This class implements a speech-to-speech translation (S2ST) model that can
    handle various types of input and output features. The model can be
    configured with different frontends, encoders, decoders, and loss functions
    to support diverse speech translation tasks.

    Attributes:
        sos (int): Start-of-sequence token index for target vocabulary.
        eos (int): End-of-sequence token index for target vocabulary.
        src_sos (int): Start-of-sequence token index for source vocabulary.
        src_eos (int): End-of-sequence token index for source vocabulary.
        unit_sos (int): Start-of-sequence token index for unit vocabulary.
        unit_eos (int): End-of-sequence token index for unit vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_vocab_size (int): Size of the source vocabulary.
        unit_vocab_size (int): Size of the unit vocabulary.
        ignore_id (int): Index to ignore during loss computation.
        tgt_token_list (list): List of tokens for target language.
        src_token_list (list): List of tokens for source language.
        unit_token_list (list): List of tokens for unit representation.
        s2st_type (str): Type of the S2ST model (e.g., "translatotron").
        frontend (AbsFrontend): Frontend processing module.
        tgt_feats_extract (AbsTgtFeatsExtract): Target feature extraction module.
        specaug (AbsSpecAug): Spectral augmentation module.
        src_normalize (AbsNormalize): Normalization module for source features.
        tgt_normalize (AbsNormalize): Normalization module for target features.
        preencoder (AbsPreEncoder): Pre-encoder module for raw input data.
        postencoder (AbsPostEncoder): Post-encoder module for additional processing.
        encoder (AbsEncoder): Encoder module for feature extraction.
        asr_decoder (AbsDecoder): ASR decoder module.
        st_decoder (AbsDecoder): ST decoder module.
        aux_attention (AbsS2STAuxAttention): Auxiliary attention mechanism.
        unit_encoder (AbsEncoder): Encoder module for unit representation.
        synthesizer (AbsSynthesizer): Synthesizer module for generating output.
        asr_ctc (CTC): CTC loss module for ASR.
        st_ctc (CTC): CTC loss module for ST.
        losses (dict): Dictionary of loss functions for different tasks.
        extract_feats_in_collect_stats (bool): Flag to indicate feature extraction
            during statistics collection.

    Args:
        s2st_type (str): Type of the S2ST model.
        frontend (Optional[AbsFrontend]): Frontend processing module.
        tgt_feats_extract (Optional[AbsTgtFeatsExtract]): Target feature extraction module.
        specaug (Optional[AbsSpecAug]): Spectral augmentation module.
        src_normalize (Optional[AbsNormalize]): Normalization module for source features.
        tgt_normalize (Optional[AbsNormalize]): Normalization module for target features.
        preencoder (Optional[AbsPreEncoder]): Pre-encoder module for raw input data.
        encoder (AbsEncoder): Encoder module for feature extraction.
        postencoder (Optional[AbsPostEncoder]): Post-encoder module for additional processing.
        asr_decoder (Optional[AbsDecoder]): ASR decoder module.
        st_decoder (Optional[AbsDecoder]): ST decoder module.
        aux_attention (Optional[AbsS2STAuxAttention]): Auxiliary attention mechanism.
        unit_encoder (Optional[AbsEncoder]): Encoder module for unit representation.
        synthesizer (Optional[AbsSynthesizer]): Synthesizer module for generating output.
        asr_ctc (Optional[CTC]): CTC loss module for ASR.
        st_ctc (Optional[CTC]): CTC loss module for ST.
        losses (Dict[str, AbsS2STLoss]): Dictionary of loss functions for different tasks.
        tgt_vocab_size (Optional[int]): Size of the target vocabulary.
        tgt_token_list (Optional[Union[Tuple[str, ...], List[str]]]): List of tokens for
            target language.
        src_vocab_size (Optional[int]): Size of the source vocabulary.
        src_token_list (Optional[Union[Tuple[str, ...], List[str]]]): List of tokens for
            source language.
        unit_vocab_size (Optional[int]): Size of the unit vocabulary.
        unit_token_list (Optional[Union[Tuple[str, ...], List[str]]]): List of tokens for
            unit representation.
        ignore_id (int): Index to ignore during loss computation.
        report_cer (bool): Flag to report character error rate.
        report_wer (bool): Flag to report word error rate.
        report_bleu (bool): Flag to report BLEU score.
        sym_space (str): Symbol representing space.
        sym_blank (str): Symbol representing blank.
        extract_feats_in_collect_stats (bool): Flag to indicate feature extraction
            during statistics collection.

    Raises:
        AssertionError: If certain configurations are not met.

    Examples:
        # Create an instance of the ESPnetS2STModel
        model = ESPnetS2STModel(
            s2st_type="translatotron",
            frontend=my_frontend,
            tgt_feats_extract=my_tgt_feats_extract,
            ...
        )

        # Forward pass through the model
        loss, stats, weight = model(
            src_speech=my_src_speech,
            src_speech_lengths=my_src_speech_lengths,
            tgt_speech=my_tgt_speech,
            tgt_speech_lengths=my_tgt_speech_lengths,
            ...
        )
    """

    @typechecked
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
        unit_encoder: Optional[AbsEncoder],
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

        super().__init__()
        self.sos = tgt_vocab_size - 1 if tgt_vocab_size else None
        self.eos = tgt_vocab_size - 1 if tgt_vocab_size else None
        self.src_sos = src_vocab_size - 1 if src_vocab_size else None
        self.src_eos = src_vocab_size - 1 if src_vocab_size else None
        self.unit_sos = unit_vocab_size - 1 if unit_vocab_size else None
        self.unit_eos = unit_vocab_size - 1 if unit_vocab_size else None
        self.tgt_vocab_size = tgt_vocab_size
        self.src_vocab_size = src_vocab_size
        self.unit_vocab_size = unit_vocab_size
        self.ignore_id = ignore_id
        self.tgt_token_list = tgt_token_list.copy() if tgt_token_list else None
        self.src_token_list = src_token_list.copy() if src_token_list else None
        self.unit_token_list = unit_token_list.copy() if unit_token_list else None
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
        self.unit_encoder = unit_encoder
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
            ), "only support conformer or transformer-based encoder now"

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
        """
        Perform the forward pass of the speech-to-speech translation model.

        This method takes the source and target speech along with optional
        text representations, processes them through the model, and returns
        the computed loss, statistics, and batch size.

        Args:
            src_speech (torch.Tensor): Source speech tensor of shape
                (Batch, Length, ...).
            src_speech_lengths (torch.Tensor): Lengths of source speech
                sequences of shape (Batch,).
            tgt_speech (torch.Tensor): Target speech tensor of shape
                (Batch, Length, ...).
            tgt_speech_lengths (torch.Tensor): Lengths of target speech
                sequences of shape (Batch,).
            tgt_text (Optional[torch.Tensor], optional): Target text tensor
                of shape (Batch, Length, ...). Defaults to None.
            tgt_text_lengths (Optional[torch.Tensor], optional): Lengths of
                target text sequences of shape (Batch,). Defaults to None.
            src_text (Optional[torch.Tensor], optional): Source text tensor
                of shape (Batch, Length, ...). Defaults to None.
            src_text_lengths (Optional[torch.Tensor], optional): Lengths of
                source text sequences of shape (Batch,). Defaults to None.
            spembs (Optional[torch.Tensor], optional): Speaker embeddings
                tensor. Defaults to None.
            sids (Optional[torch.Tensor], optional): Speaker IDs tensor.
                Defaults to None.
            lids (Optional[torch.Tensor], optional): Language IDs tensor.
                Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
            A tuple containing:
                - loss (torch.Tensor): The computed loss for the batch.
                - stats (Dict[str, torch.Tensor]): A dictionary of
                  statistics, including losses and accuracies.
                - weight (torch.Tensor): The batch size for DataParallel
                  compatibility.

        Raises:
            ValueError: If the specified speech lengths do not match
            the batch size or if an unsupported s2st type is encountered.

        Examples:
            >>> model = ESPnetS2STModel(...)
            >>> loss, stats, weight = model.forward(
            ...     src_speech, src_speech_lengths, tgt_speech,
            ...     tgt_speech_lengths, tgt_text, tgt_text_lengths
            ... )

        Note:
            The method includes various checks to ensure that the input
            dimensions are consistent and raises assertions if they are not.
        """
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
                src_speech, src_speech_lengths, return_all_hs=True
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
                src_attn_loss=(
                    src_attn_loss.item() if src_attn_loss is not None else None
                ),
                acc_src_attn=acc_src_attn,
                cer_src_attn=cer_src_attn,
                wer_src_attn=wer_src_attn,
                tgt_attn_loss=(
                    tgt_attn_loss.item() if tgt_attn_loss is not None else None
                ),
                acc_tgt_attn=acc_tgt_attn,
                bleu_tgt_attn=bleu_tgt_attn,
                syn_loss=syn_loss.item() if syn_loss is not None else None,
                syn_guided_attn_loss=(
                    syn_guided_attn_loss.item()
                    if syn_guided_attn_loss is not None
                    else None
                ),
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
                    return_hs=True,
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
                tgt_attn_loss=(
                    tgt_attn_loss.item() if tgt_attn_loss is not None else None
                ),
                acc_tgt_attn=acc_tgt_attn,
                bleu_tgt_attn=bleu_tgt_attn,
                syn_loss=syn_loss.item() if syn_loss is not None else None,
                syn_l1_loss=l1_loss.item() if l1_loss is not None else None,
                syn_mse_loss=mse_loss.item() if mse_loss is not None else None,
                syn_bce_loss=bce_loss.item() if bce_loss is not None else None,
            )

        #########################
        # Discrete unit Forward #
        #########################
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
                (
                    tgt_attn_loss,
                    acc_tgt_attn,
                    bleu_tgt_attn,
                ) = self._calc_st_att_loss(
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
                return_all_hs=True,
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
                src_attn_loss=(
                    src_attn_loss.item() if src_attn_loss is not None else None
                ),
                acc_src_attn=acc_src_attn,
                cer_src_attn=cer_src_attn,
                wer_src_attn=wer_src_attn,
                tgt_attn_loss=(
                    tgt_attn_loss.item() if tgt_attn_loss is not None else None
                ),
                acc_tgt_attn=acc_tgt_attn,
                bleu_tgt_attn=bleu_tgt_attn,
                st_ctc_loss=st_ctc_loss.item() if st_ctc_loss is not None else None,
                cer_st_ctc=cer_st_ctc,
                unit_attn_loss=(
                    unit_attn_loss.item() if unit_attn_loss is not None else None
                ),
                acc_unit_attn=acc_unit_attn if acc_unit_attn is not None else None,
            )

        #################
        # Unity Forward #
        #################
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
                return_hs=True,
            )
            loss_record.append(tgt_attn_loss * self.losses["tgt_attn"].weight)

            assert (
                self.unit_encoder is not None
            ), "unit_encoder is necessary for unity-based model"

            unit_encoder_out, unit_encoder_out_lengths, _ = self.unit_encoder(
                decoder_out, tgt_text_lengths + 1
            )

            # synthesizer
            unit_attn_loss, acc_unit_attn = self._calc_unit_att_loss(
                unit_encoder_out,
                unit_encoder_out_lengths,
                tgt_speech,
                tgt_speech_lengths,
            )
            loss_record.append(unit_attn_loss * self.losses["synthesis"].weight)

            loss = sum(loss_record)

            stats = dict(
                loss=loss.item(),
                asr_ctc_loss=asr_ctc_loss.item() if asr_ctc_loss is not None else None,
                cer_asr_ctc=cer_asr_ctc,
                tgt_attn_loss=(
                    tgt_attn_loss.item() if tgt_attn_loss is not None else None
                ),
                acc_tgt_attn=acc_tgt_attn,
                bleu_tgt_attn=bleu_tgt_attn,
                unit_attn_loss=(
                    unit_attn_loss.item() if unit_attn_loss is not None else None
                ),
                acc_unit_attn=acc_unit_attn if acc_unit_attn is not None else None,
            )

        else:
            raise ValueError("Not supported s2st type {}")

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    @typechecked
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
        """
        Run inference for the speech-to-speech translation model.

        This method takes input speech and generates the corresponding output
        speech features. The method utilizes the encoder and synthesizer to
        produce the output based on the specified model type.

        Args:
            src_speech (torch.Tensor): Input source speech tensor.
            src_speech_lengths (Optional[torch.Tensor]): Lengths of the source
                speech tensor.
            tgt_speech (Optional[torch.Tensor]): Target speech tensor (for
                feature extraction).
            tgt_speech_lengths (Optional[torch.Tensor]): Lengths of the target
                speech tensor.
            spembs (Optional[torch.Tensor]): Speaker embeddings.
            sids (Optional[torch.Tensor]): Speaker IDs.
            lids (Optional[torch.Tensor]): Language IDs.
            threshold (float): Threshold for synthesizer output. Default is 0.5.
            minlenratio (float): Minimum length ratio for output. Default is 0.0.
            maxlenratio (float): Maximum length ratio for output. Default is 10.0.
            use_att_constraint (bool): Flag to use attention constraint.
                Default is False.
            backward_window (int): Number of frames to consider backward. Default
                is 1.
            forward_window (int): Number of frames to consider forward. Default
                is 3.
            use_teacher_forcing (bool): Flag to use teacher forcing during
                inference. Default is False.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing generated features
            and any additional output data, including:
                - 'feat_gen': Generated features.
                - 'feat_gen_denorm': Denormalized generated features if
                  normalization was applied.

        Raises:
            ValueError: If an unsupported s2st type is encountered.

        Examples:
            >>> model.inference(src_speech, src_speech_lengths)
            {
                'feat_gen': <tensor>,
                'feat_gen_denorm': <tensor>
            }
        """

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
        else:
            raise ValueError("Not supported s2st type {}")

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
        """
        Collects features from source and target speech tensors for analysis.

        This method extracts features from the provided source and target speech
        tensors. If `extract_feats_in_collect_stats` is set to `True`, it performs
        feature extraction; otherwise, it generates dummy statistics.

        Attributes:
            extract_feats_in_collect_stats (bool): Determines whether to extract
            features or generate dummy stats.

        Args:
            src_speech (torch.Tensor): Source speech tensor of shape (Batch, Length).
            src_speech_lengths (torch.Tensor): Lengths of source speech tensor of shape (Batch,).
            tgt_speech (torch.Tensor): Target speech tensor of shape (Batch, Length).
            tgt_speech_lengths (torch.Tensor): Lengths of target speech tensor of shape (Batch,).
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing extracted features and
            their lengths. The keys are:
                - "src_feats": Extracted source features.
                - "src_feats_lengths": Lengths of the source features.
                - "tgt_feats": Extracted target features (if applicable).
                - "tgt_feats_lengths": Lengths of the target features (if applicable).

        Examples:
            >>> src_speech = torch.randn(2, 16000)  # Example source speech
            >>> src_lengths = torch.tensor([16000, 15000])  # Example lengths
            >>> tgt_speech = torch.randn(2, 16000)  # Example target speech
            >>> tgt_lengths = torch.tensor([16000, 15000])  # Example lengths
            >>> features = model.collect_feats(src_speech, src_lengths, tgt_speech, tgt_lengths)
            >>> print(features["src_feats"].shape)  # Output: torch.Size([2, N, D])

        Note:
            If `extract_feats_in_collect_stats` is `False`, this method will log a
            warning and return the original speech tensors as dummy statistics.

        Raises:
            AssertionError: If the input tensors do not match the expected dimensions.
        """
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
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        return_all_hs: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Encode the input speech using the frontend and encoder components of the model.

        This method performs several preprocessing steps, including feature extraction,
        data augmentation, and normalization before passing the processed features to the
        encoder. It can return intermediate hidden states if requested.

        Args:
            speech: A tensor of shape (Batch, Length, ...) representing the input speech.
            speech_lengths: A tensor of shape (Batch,) representing the lengths of each
                input sequence in the batch.
            return_all_hs: A boolean indicating whether to return all hidden states from
                the encoder. Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the encoder.

        Returns:
            A tuple containing:
                - encoder_out: A tensor of shape (Batch, Length2, Dim2) representing the
                  encoded output.
                - encoder_out_lens: A tensor of shape (Batch,) representing the lengths
                  of the encoded sequences.
                - inter_encoder_out (optional): Intermediate hidden states from the encoder
                  if return_all_hs is True.

        Examples:
            >>> model = ESPnetS2STModel(...)
            >>> speech = torch.randn(8, 16000)  # Example input (8 samples, 1 second each)
            >>> speech_lengths = torch.tensor([16000] * 8)  # Lengths of the input
            >>> encoder_out, encoder_out_lens = model.encode(speech, speech_lengths)

        Note:
            This method is used by the speech-to-speech translation inference process.
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
        encoder_out, encoder_out_lens, _ = self.encoder(
            feats, feats_lengths, return_all_hs=return_all_hs
        )
        if return_all_hs:
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

        if return_all_hs:
            return (encoder_out, inter_encoder_out), encoder_out_lens
        else:
            return encoder_out, encoder_out_lens

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
        return_hs: bool = False,
        return_all_hs: bool = False,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(
            ys_pad, self.unit_sos, self.unit_eos, self.ignore_id
        )
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        (
            decoder_outs,
            decoder_out_lengths,
        ) = self.synthesizer(
            encoder_out,
            encoder_out_lens,
            ys_in_pad,
            ys_in_lens,
            spembs,
            sids,
            lids,
            return_hs=return_hs,
            return_all_hs=return_all_hs,
        )

        if return_hs or return_all_hs:
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

        if return_hs or return_all_hs:
            return loss_att, acc_att, decoder_hidden, decoder_out_lengths
        else:
            return loss_att, acc_att

    def _calc_st_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        return_hs: bool = False,
        return_all_hs: bool = False,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        assert (
            not return_hs or not return_all_hs
        ), "cannot return both last hiddens or all hiddens"

        # 1. Forward decoder
        (
            decoder_outs,
            decoder_out_lengths,
        ) = self.st_decoder(
            encoder_out,
            encoder_out_lens,
            ys_in_pad,
            ys_in_lens,
            return_hs=return_hs,
            return_all_hs=return_all_hs,
        )

        if return_hs or return_all_hs:
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

        if return_hs:
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
                "Cannot recognize the ctc-type: need 'src'/'tgt', but found {}".format(
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
