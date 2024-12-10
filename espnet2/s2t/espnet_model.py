import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.cuda.amp import autocast
from typeguard import typechecked

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
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import pad_list, th_accuracy
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)


class ESPnetS2TModel(AbsESPnetModel):
    """
    CTC-attention hybrid Encoder-Decoder model for speech-to-text tasks.

    This model integrates Connectionist Temporal Classification (CTC) and attention-based decoding
    to facilitate effective end-to-end speech-to-text conversion. The architecture includes various
    components such as a frontend for feature extraction, an encoder, and a decoder, allowing it to
    handle a variety of speech input formats and enhance transcription accuracy through the use of
    attention mechanisms and CTC.

    Attributes:
        blank_id (int): The index of the blank token in the token list.
        sos (int): The index of the start-of-sequence token.
        eos (int): The index of the end-of-sequence token.
        sop (int): The index of the start-of-previous token.
        na (int): The index of the not-available token.
        vocab_size (int): The size of the vocabulary.
        ignore_id (int): The index used for padding or ignored tokens.
        ctc_weight (float): Weight for CTC loss in the combined loss calculation.
        interctc_weight (float): Weight for intermediate CTC loss in the combined loss calculation.
        token_list (List[str]): The list of tokens used for the model.
        frontend (Optional[AbsFrontend]): Frontend component for feature extraction.
        specaug (Optional[AbsSpecAug]): SpecAugment component for data augmentation.
        normalize (Optional[AbsNormalize]): Normalization component for input features.
        preencoder (Optional[AbsPreEncoder]): Pre-encoder component for raw input data.
        encoder (AbsEncoder): The main encoder component.
        postencoder (Optional[AbsPostEncoder]): Post-encoder component for further processing.
        decoder (Optional[AbsDecoder]): The decoder component for generating outputs.
        ctc (CTC): The CTC loss function used for training.
        criterion_att (LabelSmoothingLoss): The loss function for attention-based decoding.
        error_calculator (Optional[ErrorCalculator]): An optional calculator for error metrics.
        extract_feats_in_collect_stats (bool): Flag to determine if features are extracted during
            statistics collection.

    Args:
        vocab_size (int): Size of the vocabulary.
        token_list (Union[Tuple[str, ...], List[str]]): List of tokens used in the model.
        frontend (Optional[AbsFrontend]): Frontend for feature extraction (default: None).
        specaug (Optional[AbsSpecAug]): Data augmentation component (default: None).
        normalize (Optional[AbsNormalize]): Normalization component (default: None).
        preencoder (Optional[AbsPreEncoder]): Pre-encoder component (default: None).
        encoder (AbsEncoder): Encoder component.
        postencoder (Optional[AbsPostEncoder]): Post-encoder component (default: None).
        decoder (Optional[AbsDecoder]): Decoder component (default: None).
        ctc (CTC): CTC loss function.
        ctc_weight (float): Weight for CTC loss (default: 0.5).
        interctc_weight (float): Weight for intermediate CTC loss (default: 0.0).
        ignore_id (int): Padding index (default: -1).
        lsm_weight (float): Label smoothing weight (default: 0.0).
        length_normalized_loss (bool): Flag for length normalization in loss (default: False).
        report_cer (bool): Flag to report Character Error Rate (default: True).
        report_wer (bool): Flag to report Word Error Rate (default: True).
        sym_space (str): Symbol for space (default: "<space>").
        sym_blank (str): Symbol for blank (default: "<blank>").
        sym_sos (str): Symbol for start-of-sequence (default: "<sos>").
        sym_eos (str): Symbol for end-of-sequence (default: "<eos>").
        sym_sop (str): Symbol for start-of-previous (default: "<sop>").
        sym_na (str): Symbol for not available (default: "<na>").
        extract_feats_in_collect_stats (bool): Flag to extract features during statistics
            collection (default: True).

    Raises:
        AssertionError: If the CTC weights are not in the range [0.0, 1.0].

    Examples:
        >>> model = ESPnetS2TModel(
        ...     vocab_size=5000,
        ...     token_list=["<blank>", "<sos>", "<eos>", "<space>", "<na>"],
        ...     frontend=None,
        ...     specaug=None,
        ...     normalize=None,
        ...     preencoder=None,
        ...     encoder=my_encoder,
        ...     postencoder=None,
        ...     decoder=my_decoder,
        ...     ctc=my_ctc,
        ...     ctc_weight=0.5,
        ...     interctc_weight=0.1,
        ...     ignore_id=-1,
        ...     lsm_weight=0.1,
        ...     length_normalized_loss=True,
        ...     report_cer=True,
        ...     report_wer=True
        ... )
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
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        sym_sos: str = "<sos>",
        sym_eos: str = "<eos>",
        sym_sop: str = "<sop>",  # start of prev
        sym_na: str = "<na>",  # not available
        extract_feats_in_collect_stats: bool = True,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()

        self.blank_id = token_list.index(sym_blank)
        self.sos = token_list.index(sym_sos)
        self.eos = token_list.index(sym_eos)
        self.sop = token_list.index(sym_sop)
        self.na = token_list.index(sym_na)
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.error_calculator = None

        if ctc_weight < 1.0:
            assert (
                decoder is not None
            ), "decoder should not be None when attention is used"
        else:
            decoder = None
            logging.warning("Set decoder to none as ctc_weight==1.0")

        self.decoder = decoder

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        self.is_encoder_whisper = "Whisper" in type(self.encoder).__name__

        if self.is_encoder_whisper:
            assert (
                self.frontend is None
            ), "frontend should be None when using full Whisper model"

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        text_prev: torch.Tensor,
        text_prev_lengths: torch.Tensor,
        text_ctc: torch.Tensor,
        text_ctc_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Process input through the model's components and compute the loss.

        This method orchestrates the flow of input data through the frontend,
        encoder, and decoder, calculating the loss for both CTC and attention-based
        branches as necessary. It handles different types of input, computes
        relevant statistics, and returns the final loss along with statistics.

        Args:
            speech (torch.Tensor): Input speech tensor of shape (Batch, Length, ...).
            speech_lengths (torch.Tensor): Lengths of the speech inputs of shape (Batch,).
            text (torch.Tensor): Input text tensor of shape (Batch, Length).
            text_lengths (torch.Tensor): Lengths of the text inputs of shape (Batch,).
            text_prev (torch.Tensor): Previous text inputs for attention mechanism of shape (Batch, Length).
            text_prev_lengths (torch.Tensor): Lengths of the previous text inputs of shape (Batch,).
            text_ctc (torch.Tensor): CTC-targeted text tensor of shape (Batch, Length).
            text_ctc_lengths (torch.Tensor): Lengths of the CTC-targeted text inputs of shape (Batch,).
            kwargs: Additional keyword arguments, expected to include "utt_id".

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: A tuple containing:
                - loss (torch.Tensor): The computed loss for the current batch.
                - stats (Dict[str, torch.Tensor]): A dictionary containing various statistics:
                    - loss_ctc: CTC loss.
                    - cer_ctc: Character Error Rate for CTC.
                    - loss_att: Attention loss.
                    - acc: Accuracy for the attention mechanism.
                    - cer: Character Error Rate for the attention mechanism.
                    - wer: Word Error Rate for the attention mechanism.
                    - loss: Total computed loss.
                - weight (torch.Tensor): The batch size for loss normalization.

        Raises:
            AssertionError: If the dimensions of the input tensors do not match.

        Examples:
            >>> model = ESPnetS2TModel(...)
            >>> speech = torch.randn(4, 16000)  # 4 samples of 1 second audio
            >>> speech_lengths = torch.tensor([16000, 16000, 16000, 16000])
            >>> text = torch.randint(0, 100, (4, 20))  # Random text tensor
            >>> text_lengths = torch.tensor([20, 20, 20, 20])
            >>> text_prev = torch.randint(0, 100, (4, 20))
            >>> text_prev_lengths = torch.tensor([20, 20, 20, 20])
            >>> text_ctc = torch.randint(0, 100, (4, 20))
            >>> text_ctc_lengths = torch.tensor([20, 20, 20, 20])
            >>> loss, stats, weight = model.forward(speech, speech_lengths, text, text_lengths,
            ...                                      text_prev, text_prev_lengths,
            ...                                      text_ctc, text_ctc_lengths)

        Note:
            This method is typically called during the training loop, where it is
            essential to compute both the forward pass and the associated loss for
            model optimization.

        Todo:
            - Extend support for additional loss metrics in the future.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
            == text_prev.shape[0]
            == text_prev_lengths.shape[0]
            == text_ctc.shape[0]
            == text_ctc_lengths.shape[0]
        ), (
            speech.shape,
            speech_lengths.shape,
            text.shape,
            text_lengths.shape,
            text_prev.shape,
            text_prev_lengths.shape,
            text_ctc.shape,
            text_ctc_lengths.shape,
        )
        batch_size = speech.shape[0]

        # -1 is used as padding index in collate fn
        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text_ctc, text_ctc_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text_ctc, text_ctc_lengths
                )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        # 2. Attention decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out,
                encoder_out_lens,
                text,
                text_lengths,
                text_prev,
                text_prev_lengths,
            )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        text_prev: torch.Tensor,
        text_prev_lengths: torch.Tensor,
        text_ctc: torch.Tensor,
        text_ctc_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from the input speech tensor.

        This method processes the input speech tensor and returns the extracted features
        along with their lengths. It is typically used during the training or evaluation
        phases of the model to collect the features for further processing.

        Args:
            speech (torch.Tensor): A tensor of shape (Batch, Length, ...) representing
                the input speech signals.
            speech_lengths (torch.Tensor): A tensor of shape (Batch,) indicating the
                lengths of each input speech signal.
            text (torch.Tensor): A tensor of shape (Batch, Length) containing the target
                text sequences.
            text_lengths (torch.Tensor): A tensor of shape (Batch,) indicating the lengths
                of each target text sequence.
            text_prev (torch.Tensor): A tensor of shape (Batch, Length) containing the
                previous text sequences.
            text_prev_lengths (torch.Tensor): A tensor of shape (Batch,) indicating the
                lengths of each previous text sequence.
            text_ctc (torch.Tensor): A tensor of shape (Batch, Length) representing the
                CTC target text sequences.
            text_ctc_lengths (torch.Tensor): A tensor of shape (Batch,) indicating the
                lengths of each CTC target text sequence.
            **kwargs: Additional keyword arguments that may be needed for other processing.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - "feats": A tensor of extracted features of shape (Batch, NFrames, Dim).
                - "feats_lengths": A tensor of lengths for the extracted features of shape
                  (Batch,).

        Examples:
            >>> model = ESPnetS2TModel(...)
            >>> speech_tensor = torch.randn(32, 16000)  # Example input tensor for 32 signals
            >>> speech_lengths = torch.tensor([16000] * 32)  # All signals have the same length
            >>> text_tensor = torch.randint(0, 100, (32, 20))  # Example target text tensor
            >>> text_lengths = torch.tensor([20] * 32)  # All texts have the same length
            >>> features = model.collect_feats(speech_tensor, speech_lengths, text_tensor,
            ...                                  text_lengths, text_tensor, text_lengths,
            ...                                  text_tensor, text_lengths)
            >>> print(features["feats"].shape)  # Should output the shape of extracted features

        Note:
            The method assumes that the `frontend` is set up properly to handle the
            feature extraction from the raw speech input.

        Raises:
            AssertionError: If the input dimensions do not match or if there are issues
            with the speech lengths.
        """
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes the input speech through the frontend and encoder.

        This method is primarily responsible for extracting features from
        the raw speech input and then passing those features through the
        encoder to produce encoded outputs. This function is also used
        during inference in `s2t_inference.py`.

        Args:
            speech: A tensor of shape (Batch, Length, ...) representing
                the input speech waveforms.
            speech_lengths: A tensor of shape (Batch,) indicating the
                lengths of each input sequence in the batch.

        Returns:
            A tuple containing:
                - encoder_out: A tensor of shape (Batch, Length2, Dim2)
                  representing the output of the encoder.
                - encoder_out_lens: A tensor of shape (Batch,)
                  representing the lengths of the encoder outputs.

        Note:
            This method incorporates optional data augmentation,
            normalization, and pre-encoding steps, depending on the
            model configuration.

        Examples:
            >>> model = ESPnetS2TModel(...)
            >>> speech = torch.randn(2, 16000)  # Example batch of 2 audio signals
            >>> speech_lengths = torch.tensor([16000, 15000])  # Lengths of each audio
            >>> encoder_out, encoder_out_lens = model.encode(speech, speech_lengths)
            >>> print(encoder_out.shape)  # Output shape will depend on encoder configuration
            >>> print(encoder_out_lens)  # Lengths of encoder outputs
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
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        if (
            getattr(self.encoder, "selfattention_layer_type", None) != "lf_selfattn"
            and not self.is_encoder_whisper
        ):
            assert encoder_out.size(-2) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

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

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        ys_prev_pad: torch.Tensor,
        ys_prev_lens: torch.Tensor,
    ):
        # 0. Prepare input and output with sos, eos, sop
        ys = [y[y != self.ignore_id] for y in ys_pad]
        ys_prev = [y[y != self.ignore_id] for y in ys_prev_pad]

        _sos = ys_pad.new([self.sos])
        _eos = ys_pad.new([self.eos])
        _sop = ys_pad.new([self.sop])
        ys_in = []
        ys_in_lens = []
        ys_out = []
        for y_prev, y in zip(ys_prev, ys):
            if self.na in y_prev:
                # Prev is not available in this case
                y_in = [_sos, y]
                y_in_len = len(y) + 1
                y_out = [y, _eos]
            else:
                y_in = [_sop, y_prev, _sos, y]
                y_in_len = len(y_prev) + len(y) + 2
                y_out = [self.ignore_id * ys_pad.new_ones(len(y_prev) + 1), y, _eos]

            ys_in.append(torch.cat(y_in))
            ys_in_lens.append(y_in_len)
            ys_out.append(torch.cat(y_out))

        ys_in_pad = pad_list(ys_in, self.eos)
        ys_in_lens = torch.tensor(ys_in_lens).to(ys_pad_lens)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_out_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Filter out invalid samples where text is not available
        is_valid = [self.na not in y for y in ys_pad]
        if not any(is_valid):
            return torch.tensor(0.0), None

        encoder_out = encoder_out[is_valid]
        encoder_out_lens = encoder_out_lens[is_valid]
        ys_pad = ys_pad[is_valid]
        ys_pad_lens = ys_pad_lens[is_valid]

        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc
