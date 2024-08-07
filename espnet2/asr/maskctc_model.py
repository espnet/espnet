import logging
from contextlib import contextmanager
from itertools import groupby
from typing import Dict, List, Optional, Tuple, Union

import numpy
import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.mlm_decoder import MLMDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet.nets.beam_search import Hypothesis
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.maskctc.add_mask_token import mask_uniform
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
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


class MaskCTCModel(ESPnetASRModel):
    """
        Hybrid CTC/Masked LM Encoder-Decoder model (Mask-CTC).

    This class implements the Mask-CTC model, which combines Connectionist Temporal
    Classification (CTC) and Masked Language Model (MLM) approaches for automatic
    speech recognition (ASR).

    The model consists of an encoder, a CTC branch, and an MLM decoder branch. It
    supports various components such as frontend processing, spectrogram
    augmentation, normalization, pre-encoding, post-encoding, and joint network
    functionalities.

    Attributes:
        vocab_size (int): Size of the vocabulary, including the mask token.
        token_list (List[str]): List of tokens in the vocabulary.
        mask_token (int): Token ID for the mask token.
        criterion_mlm (LabelSmoothingLoss): Loss function for the MLM branch.
        error_calculator (ErrorCalculator): Calculator for CER and WER metrics.

    Args:
        vocab_size (int): Size of the vocabulary.
        token_list (Union[Tuple[str, ...], List[str]]): List of tokens in the vocabulary.
        frontend (Optional[AbsFrontend]): Frontend processing module.
        specaug (Optional[AbsSpecAug]): Spectrogram augmentation module.
        normalize (Optional[AbsNormalize]): Normalization module.
        preencoder (Optional[AbsPreEncoder]): Pre-encoder module.
        encoder (AbsEncoder): Encoder module.
        postencoder (Optional[AbsPostEncoder]): Post-encoder module.
        decoder (MLMDecoder): Masked Language Model decoder.
        ctc (CTC): Connectionist Temporal Classification module.
        joint_network (Optional[torch.nn.Module]): Joint network module.
        ctc_weight (float): Weight for the CTC loss (default: 0.5).
        interctc_weight (float): Weight for intermediate CTC loss (default: 0.0).
        ignore_id (int): ID to be ignored in loss computation (default: -1).
        lsm_weight (float): Label smoothing weight (default: 0.0).
        length_normalized_loss (bool): Whether to normalize loss by length (default: False).
        report_cer (bool): Whether to report Character Error Rate (default: True).
        report_wer (bool): Whether to report Word Error Rate (default: True).
        sym_space (str): Space symbol (default: "<space>").
        sym_blank (str): Blank symbol (default: "<blank>").
        sym_mask (str): Mask symbol (default: "<mask>").
        extract_feats_in_collect_stats (bool): Whether to extract features in collect_stats (default: True).

    Note:
        This model extends the ESPnetASRModel and modifies it to incorporate
        the Mask-CTC approach, which combines CTC and MLM for improved ASR performance.
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
        decoder: MLMDecoder,
        ctc: CTC,
        joint_network: Optional[torch.nn.Module] = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        sym_mask: str = "<mask>",
        extract_feats_in_collect_stats: bool = True,
    ):

        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            ctc_weight=ctc_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
        )

        # Add <mask> and override inherited fields
        token_list.append(sym_mask)
        vocab_size += 1
        self.vocab_size = vocab_size
        self.mask_token = vocab_size - 1
        self.token_list = token_list.copy()

        # MLM loss
        del self.criterion_att
        self.criterion_mlm = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.error_calculator = None
        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
                Forward pass for the Mask-CTC model.

        This method performs the forward pass through the entire model, including
        frontend processing, encoding, CTC loss calculation, and MLM loss calculation.

        Args:
            speech (torch.Tensor): Input speech tensor of shape (Batch, Length, ...).
            speech_lengths (torch.Tensor): Tensor of input speech lengths of shape (Batch,).
            text (torch.Tensor): Target text tensor of shape (Batch, Length).
            text_lengths (torch.Tensor): Tensor of target text lengths of shape (Batch,).
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: A tuple containing:
                - loss (torch.Tensor): The total loss for the forward pass.
                - stats (Dict[str, torch.Tensor]): A dictionary of statistics including
                  CTC loss, MLM loss, accuracies, and error rates.
                - weight (torch.Tensor): The batch size as a weight for the loss.

        Raises:
            AssertionError: If the input tensors have inconsistent batch sizes or
                            if text_lengths is not a 1D tensor.

        Note:
            This method calculates both CTC and MLM losses, combining them based on
            the specified CTC weight. It also handles intermediate CTC loss if enabled.

        Examples:
            >>> speech = torch.randn(2, 1000, 80)
            >>> speech_lengths = torch.tensor([1000, 800])
            >>> text = torch.randint(0, 100, (2, 20))
            >>> text_lengths = torch.tensor([20, 15])
            >>> model = MaskCTCModel(...)
            >>> loss, stats, weight = model.forward(speech, speech_lengths, text, text_lengths)
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

        # For data-parallel
        text = text[:, : text_lengths.max()]

        # Define stats to report
        loss_mlm, acc_mlm = None, None
        loss_ctc, cer_ctc = None, None
        stats = dict()

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # 2. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # 2a. Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text, text_lengths
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

        # 3. MLM decoder branch
        if self.ctc_weight != 1.0:
            loss_mlm, acc_mlm = self._calc_mlm_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 4. CTC/MLM loss definition
        if self.ctc_weight == 0.0:
            loss = loss_mlm
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_mlm

        # Collect MLM branch stats
        stats["loss_mlm"] = loss_mlm.detach() if loss_mlm is not None else None
        stats["acc_mlm"] = acc_mlm

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _calc_mlm_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # 1. Apply masks
        ys_in_pad, ys_out_pad = mask_uniform(
            ys_pad, self.mask_token, self.eos, self.ignore_id
        )

        # 2. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_pad_lens
        )

        # 3. Compute mlm loss
        loss_mlm = self.criterion_mlm(decoder_out, ys_out_pad)
        acc_mlm = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        return loss_mlm, acc_mlm

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
                Calculate negative log-likelihood for the Mask-CTC model.

        This method is not implemented for the Mask-CTC model and will raise a
        NotImplementedError if called.

        Args:
            encoder_out (torch.Tensor): Output from the encoder.
            encoder_out_lens (torch.Tensor): Lengths of encoder outputs.
            ys_pad (torch.Tensor): Padded target sequences.
            ys_pad_lens (torch.Tensor): Lengths of target sequences.

        Returns:
            torch.Tensor: This method does not return a value.

        Raises:
            NotImplementedError: This method is not implemented for the Mask-CTC model.

        Note:
            The negative log-likelihood calculation is not applicable to the Mask-CTC
            model due to its hybrid nature combining CTC and MLM approaches.
        """
        raise NotImplementedError

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """
                Batchify the calculation of negative log-likelihood for the Mask-CTC model.

        This method is not implemented for the Mask-CTC model and will raise a
        NotImplementedError if called.

        Args:
            encoder_out (torch.Tensor): Output from the encoder.
            encoder_out_lens (torch.Tensor): Lengths of encoder outputs.
            ys_pad (torch.Tensor): Padded target sequences.
            ys_pad_lens (torch.Tensor): Lengths of target sequences.
            batch_size (int): Size of each batch for processing (default: 100).

        Returns:
            This method does not return a value.

        Raises:
            NotImplementedError: This method is not implemented for the Mask-CTC model.

        Note:
            The batchified negative log-likelihood calculation is not applicable to the
            Mask-CTC model due to its hybrid nature combining CTC and MLM approaches.
            This method is included for compatibility with other ASR model interfaces
            but is not functional in the Mask-CTC context.
        """
        raise NotImplementedError


class MaskCTCInference(torch.nn.Module):
    """
        Mask-CTC-based non-autoregressive inference for automatic speech recognition.

    This class implements the inference process for the Mask-CTC model, which combines
    Connectionist Temporal Classification (CTC) and Masked Language Model (MLM) approaches
    for non-autoregressive speech recognition.

    The inference process involves iterative decoding, where masked tokens are
    progressively predicted based on CTC probabilities and MLM predictions.

    Attributes:
        ctc (CTC): The CTC module from the ASR model.
        mlm (MLMDecoder): The MLM decoder from the ASR model.
        mask_token (int): The token ID representing the mask.
        n_iterations (int): Number of iterations for the decoding process.
        threshold_probability (float): Probability threshold for masking tokens.
        converter (TokenIDConverter): Converter for token IDs to text.

    Args:
        asr_model (MaskCTCModel): The trained Mask-CTC ASR model.
        n_iterations (int): Number of iterations for the decoding process.
        threshold_probability (float): Probability threshold for masking tokens.

    Note:
        This class is designed to work with the MaskCTCModel and provides a
        non-autoregressive inference method that can potentially be faster than
        traditional autoregressive decoding approaches.
    """

    def __init__(
        self,
        asr_model: MaskCTCModel,
        n_iterations: int,
        threshold_probability: float,
    ):
        """Initialize Mask-CTC inference"""
        super().__init__()
        self.ctc = asr_model.ctc
        self.mlm = asr_model.decoder
        self.mask_token = asr_model.mask_token
        self.n_iterations = n_iterations
        self.threshold_probability = threshold_probability
        self.converter = TokenIDConverter(token_list=asr_model.token_list)

    def ids2text(self, ids: List[int]):
        """
                Convert a list of token IDs to readable text.

        This method converts a sequence of token IDs to a human-readable string,
        replacing special tokens with their corresponding symbols.

        Args:
            ids (List[int]): A list of token IDs to be converted to text.

        Returns:
            str: The converted text string.

        Note:
            - The method replaces "<mask>" with "_" and "<space>" with a space character.
            - This conversion is useful for visualizing the output during the inference process.

        Example:
            >>> inference = MaskCTCInference(...)
            >>> ids = [1, 2, 3, 4, 5]  # Assuming these are valid token IDs
            >>> text = inference.ids2text(ids)
            >>> print(text)
            'converted text'
        """
        text = "".join(self.converter.ids2tokens(ids))
        return text.replace("<mask>", "_").replace("<space>", " ")

    def forward(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """
                Perform Mask-CTC inference on the given encoder output.

        This method implements the non-autoregressive Mask-CTC inference algorithm,
        which iteratively refines the output by predicting masked tokens.

        Args:
            enc_out (torch.Tensor): The encoder output tensor of shape (T, D),
                where T is the sequence length and D is the feature dimension.

        Returns:
            List[Hypothesis]: A list containing a single Hypothesis object with the
                predicted token sequence.

        Note:
            The inference process involves the following steps:
            1. Generate initial CTC greedy outputs
            2. Mask low-confidence tokens based on CTC probabilities
            3. Iteratively predict masked tokens using the MLM decoder
            4. Finalize the output by predicting any remaining masked tokens

            The method logs intermediate results for debugging purposes.

        Example:
            >>> inference = MaskCTCInference(...)
            >>> encoder_output = torch.randn(100, 256)  # (T, D)
            >>> hypothesis = inference(encoder_output)
            >>> predicted_tokens = hypothesis[0].yseq
        """
        # greedy ctc outputs
        enc_out = enc_out.unsqueeze(0)
        ctc_probs, ctc_ids = torch.exp(self.ctc.log_softmax(enc_out)).max(dim=-1)
        y_hat = torch.stack([x[0] for x in groupby(ctc_ids[0])])
        y_idx = torch.nonzero(y_hat != 0).squeeze(-1)

        logging.info("ctc:{}".format(self.ids2text(y_hat[y_idx].tolist())))

        # calculate token-level ctc probabilities by taking
        # the maximum probability of consecutive frames with
        # the same ctc symbols
        probs_hat = []
        cnt = 0
        for i, y in enumerate(y_hat.tolist()):
            probs_hat.append(-1)
            while cnt < ctc_ids.shape[1] and y == ctc_ids[0][cnt]:
                if probs_hat[i] < ctc_probs[0][cnt]:
                    probs_hat[i] = ctc_probs[0][cnt].item()
                cnt += 1
        probs_hat = torch.from_numpy(numpy.array(probs_hat)).to(enc_out.device)

        # mask ctc outputs based on ctc probabilities
        p_thres = self.threshold_probability
        mask_idx = torch.nonzero(probs_hat[y_idx] < p_thres).squeeze(-1)
        confident_idx = torch.nonzero(probs_hat[y_idx] >= p_thres).squeeze(-1)
        mask_num = len(mask_idx)

        y_in = (
            torch.zeros(1, len(y_idx), dtype=torch.long).to(enc_out.device)
            + self.mask_token
        )
        y_in[0][confident_idx] = y_hat[y_idx][confident_idx]

        logging.info("msk:{}".format(self.ids2text(y_in[0].tolist())))

        # iterative decoding
        if not mask_num == 0:
            K = self.n_iterations
            num_iter = K if mask_num >= K and K > 0 else mask_num

            for t in range(num_iter - 1):
                pred, _ = self.mlm(enc_out, [enc_out.size(1)], y_in, [y_in.size(1)])
                pred_score, pred_id = pred[0][mask_idx].max(dim=-1)
                cand = torch.topk(pred_score, mask_num // num_iter, -1)[1]
                y_in[0][mask_idx[cand]] = pred_id[cand]
                mask_idx = torch.nonzero(y_in[0] == self.mask_token).squeeze(-1)

                logging.info("msk:{}".format(self.ids2text(y_in[0].tolist())))

            # predict leftover masks (|masks| < mask_num // num_iter)
            pred, _ = self.mlm(enc_out, [enc_out.size(1)], y_in, [y_in.size(1)])
            y_in[0][mask_idx] = pred[0][mask_idx].argmax(dim=-1)

            logging.info("msk:{}".format(self.ids2text(y_in[0].tolist())))

        # pad with mask tokens to ensure compatibility with sos/eos tokens
        yseq = torch.tensor(
            [self.mask_token] + y_in.tolist()[0] + [self.mask_token], device=y_in.device
        )

        return Hypothesis(yseq=yseq)
