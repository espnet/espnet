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

    This model combines Connectionist Temporal Classification (CTC) and
    Masked Language Modeling (MLM) to perform automatic speech recognition
    tasks. It utilizes an encoder-decoder architecture where the encoder
    processes the input speech, and the decoder predicts the output tokens
    using both CTC and MLM loss functions.

    Attributes:
        vocab_size (int): The size of the vocabulary including the mask token.
        token_list (List[str]): A list of tokens corresponding to the vocabulary.
        mask_token (int): The index of the mask token in the vocabulary.
        criterion_mlm (LabelSmoothingLoss): The loss function used for MLM.
        error_calculator (Optional[ErrorCalculator]): Object to calculate error metrics.

    Args:
        vocab_size (int): Size of the vocabulary.
        token_list (Union[Tuple[str, ...], List[str]]): List of tokens.
        frontend (Optional[AbsFrontend]): Frontend module for feature extraction.
        specaug (Optional[AbsSpecAug]): SpecAugment module for data augmentation.
        normalize (Optional[AbsNormalize]): Normalization layer.
        preencoder (Optional[AbsPreEncoder]): Pre-encoder module.
        encoder (AbsEncoder): Encoder module.
        postencoder (Optional[AbsPostEncoder]): Post-encoder module.
        decoder (MLMDecoder): Decoder module for MLM.
        ctc (CTC): CTC module.
        joint_network (Optional[torch.nn.Module]): Joint network module, if any.
        ctc_weight (float): Weight for CTC loss (default: 0.5).
        interctc_weight (float): Weight for intermediate CTC loss (default: 0.0).
        ignore_id (int): ID to ignore during loss calculation (default: -1).
        lsm_weight (float): Label smoothing weight (default: 0.0).
        length_normalized_loss (bool): If True, normalize loss by length (default: False).
        report_cer (bool): If True, report Character Error Rate (default: True).
        report_wer (bool): If True, report Word Error Rate (default: True).
        sym_space (str): Token representing space (default: "<space>").
        sym_blank (str): Token representing blank (default: "<blank>").
        sym_mask (str): Token representing mask (default: "<mask>").
        extract_feats_in_collect_stats (bool): If True, extract features during
            statistics collection (default: True).

    Examples:
        >>> model = MaskCTCModel(
        ...     vocab_size=100,
        ...     token_list=["<blank>", "<space>", "<mask>"] + ["a", "b", "c"],
        ...     frontend=None,
        ...     specaug=None,
        ...     normalize=None,
        ...     preencoder=None,
        ...     encoder=SomeEncoder(),
        ...     postencoder=None,
        ...     decoder=SomeMLMDecoder(),
        ...     ctc=SomeCTC(),
        ...     ctc_weight=0.5,
        ...     interctc_weight=0.0,
        ...     ignore_id=-1,
        ...     lsm_weight=0.1,
        ...     length_normalized_loss=False,
        ...     report_cer=True,
        ...     report_wer=True,
        ...     sym_space="<space>",
        ...     sym_blank="<blank>",
        ...     sym_mask="<mask>",
        ...     extract_feats_in_collect_stats=True
        ... )

    Note:
        This model is designed for tasks where both CTC and MLM are beneficial,
        such as in noisy speech recognition or when the input data is limited.

    Raises:
        AssertionError: If the dimensions of the input tensors do not match.
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
        Process input through the model's frontend, encoder, and decoder, and
        compute the associated loss.

        This method takes speech and text input, processes them through the
        model's architecture, and calculates the CTC and MLM losses. The output
        includes the total loss, statistics for loss and accuracy, and the
        batch size.

        Args:
            speech (torch.Tensor): Input speech tensor of shape
                (Batch, Length, ...).
            speech_lengths (torch.Tensor): Lengths of the input speech tensor
                of shape (Batch,).
            text (torch.Tensor): Input text tensor of shape
                (Batch, Length).
            text_lengths (torch.Tensor): Lengths of the input text tensor
                of shape (Batch,).

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
                - Total loss for the batch.
                - A dictionary containing statistics such as CTC loss,
                  MLM loss, and accuracies.
                - Batch size for data-parallel processing.

        Raises:
            AssertionError: If the dimensions of input tensors do not match.

        Examples:
            >>> model = MaskCTCModel(...)
            >>> speech = torch.randn(16, 100, 40)  # Batch of 16, 100 time steps, 40 features
            >>> speech_lengths = torch.tensor([100] * 16)
            >>> text = torch.randint(0, 50, (16, 20))  # Batch of 16, 20 tokens
            >>> text_lengths = torch.tensor([20] * 16)
            >>> loss, stats, batch_size = model.forward(speech, speech_lengths, text, text_lengths)

        Note:
            This function assumes that the input speech and text tensors
            are properly preprocessed and padded to the same batch size.

        Todo:
            - Improve error handling for mismatched tensor shapes.
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
        Computes the negative log-likelihood (NLL) for the MaskCTCModel.

        This method is intended to be implemented in subclasses to provide
        the functionality for calculating the NLL based on the encoder outputs,
        lengths, and the target sequences. This function is currently not
        implemented and raises a NotImplementedError.

        Args:
            encoder_out (torch.Tensor): The output from the encoder, typically of
                shape (Batch, Length, Features).
            encoder_out_lens (torch.Tensor): The lengths of the encoder outputs,
                of shape (Batch,).
            ys_pad (torch.Tensor): The padded target sequences, of shape
                (Batch, Length).
            ys_pad_lens (torch.Tensor): The lengths of the target sequences,
                of shape (Batch,).

        Returns:
            torch.Tensor: The negative log-likelihood value for the given
            encoder outputs and target sequences.

        Raises:
            NotImplementedError: This method is not implemented in the base class.

        Examples:
            >>> model = MaskCTCModel(...)  # Initialize your model
            >>> encoder_output = torch.rand(32, 100, 256)  # Example encoder output
            >>> encoder_output_lengths = torch.randint(1, 100, (32,))
            >>> target_sequences = torch.randint(0, model.vocab_size, (32, 50))
            >>> target_lengths = torch.randint(1, 50, (32,))
            >>> nll_value = model.nll(encoder_output, encoder_output_lengths,
            ...                        target_sequences, target_lengths)
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
        Batchify the negative log likelihood (NLL) computation.

        This method takes the encoded outputs and targets, and splits them into
        smaller batches for the computation of the negative log likelihood. This
        is useful for efficiently processing large datasets that may not fit
        into memory all at once.

        Args:
            encoder_out (torch.Tensor): The output from the encoder, shaped
                (Batch, Length, Features).
            encoder_out_lens (torch.Tensor): The lengths of each sequence in
                the encoder output, shaped (Batch,).
            ys_pad (torch.Tensor): The target sequences, shaped (Batch, Length).
            ys_pad_lens (torch.Tensor): The lengths of each target sequence,
                shaped (Batch,).
            batch_size (int, optional): The size of each batch to process.
                Defaults to 100.

        Returns:
            torch.Tensor: A tensor containing the computed negative log
            likelihoods for each batch.

        Raises:
            NotImplementedError: This method is not yet implemented.

        Examples:
            >>> encoder_out = torch.randn(200, 50, 256)  # Example encoder output
            >>> encoder_out_lens = torch.randint(1, 50, (200,))
            >>> ys_pad = torch.randint(0, 100, (200, 30))  # Example target
            >>> ys_pad_lens = torch.randint(1, 30, (200,))
            >>> nll_values = model.batchify_nll(encoder_out, encoder_out_lens,
            ...                                   ys_pad, ys_pad_lens, batch_size=50)
        """
        raise NotImplementedError


class MaskCTCInference(torch.nn.Module):
    """
    Mask-CTC-based non-autoregressive inference.

    This class implements a non-autoregressive inference method for the
    Mask-CTC model. It utilizes the CTC probabilities and a masked language
    model to iteratively predict masked tokens in the output sequence.
    The inference process leverages a greedy CTC decoding followed by a
    series of updates to refine the predictions for masked tokens.

    Attributes:
        ctc: The CTC module of the ASR model.
        mlm: The masked language model (MLM) decoder.
        mask_token: The token ID used for masking in the output sequence.
        n_iterations: The number of iterations for iterative decoding.
        threshold_probability: The probability threshold for masking tokens.
        converter: A TokenIDConverter for converting token IDs to text.

    Args:
        asr_model (MaskCTCModel): The Mask-CTC model used for inference.
        n_iterations (int): The number of iterations for iterative decoding.
        threshold_probability (float): The threshold probability for masking
            tokens during inference.

    Examples:
        >>> model = MaskCTCModel(...)
        >>> inference = MaskCTCInference(model, n_iterations=5,
                                          threshold_probability=0.5)
        >>> enc_out = torch.randn(1, 10, model.vocab_size)  # Example encoder output
        >>> hypotheses = inference(enc_out)
        >>> print(hypotheses[0].yseq)  # Output the predicted sequence

    Note:
        This implementation requires that the CTC output be in log probabilities.

    Raises:
        ValueError: If `n_iterations` or `threshold_probability` are not
            positive values.
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
        Convert a list of token IDs to a human-readable text string.

        This method takes a list of token IDs, converts them to their corresponding
        tokens using the TokenIDConverter, and formats the output by replacing
        special tokens with more human-readable representations. Specifically,
        it replaces the "<mask>" token with an underscore ("_") and the "<space>"
        token with a space (" ").

        Args:
            ids (List[int]): A list of token IDs to be converted to text.

        Returns:
            str: A human-readable string representation of the input token IDs.

        Examples:
            >>> inference = MaskCTCInference(...)
            >>> token_ids = [1, 2, 3, 4, 5]
            >>> text = inference.ids2text(token_ids)
            >>> print(text)
            "Token1 Token2 Token3 _ Token5"

        Note:
            Ensure that the input list of IDs corresponds to the correct token
            mapping as defined in the TokenIDConverter.
        """
        text = "".join(self.converter.ids2tokens(ids))
        return text.replace("<mask>", "_").replace("<space>", " ")

    def forward(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """
        Perform Mask-CTC inference.

        This method executes the Mask-CTC inference process using the given
        encoded outputs from the speech recognition model. It performs greedy
        decoding with the CTC outputs and iteratively refines the predictions
        for masked tokens using the masked language model (MLM) decoder.

        Args:
            enc_out: A tensor of shape (1, Length, ...) containing the encoded
                      outputs from the CTC model. The first dimension is
                      artificially added for batch compatibility.

        Returns:
            A list of Hypothesis objects representing the predicted sequences
            from the inference process.

        Examples:
            >>> enc_out = torch.randn(1, 100, 256)  # Example encoded output
            >>> inference_model = MaskCTCInference(asr_model, n_iterations=5,
            ...                                      threshold_probability=0.5)
            >>> hypotheses = inference_model(enc_out)
            >>> print(hypotheses)

        Note:
            The inference process involves applying a threshold on the CTC
            probabilities to determine which tokens to mask and iteratively
            filling in these masked tokens using the MLM decoder.

        Todo:
            - Optimize the masking and inference process for better performance.
            - Implement additional evaluation metrics for the inference results.
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
