#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import typechecked

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


class TorchAudioHubertPretrainModel(AbsESPnetModel):
    """
    TorchAudio Hubert Pretrain model.

    This model implements the HuBERT pretraining for audio representations,
    utilizing a combination of frontend processing, data augmentation, and
    normalization techniques. It inherits from the `AbsESPnetModel` class.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        ignore_id (int): ID to ignore in the loss calculation.
        token_list (List[str]): List of tokens used in the model.
        frontend (AbsFrontend): Frontend for audio feature extraction.
        specaug (AbsSpecAug): SpecAugment for data augmentation.
        normalize (AbsNormalize): Normalization layer.
        preencoder (AbsPreEncoder): Pre-encoder for raw input data.
        encoder (AbsEncoder): Main encoder for processing features.
        error_calculator (Optional[ErrorCalculator]): Error calculation utility.
        nan_loss_count (float): Counter for NaN losses encountered.

    Args:
        vocab_size (int): Size of the vocabulary.
        token_list (Union[Tuple[str, ...], List[str]]): List of tokens.
        frontend (Optional[AbsFrontend]): Frontend module.
        specaug (Optional[AbsSpecAug]): SpecAugment module.
        normalize (Optional[AbsNormalize]): Normalization module.
        preencoder (Optional[AbsPreEncoder]): Pre-encoder module.
        encoder (AbsEncoder): Encoder module.
        ignore_id (int, optional): ID to ignore (default: -1).
        lsm_weight (float, optional): Label smoothing weight (default: 0.0).
        length_normalized_loss (bool, optional): Whether to use length-normalized loss (default: False).
        report_cer (bool, optional): Whether to report Character Error Rate (default: False).
        report_wer (bool, optional): Whether to report Word Error Rate (default: False).
        sym_space (str, optional): Symbol for space (default: "<space>").
        sym_blank (str, optional): Symbol for blank (default: "<blank>").
        pred_masked_weight (float, optional): Weight for masked prediction (default: 1.0).
        pred_nomask_weight (float, optional): Weight for non-masked prediction (default: 0.0).
        loss_weights (float, optional): Additional weights for loss calculation (default: 0.0).
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
            A tuple containing the loss tensor, statistics dictionary,
            and weight tensor.

    Examples:
        model = TorchAudioHubertPretrainModel(vocab_size=100, token_list=["<pad>", "<sos>", "<eos>"],
                                               frontend=my_frontend, encoder=my_encoder)
        loss, stats, weight = model(speech_tensor, speech_lengths_tensor, text_tensor, text_lengths_tensor)

    Note:
        This model is based on the work by Abdelrahman Mohamed and Wei-Ning Hsu,
        detailed in the paper: https://arxiv.org/pdf/2106.07447.pdf.

    Raises:
        AssertionError: If input tensor dimensions do not match expected shapes.
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
        **kwargs,
    ):

        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        self.error_calculator = None

        self.nan_loss_count = 0.0

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Frontend + Encoder + Calc loss

        This method processes input speech and text data through the model's
        frontend and encoder components, computes the loss, and returns the
        results along with accuracy statistics. It ensures that the input
        dimensions are consistent and handles data-parallelism for batch
        processing.

        Args:
            speech: A tensor of shape (Batch, Length, ...) representing the
                input speech data.
            speech_lengths: A tensor of shape (Batch,) containing the lengths
                of each speech sample in the batch.
            text: A tensor of shape (Batch, Length) representing the input text
                data.
            text_lengths: A tensor of shape (Batch,) containing the lengths of
                each text sample in the batch.
            kwargs: Additional keyword arguments, which may include "utt_id".

        Returns:
            A tuple containing:
                - loss (torch.Tensor): The computed loss value.
                - stats (Dict[str, torch.Tensor]): A dictionary with
                  statistics including accuracy metrics.
                - weight (torch.Tensor): A tensor representing the batch size
                  for DataParallel handling.

        Raises:
            AssertionError: If the dimensions of input tensors do not match.

        Examples:
            >>> model = TorchAudioHubertPretrainModel(...)
            >>> speech_tensor = torch.randn(4, 16000)  # Batch of 4, 1 second audio
            >>> speech_lengths = torch.tensor([16000, 16000, 16000, 16000])
            >>> text_tensor = torch.randint(0, 100, (4, 20))  # Batch of 4, text
            >>> text_lengths = torch.tensor([20, 20, 20, 20])
            >>> loss, stats, weight = model.forward(speech_tensor,
                                                    speech_lengths,
                                                    text_tensor,
                                                    text_lengths)
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
        logit_m, logit_u, feature_penalty = self.encode(
            speech, speech_lengths, text, text_lengths
        )

        # 2a. Hubert criterion
        loss = self._calc_hubert_loss(
            logit_m,
            logit_u,
            feature_penalty,
        )

        if not torch.isinf(loss) and not torch.isnan(loss):
            pass
            # logging.warning(f"loss, {loss.item() / logit_m.size(0)}")
        else:
            self.nan_loss_count += 1
            logging.warning(f"nan_loss_count, {self.nan_loss_count}")

        # log accuracies of masked and unmasked frames
        correct_m, count_m = self._compute_correct(logit_m)
        correct_u, count_u = self._compute_correct(logit_u)

        stats = dict(
            loss=loss.detach(),
            correct_m=correct_m,
            count_m=count_m,
            acc_m=correct_m / count_m,
            correct_u=correct_u,
            count_u=count_u,
            acc_u=correct_u / count_u,
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
        """
        Collect features from the input speech tensor and its lengths.

        This method extracts features from the provided speech data using the
        frontend defined in the model. It returns a dictionary containing the
        extracted features and their corresponding lengths.

        Args:
            speech (torch.Tensor): The input speech tensor of shape
                (Batch, Length, ...).
            speech_lengths (torch.Tensor): A tensor containing the lengths of
                each speech input in the batch of shape (Batch,).
            text (torch.Tensor): A tensor containing the text data of shape
                (Batch, Length).
            text_lengths (torch.Tensor): A tensor containing the lengths of
                each text input in the batch of shape (Batch,).
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'feats': Extracted features of shape (Batch, NFrames, Dim).
                - 'feats_lengths': Lengths of the extracted features of shape
                  (Batch,).

        Examples:
            >>> model = HubertPretrainModel(...)
            >>> speech = torch.randn(10, 16000)  # 10 samples of 1 second audio
            >>> speech_lengths = torch.tensor([16000] * 10)  # lengths for each sample
            >>> text = torch.randint(0, 100, (10, 20))  # random text tensor
            >>> text_lengths = torch.tensor([20] * 10)  # lengths for each text
            >>> features = model.collect_feats(speech, speech_lengths, text, text_lengths)
            >>> print(features['feats'].shape)  # Output: torch.Size([10, NFrames, Dim])
            >>> print(features['feats_lengths'])  # Output: lengths of features
        """
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        y_pad: torch.Tensor,
        y_pad_length: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Frontend + Encoder. Note that this method is used by asr_inference.py

        This method processes the input speech data through the frontend and
        encoder to produce encoded features.

        Args:
            speech: A tensor of shape (Batch, Length, ...) representing the input
                speech signals.
            speech_lengths: A tensor of shape (Batch,) containing the lengths of
                each speech signal in the batch.
            y_pad: A tensor of shape (Batch, Length, ...) representing the padded
                target sequences.
            y_pad_length: A tensor of shape (Batch,) containing the lengths of
                each padded target sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - encoder_out: The output from the encoder, a tensor of shape
                  (Batch, Length2, Dim2).
                - feats: The extracted features after passing through the frontend.

        Note:
            This method is typically called during the forward pass of the model
            to obtain encoded representations of the input speech data.
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

    def _compute_correct(
        self,
        logits,
    ):
        if logits.numel() == 0:
            corr, count = 0, 0
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
        logit_m: Optional[torch.Tensor],
        logit_u: Optional[torch.Tensor],
        feature_penalty: torch.Tensor,
        masked_weight: float = 1.0,
        unmasked_weight: float = 0.0,
        feature_weight: float = 10.0,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute the cross-entropy loss on HuBERT masked and non-masked logits.

        Args:
            logit_m (Tensor or None): The masked logit Tensor of dimension
                `(masked_frames, final_dim)`.
            logit_u (Tensor or None): The non-masked logit Tensor of dimension
                `(unmasked_frames, final_dim)`.
            feature_penalty (Tensor): The feature mean value for additional penalty
                loss.
            masked_weight (float, optional): The weight for masked cross-entropy loss
                (Default: ``1.0``).
            unmasked_weight (float, optional): The weight for non-masked cross-entropy
                loss (Default: ``0.0``).
            feature_weight (float, optional): The weight for feature penalty loss
                (Default: ``10.0``).
            reduction (str, optional): The reduction method for cross-entropy loss
                (Default: ``"sum"``).
        Ref:
            torchaudio: examples/hubert/loss/hubert_loss.py
        """
        loss = feature_penalty * feature_weight * logit_m.shape[0]
        if logit_m is not None:
            target_m = torch.zeros(
                logit_m.shape[0], dtype=torch.long, device=logit_m.device
            )
            loss_m = torch.nn.functional.cross_entropy(
                logit_m, target_m, reduction=reduction
            )
            loss += loss_m * masked_weight
        if logit_u is not None:
            target_u = torch.zeros(
                logit_u.shape[0], dtype=torch.long, device=logit_m.device
            )
            loss_u = torch.nn.functional.cross_entropy(
                logit_u, target_u, reduction=reduction
            )
            loss += loss_u * unmasked_weight
        return loss


class HubertPretrainModel(AbsESPnetModel):
    """
        HubertPretrainModel is a model class for pre-training HuBERT (Hidden Unit
    BERT) using self-supervised learning techniques.

    This model takes speech input and associated text input, processes them
    through a series of layers, and computes the loss based on the predictions
    made. It is designed for training with masked and unmasked tokens to
    improve the model's understanding of speech.

    Attributes:
        sos (int): Start of sequence token ID.
        eos (int): End of sequence token ID.
        vocab_size (int): Size of the vocabulary.
        ignore_id (int): Token ID to ignore during loss computation.
        token_list (list): List of tokens corresponding to the vocabulary.
        frontend (AbsFrontend): Frontend processing module.
        specaug (AbsSpecAug): SpecAugment module for data augmentation.
        normalize (AbsNormalize): Normalization module.
        preencoder (AbsPreEncoder): Pre-encoder module for raw input data.
        encoder (AbsEncoder): Main encoder module.
        criterion_hubert (HubertPretrainLoss): Loss computation module for HuBERT.
        pred_masked_weight (float): Weight for masked predictions in loss.
        pred_nomask_weight (float): Weight for unmasked predictions in loss.
        loss_weights (float): Additional loss weights for training.
        error_calculator (ErrorCalculator): Optional error calculator for
            evaluation metrics.

    Args:
        vocab_size (int): Size of the vocabulary.
        token_list (Union[Tuple[str, ...], List[str]]): List of tokens.
        frontend (Optional[AbsFrontend]): Frontend processing module.
        specaug (Optional[AbsSpecAug]): SpecAugment module for data augmentation.
        normalize (Optional[AbsNormalize]): Normalization module.
        preencoder (Optional[AbsPreEncoder]): Pre-encoder module for raw input data.
        encoder (AbsEncoder): Main encoder module.
        ignore_id (int): Token ID to ignore during loss computation (default: -1).
        lsm_weight (float): Label smoothing weight (default: 0.0).
        length_normalized_loss (bool): Whether to normalize loss by length (default: False).
        report_cer (bool): Whether to report character error rate (default: False).
        report_wer (bool): Whether to report word error rate (default: False).
        sym_space (str): Token representing space (default: "<space>").
        sym_blank (str): Token representing blank (default: "<blank>").
        pred_masked_weight (float): Weight for masked predictions in loss (default: 1.0).
        pred_nomask_weight (float): Weight for unmasked predictions in loss (default: 0.0).
        loss_weights (float): Additional loss weights for training (default: 0.0).
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        A tuple containing:
            - loss (torch.Tensor): Computed loss value.
            - stats (Dict[str, torch.Tensor]): Statistics of the model including
              accuracy metrics.
            - weight (torch.Tensor): Weight tensor for DataParallel.

    Examples:
        model = HubertPretrainModel(vocab_size=5000, token_list=["<blank>", "<space>", ...])
        speech_tensor = torch.randn(32, 16000)  # Example batch of speech data
        speech_lengths = torch.tensor([16000] * 32)  # Example lengths
        text_tensor = torch.randint(0, 5000, (32, 100))  # Example text data
        text_lengths = torch.tensor([100] * 32)  # Example lengths

        loss, stats, weight = model(speech_tensor, speech_lengths, text_tensor, text_lengths)

    Note:
        This model is built upon the ESPnet framework and requires appropriate
        backend components such as frontend, encoder, and normalization layers
        to function correctly.

    Todo:
        - Add support for more advanced loss calculations.
        - Implement more robust error handling for input data.
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
        pred_masked_weight: float = 1.0,
        pred_nomask_weight: float = 0.0,
        loss_weights: float = 0.0,
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
        self.criterion_hubert = HubertPretrainLoss(
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
        """
        Frontend + Encoder + Calc loss

        This method processes the input speech and text data through the
        frontend and encoder components, computes the loss, and returns
        the relevant statistics.

        Args:
            speech: A tensor of shape (Batch, Length, ...) representing the input
                speech data.
            speech_lengths: A tensor of shape (Batch,) representing the lengths of
                each speech sample in the batch.
            text: A tensor of shape (Batch, Length) representing the input text data.
            text_lengths: A tensor of shape (Batch,) representing the lengths of
                each text sample in the batch.
            kwargs: Additional keyword arguments, where "utt_id" is among the inputs.

        Returns:
            A tuple containing:
                - loss: A tensor representing the computed loss.
                - stats: A dictionary with statistics including accuracy.
                - weight: A tensor representing the weight for data-parallel
                  processing.

        Raises:
            AssertionError: If the dimensions of the input tensors do not match.

        Examples:
            >>> model = HubertPretrainModel(...)
            >>> speech = torch.randn(4, 16000)  # Example speech tensor
            >>> speech_lengths = torch.tensor([16000, 16000, 16000, 16000])
            >>> text = torch.randint(0, 100, (4, 20))  # Example text tensor
            >>> text_lengths = torch.tensor([20, 20, 20, 20])
            >>> loss, stats, weight = model.forward(speech, speech_lengths, text,
            ...                                       text_lengths)
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
        """
        Extract features from speech input.

        This method takes the speech input and its corresponding lengths,
        extracts the features using the model's frontend, and returns
        the features along with their lengths in a dictionary.

        Args:
            speech: A tensor of shape (Batch, Length, ...) representing the
                speech signals.
            speech_lengths: A tensor of shape (Batch,) containing the lengths
                of the speech signals.
            text: A tensor of shape (Batch, Length) representing the text
                input (not used in this method).
            text_lengths: A tensor of shape (Batch,) containing the lengths
                of the text input (not used in this method).
            kwargs: Additional keyword arguments.

        Returns:
            A dictionary containing:
                - 'feats': The extracted features as a tensor.
                - 'feats_lengths': The lengths of the extracted features as a tensor.

        Examples:
            >>> model = HubertPretrainModel(...)
            >>> speech = torch.randn(2, 16000)  # Example speech tensor
            >>> speech_lengths = torch.tensor([16000, 12000])  # Lengths
            >>> text = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Example text
            >>> text_lengths = torch.tensor([3, 3])  # Lengths
            >>> features = model.collect_feats(speech, speech_lengths, text, text_lengths)
            >>> print(features['feats'].shape)  # Output shape of features
            >>> print(features['feats_lengths'])  # Output lengths of features
        """
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
        """
            Computes the number of correct predictions from logits.

        This method evaluates the logits to determine the number of correct
        predictions based on the argmax and argmin criteria. It calculates
        the count of correct predictions while ensuring that both max and
        min predictions do not contribute to the correct count simultaneously.

        Args:
            logits (torch.Tensor): A tensor containing the logits for which
                to compute correct predictions. The tensor must have at least
                two dimensions.

        Returns:
            Tuple[int, int]: A tuple containing:
                - corr (int): The number of correct predictions.
                - count (int): The total number of predictions evaluated.

        Examples:
            >>> import torch
            >>> logits = torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
            >>> correct, total = compute_correct(logits)
            >>> print(correct, total)
            (2, 3)  # Example output may vary based on the content of logits.
        """
        if logits.numel() == 0:
            corr, count = 0, 0
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
        # 1. Compute hubert loss
        loss, logp_m_list, logp_u_list = self.criterion_hubert(
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

        acc_m = corr_masked / (count_masked + 1e-10)
        acc_u = corr_unmask / (count_unmask + 1e-10)

        return loss, acc_m, acc_u
