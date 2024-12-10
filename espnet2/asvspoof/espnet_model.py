# Copyright 2022 Jiatong Shi (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asvspoof.decoder.abs_decoder import AbsDecoder
from espnet2.asvspoof.loss.abs_loss import AbsASVSpoofLoss
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASVSpoofModel(AbsESPnetModel):
    """
    ASV Spoofing Model for Audio Signal Verification

    This class implements a model for Automatic Speaker Verification (ASV) 
    Spoofing detection. The model processes audio input through a series of 
    components including a frontend, encoder, decoder, and loss calculation 
    mechanisms.

    Attributes:
        preencoder (Optional[AbsPreEncoder]): An optional pre-encoder for raw 
            input data.
        encoder (AbsEncoder): The encoder component that processes features.
        normalize (Optional[AbsNormalize]): An optional normalization layer for 
            feature scaling.
        frontend (Optional[AbsFrontend]): An optional frontend for feature 
            extraction.
        specaug (Optional[AbsSpecAug]): An optional specification augmentation 
            component.
        decoder (AbsDecoder): The decoder component that predicts outcomes 
            based on encoded features.
        losses (Dict[str, AbsASVSpoofLoss]): A dictionary containing various 
            loss functions for training.

    Args:
        frontend (Optional[AbsFrontend]): An optional frontend for feature 
            extraction.
        specaug (Optional[AbsSpecAug]): An optional specification augmentation 
            component.
        normalize (Optional[AbsNormalize]): An optional normalization layer for 
            feature scaling.
        encoder (AbsEncoder): The encoder component that processes features.
        preencoder (Optional[AbsPreEncoder]): An optional pre-encoder for raw 
            input data.
        decoder (AbsDecoder): The decoder component that predicts outcomes 
            based on encoded features.
        losses (Dict[str, AbsASVSpoofLoss]): A dictionary containing various 
            loss functions for training.

    Returns:
        None

    Examples:
        >>> model = ESPnetASVSpoofModel(frontend=None, specaug=None, normalize=None,
        ...                             encoder=my_encoder, preencoder=None,
        ...                             decoder=my_decoder, losses=my_losses)
        >>> speech_tensor = torch.randn(2, 16000)  # Example input
        >>> loss, stats, weight = model.forward(speech_tensor)

    Note:
        Ensure that the input audio tensor is correctly shaped and matches 
        the expected dimensions for processing.

    Todo:
        - Implement additional loss functions as needed.
        - Add support for different frontend configurations.
    """

    @typechecked
    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: AbsEncoder,
        preencoder: Optional[AbsPreEncoder],
        decoder: AbsDecoder,
        losses: Dict[str, AbsASVSpoofLoss],
    ):

        super().__init__()

        self.preencoder = preencoder
        self.encoder = encoder
        self.normalize = normalize
        self.frontend = frontend
        self.specaug = specaug
        self.decoder = decoder
        self.losses = losses

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        label: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Processes input speech through the model's components and computes the 
        loss.

        This method combines the frontend, encoder, and decoder components of the 
        ASV Spoofing model to produce predictions and calculate the associated 
        loss. The output includes the computed loss, statistics, and batch size 
        weight.

        Args:
            speech (torch.Tensor): A tensor containing the speech data with shape 
                (Batch, samples).
            speech_lengths (torch.Tensor, optional): A tensor representing the 
                lengths of each speech sample in the batch. If not provided, 
                defaults to None.
            label (torch.Tensor, optional): A tensor containing the target labels 
                for the speech data with shape (Batch, ). This is used for loss 
                computation.
            **kwargs: Additional keyword arguments, where "utt_id" may be among 
                the inputs.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: A tuple 
            containing:
                - loss (torch.Tensor): The computed loss for the current batch.
                - stats (Dict[str, torch.Tensor]): A dictionary with statistical 
                    metrics, including loss and accuracy.
                - weight (torch.Tensor): The weight of the current batch.

        Raises:
            AssertionError: If the batch size of speech does not match the 
            batch size of the label.

        Examples:
            >>> model = ESPnetASVSpoofModel(...)
            >>> speech_data = torch.randn(32, 16000)  # Example speech data
            >>> labels = torch.randint(0, 2, (32,))  # Example binary labels
            >>> loss, stats, weight = model.forward(speech_data, label=labels)

        Note:
            Ensure that the `label` tensor is provided during training to compute 
            the loss.
        """
        assert speech.shape[0] == label.shape[0], (speech.shape, label.shape)
        batch_size = speech.shape[0]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # 2. Decoder (baiscally a predction layer after encoder_out)
        pred = self.decoder(encoder_out, encoder_out_lens)

        if "oc_softmax_loss" in self.losses:
            loss = (
                self.losses["oc_softmax_loss"](label, encoder_out)
                * self.losses["oc_softmax_loss"].weight
            )
            pred = self.losses["am_softmax_loss"].score(encoder_out)
        elif "am_softmax_loss" in self.losses:
            loss = (
                self.losses["am_softmax_loss"](label, encoder_out)
                * self.losses["am_softmax_loss"].weight
            )
            pred = self.losses["am_softmax_loss"].score(encoder_out)
        else:
            loss = (
                self.losses["binary_loss"](pred, label)
                * self.losses["binary_loss"].weight
            )
        acc = torch.sum(((pred.view(-1) > 0.5) == (label.view(-1) > 0.5))) / batch_size

        stats = dict(
            loss=loss.detach(),
            acc=acc.detach(),
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts features from the input speech tensor.

        This method processes the input speech tensor and its corresponding
        lengths to extract features using the model's frontend. It returns a
        dictionary containing the extracted features and their lengths.

        Args:
            speech: A tensor of shape (Batch, Samples) representing the input
                speech signals.
            speech_lengths: A tensor of shape (Batch,) containing the lengths of
                each speech signal in the batch.
            kwargs: Additional keyword arguments for future extensibility.

        Returns:
            A dictionary with the following keys:
                - 'feats': A tensor containing the extracted features of shape
                  (Batch, NFrames, Dim).
                - 'feats_lengths': A tensor of shape (Batch,) containing the
                  lengths of the extracted features.

        Examples:
            >>> model = ESPnetASVSpoofModel(...)
            >>> speech = torch.randn(32, 16000)  # Batch of 32 audio samples
            >>> speech_lengths = torch.tensor([16000] * 32)  # All samples are 1 sec
            >>> features = model.collect_feats(speech, speech_lengths)
            >>> print(features['feats'].shape)  # Expected output: (32, NFrames, Dim)

        Note:
            Ensure that the input tensors are correctly shaped to avoid
            assertion errors during processing.

        Raises:
            AssertionError: If the input speech lengths are not of the correct
            dimension.
        """
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes the input speech through the frontend and encoder.

        This method extracts features from the input speech using the specified
        frontend and applies data augmentation, normalization, and pre-encoding
        steps if applicable. Finally, it forwards the processed features through
        the encoder to obtain the encoded outputs.

        Args:
            speech: A tensor of shape (Batch, Length, ...), representing the 
                input speech waveforms.
            speech_lengths: A tensor of shape (Batch,), indicating the lengths 
                of the input speech signals.

        Returns:
            A tuple containing:
                - encoder_out: A tensor of shape (Batch, Length2, Dim), where
                  Length2 is the output length after encoding.
                - encoder_out_lens: A tensor of shape (Batch,) that contains 
                  the lengths of the encoded outputs.

        Raises:
            AssertionError: If the output sizes do not match the expected 
            dimensions.

        Examples:
            >>> model = ESPnetASVSpoofModel(...)
            >>> speech = torch.randn(8, 16000)  # 8 samples of 1 second
            >>> speech_lengths = torch.tensor([16000] * 8)
            >>> encoder_out, encoder_out_lens = model.encode(speech, speech_lengths)
            >>> print(encoder_out.shape)  # Should be (8, Length2, Dim)
            >>> print(encoder_out_lens.shape)  # Should be (8,)

        Note:
            This method assumes that the model is initialized with appropriate 
            frontend and encoder components.
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
            # -> encoder_out: (Batch, Length2, Dim)
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

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
        batch_size = speech.shape[0]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else torch.ones(batch_size).int() * speech.shape[1]
        )

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
