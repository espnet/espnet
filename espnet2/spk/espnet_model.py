# Copyright 2023 Jee-weon Jung
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Dict, Optional, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.spk.loss.abs_loss import AbsLoss
from espnet2.spk.pooling.abs_pooling import AbsPooling
from espnet2.spk.projector.abs_projector import AbsProjector
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


class ESPnetSpeakerModel(AbsESPnetModel):
    """
    Speaker embedding extraction model.

    Core model for diverse speaker-related tasks (e.g., verification, open-set
    identification, diarization).

    The model architecture comprises mainly 'encoder', 'pooling', and
    'projector'. In common speaker recognition field, the combination of three
    would usually be named as 'speaker_encoder' (or speaker embedding extractor).
    We split it into three for flexibility in future extensions:
      - 'encoder'   : Extracts frame-level speaker embeddings.
      - 'pooling'   : Aggregates into single utterance-level embedding.
      - 'projector' : (Optional) Additional processing (e.g., one fully-
                      connected layer) to derive speaker embedding.

    Possibly, in the future, 'pooling' and/or 'projector' can be integrated as
    a 'decoder', depending on the extension for joint usage of different tasks
    (e.g., ASR, SE, target speaker extraction).

    Attributes:
        frontend (Optional[AbsFrontend]): The frontend processing module.
        specaug (Optional[AbsSpecAug]): The spec augmentation module.
        normalize (Optional[AbsNormalize]): The normalization module.
        encoder (Optional[AbsEncoder]): The encoder module.
        pooling (Optional[AbsPooling]): The pooling module.
        projector (Optional[AbsProjector]): The projector module.
        loss (Optional[AbsLoss]): The loss function used during training.

    Args:
        frontend: Frontend processing module.
        specaug: Spec augmentation module.
        normalize: Normalization module.
        encoder: Encoder module.
        pooling: Pooling module.
        projector: Projector module.
        loss: Loss function.

    Examples:
        >>> model = ESPnetSpeakerModel(frontend=None, specaug=None,
        ...                             normalize=None, encoder=my_encoder,
        ...                             pooling=my_pooling, projector=my_projector,
        ...                             loss=my_loss)
        >>> speech = torch.randn(10, 16000)  # Batch of 10 audio samples
        >>> spk_labels = torch.randint(0, 10, (10,))  # Random speaker labels
        >>> loss, stats, weight = model.forward(speech, spk_labels=spk_labels)

    Note:
        Ensure that the appropriate modules are provided to the constructor for
        correct functioning.

    Raises:
        AssertionError: If the dimensions of input tensors do not match
        expected shapes.
    """

    @typechecked
    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: Optional[AbsEncoder],
        pooling: Optional[AbsPooling],
        projector: Optional[AbsProjector],
        loss: Optional[AbsLoss],
    ):

        super().__init__()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.pooling = pooling
        self.projector = projector
        self.loss = loss

    @typechecked
    def forward(
        self,
        speech: torch.Tensor,
        spk_labels: Optional[torch.Tensor] = None,
        task_tokens: Optional[torch.Tensor] = None,
        extract_embd: bool = False,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor], torch.Tensor
    ]:
        """
        Feed-forward through encoder layers and aggregate into utterance-level
        feature.

        This method processes the input speech tensor through the model's
        components, extracting frame-level features, aggregating them into a
        single utterance-level feature, and optionally computing the loss based
        on provided speaker labels. If `extract_embd` is set to True, it returns
        the speaker embedding directly without computing the loss.

        Args:
            speech: A tensor of shape (Batch, samples) representing the input
                speech signals.
            spk_labels: A tensor of shape (Batch,) representing one-hot speaker
                labels used during training. If provided, the loss will be
                calculated.
            task_tokens: A tensor of shape (Batch,) used for token-based
                training, indicating the task for each input.
            extract_embd: A boolean flag indicating whether to return the
                speaker embedding directly without going through the classification
                head. Defaults to False.
            **kwargs: Additional keyword arguments for future extensions.

        Returns:
            If `extract_embd` is True, returns the speaker embedding tensor.
            Otherwise, returns a tuple containing:
                - loss: A tensor representing the computed loss.
                - stats: A dictionary containing statistics (e.g., loss).
                - weight: The batch size.

        Raises:
            AssertionError: If `spk_labels` is provided but its shape does not
            match the batch size of `speech`.
            AssertionError: If `task_tokens` is provided but its shape does not
            match the batch size of `speech`.
            AssertionError: If `spk_labels` is None when calculating the loss.

        Examples:
            >>> model = ESPnetSpeakerModel(...)
            >>> speech_input = torch.randn(32, 16000)  # 32 samples of 1 second
            >>> speaker_labels = torch.randint(0, 10, (32,))  # Random labels
            >>> loss, stats, weight = model.forward(speech_input, speaker_labels)
            >>> spk_embd = model.forward(speech_input, extract_embd=True)

        Note:
            This method is designed to be called during both training and
            inference, with behavior changing based on the provided arguments.
        """
        if spk_labels is not None:
            assert speech.shape[0] == spk_labels.shape[0], (
                speech.shape,
                spk_labels.shape,
            )
        if task_tokens is not None:
            assert speech.shape[0] == task_tokens.shape[0], (
                speech.shape,
                task_tokens.shape,
            )
        batch_size = speech.shape[0]

        # 1. extract low-level feats (e.g., mel-spectrogram or MFCC)
        # Will do nothing for raw waveform-based models (e.g., RawNets)
        feats, _ = self.extract_feats(speech, None)

        frame_level_feats = self.encode_frame(feats)

        # 2. aggregation into utterance-level
        utt_level_feat = self.pooling(frame_level_feats, task_tokens)

        # 3. (optionally) go through further projection(s)
        spk_embd = self.project_spk_embd(utt_level_feat)

        if extract_embd:
            return spk_embd

        # 4. calculate loss
        assert spk_labels is not None, "spk_labels is None, cannot compute loss"
        loss = self.loss(spk_embd, spk_labels.squeeze())

        stats = dict(loss=loss.detach())

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Extract features from input speech tensor.

        This method processes the input speech signal to extract features using
        the defined frontend, applies any specified augmentations, and normalizes
        the resulting features. It is a critical step in the speaker embedding
        extraction process.

        Args:
            speech: A tensor of shape (Batch, Samples) representing the input
                speech waveforms.
            speech_lengths: A tensor of shape (Batch,) indicating the lengths of
                each speech signal in the batch. If None, it assumes that all
                signals are of equal length.

        Returns:
            A tuple containing:
                - feats: A tensor of extracted features.
                - feat_lengths: A tensor indicating the lengths of the extracted
                  features for each sample in the batch. Returns None if the
                  frontend is not defined.

        Examples:
            >>> model = ESPnetSpeakerModel(...)
            >>> speech_tensor = torch.randn(8, 16000)  # 8 samples of 1 second audio
            >>> lengths = torch.tensor([16000] * 8)    # All samples are 1 second long
            >>> feats, feat_lengths = model.extract_feats(speech_tensor, lengths)

        Note:
            The method first checks if a frontend is defined. If so, it will
            use it to extract features; otherwise, it will return the raw
            speech signal as features. Augmentations and normalization are only
            applied if the model is in training mode and the respective modules
            are defined.
        """
        batch_size = speech.shape[0]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else torch.ones(batch_size).int() * speech.shape[1]
        )

        # 1. extract feats
        if self.frontend is not None:
            feats, feat_lengths = self.frontend(speech, speech_lengths)
        else:
            feats = speech
            feat_lengths = None

        # 2. apply augmentations
        if self.specaug is not None and self.training:
            feats, _ = self.specaug(feats, feat_lengths)

        # 3. normalize
        if self.normalize is not None:
            feats, _ = self.normalize(feats, feat_lengths)

        return feats, feat_lengths

    def encode_frame(self, feats: torch.Tensor) -> torch.Tensor:
        """
                Encode frame-level features from the input features using the encoder.

        This method processes the input features to extract frame-level speaker
        embeddings. It utilizes the encoder component of the model to achieve this.

        Args:
            feats: A tensor of shape (Batch, Features, Time) representing the input
                   features from which frame-level embeddings are to be extracted.

        Returns:
            A tensor of shape (Batch, Frame_Features, Time) containing the
            frame-level speaker embeddings extracted from the input features.

        Examples:
            >>> model = ESPnetSpeakerModel(...)
            >>> input_feats = torch.randn(32, 40, 100)  # Batch of 32, 40 features, 100 time steps
            >>> frame_level_feats = model.encode_frame(input_feats)
            >>> print(frame_level_feats.shape)  # Output shape: (32, Frame_Features, 100)

        Note:
            Ensure that the input tensor is properly shaped according to the model's
            requirements for the encoder.
        """
        frame_level_feats = self.encoder(feats)

        return frame_level_feats

    def aggregate(self, frame_level_feats: torch.Tensor) -> torch.Tensor:
        """
            Aggregate frame-level features into utterance-level features.

        This method processes a batch of frame-level features, aggregating them
        to produce a single utterance-level feature representation. It uses the
        configured aggregator to perform this operation.

        Args:
            frame_level_feats: A tensor of shape (Batch, Frame, Features) that
                contains the frame-level features to be aggregated.

        Returns:
            A tensor of shape (Batch, Features) representing the aggregated
            utterance-level features.

        Examples:
            >>> model = ESPnetSpeakerModel(...)
            >>> frame_level_feats = torch.randn(32, 100, 64)  # Batch of 32
            >>> utt_level_feat = model.aggregate(frame_level_feats)
            >>> print(utt_level_feat.shape)  # Should output: torch.Size([32, Features])
        """
        utt_level_feat = self.aggregator(frame_level_feats)

        return utt_level_feat

    def project_spk_embd(self, utt_level_feat: torch.Tensor) -> torch.Tensor:
        """
            Speaker embedding extraction model.

        Core model for diverse speaker-related tasks (e.g., verification, open-set
        identification, diarization).

        The model architecture comprises mainly 'encoder', 'pooling', and
        'projector'. In the common speaker recognition field, the combination of
        these three components is usually referred to as 'speaker_encoder' or
        'speaker embedding extractor'. We have separated them into three distinct
        components for flexibility in future extensions:

          - 'encoder'   : Extracts frame-level speaker embeddings.
          - 'pooling'   : Aggregates frame-level embeddings into a single
                          utterance-level embedding.
          - 'projector' : (Optional) Additional processing (e.g., a fully-
                          connected layer) to derive the final speaker embedding.

        In the future, 'pooling' and/or 'projector' may be integrated as a
        'decoder', depending on the extensions for joint usage of different tasks
        (e.g., ASR, SE, target speaker extraction).

        Attributes:
            frontend (Optional[AbsFrontend]): The frontend component for feature
                extraction.
            specaug (Optional[AbsSpecAug]): The spec augmentation component.
            normalize (Optional[AbsNormalize]): The normalization component.
            encoder (Optional[AbsEncoder]): The encoder component for extracting
                embeddings.
            pooling (Optional[AbsPooling]): The pooling component for aggregating
                embeddings.
            projector (Optional[AbsProjector]): The projector component for further
                processing.
            loss (Optional[AbsLoss]): The loss function for training.

        Args:
            frontend (Optional[AbsFrontend]): Frontend for feature extraction.
            specaug (Optional[AbsSpecAug]): Spec augmentation module.
            normalize (Optional[AbsNormalize]): Normalization module.
            encoder (Optional[AbsEncoder]): Encoder module.
            pooling (Optional[AbsPooling]): Pooling module.
            projector (Optional[AbsProjector]): Projector module.
            loss (Optional[AbsLoss]): Loss module.

        Examples:
            >>> model = ESPnetSpeakerModel(frontend=None, specaug=None,
            ...                             normalize=None, encoder=encoder,
            ...                             pooling=pooling, projector=projector,
            ...                             loss=loss)
            >>> output = model.forward(speech_tensor, spk_labels_tensor)

        Note:
            Ensure that the input tensors have the correct dimensions.

        Todo:
            Extend functionality for joint usage of different tasks.
        """
        if self.projector is not None:
            spk_embd = self.projector(utt_level_feat)
        else:
            spk_embd = utt_level_feat

        return spk_embd

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spk_labels: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Collects features from the input speech tensor.

        This method extracts the features from the input speech signal and returns
        them in a dictionary format. It leverages the `extract_feats` method to
        process the speech input, applying any necessary transformations such as
        augmentation and normalization.

        Args:
            speech: A tensor containing the speech data of shape (Batch, samples).
            speech_lengths: A tensor indicating the lengths of each speech sample
                in the batch of shape (Batch,).
            spk_labels: (Optional) A tensor containing one-hot encoded speaker
                labels used for training, of shape (Batch,).

        Returns:
            A dictionary containing the extracted features:
                - "feats": The processed feature tensor.

        Examples:
            >>> model = ESPnetSpeakerModel(...)
            >>> speech_tensor = torch.randn(2, 16000)  # Example batch of speech
            >>> lengths = torch.tensor([16000, 16000])  # Example lengths
            >>> features = model.collect_feats(speech_tensor, lengths)
            >>> print(features["feats"].shape)  # Check the shape of extracted features
        """
        feats, feats_lengths = self.extract_feats(speech, speech_lengths)
        return {"feats": feats}
