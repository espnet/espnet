# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from contextlib import contextmanager
from itertools import permutations
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.diar.attractor.abs_attractor import AbsAttractor
from espnet2.diar.decoder.abs_decoder import AbsDecoder
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.nets_utils import to_device

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetDiarizationModel(AbsESPnetModel):
    """
    ESPnetDiarizationModel is a speaker diarization model that utilizes various
    components such as encoders, decoders, and attractors to process speech data.
    Depending on the presence of an attractor, it can implement either SA-EEND
    or EEND-EDA methods for diarization.

    For more details on the methodologies used, refer to the following papers:
    - SA-EEND: https://arxiv.org/pdf/1909.06247.pdf
    - EEND-EDA: https://arxiv.org/pdf/2005.09921.pdf,
    https://arxiv.org/pdf/2106.10654.pdf

    Attributes:
        encoder (AbsEncoder): The encoder component used for processing input speech.
        normalize (Optional[AbsNormalize]): The normalization component for features.
        frontend (Optional[AbsFrontend]): The frontend feature extractor.
        specaug (Optional[AbsSpecAug]): The data augmentation component.
        label_aggregator (torch.nn.Module): Aggregates speaker labels.
        diar_weight (float): Weight for the diarization loss.
        attractor_weight (float): Weight for the attractor loss.
        attractor (Optional[AbsAttractor]): The attractor component for EEND-EDA.
        decoder (Optional[AbsDecoder]): The decoder component used for predictions.

    Args:
        frontend (Optional[AbsFrontend]): The frontend feature extractor.
        specaug (Optional[AbsSpecAug]): The spec augmentation module.
        normalize (Optional[AbsNormalize]): The normalization module.
        label_aggregator (torch.nn.Module): Module to aggregate speaker labels.
        encoder (AbsEncoder): The encoder module.
        decoder (AbsDecoder): The decoder module.
        attractor (Optional[AbsAttractor]): The attractor module.
        diar_weight (float): Weight for the diarization loss (default: 1.0).
        attractor_weight (float): Weight for the attractor loss (default: 1.0).

    Examples:
        model = ESPnetDiarizationModel(
            frontend=my_frontend,
            specaug=my_specaug,
            normalize=my_normalize,
            label_aggregator=my_label_aggregator,
            encoder=my_encoder,
            decoder=my_decoder,
            attractor=my_attractor,
            diar_weight=1.0,
            attractor_weight=1.0,
        )
        loss, stats, weight = model(speech_data, speech_lengths, speaker_labels)

    Raises:
        NotImplementedError: If both attractor and decoder are None.
    """

    @typechecked
    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        label_aggregator: torch.nn.Module,
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        attractor: Optional[AbsAttractor],
        diar_weight: float = 1.0,
        attractor_weight: float = 1.0,
    ):

        super().__init__()

        self.encoder = encoder
        self.normalize = normalize
        self.frontend = frontend
        self.specaug = specaug
        self.label_aggregator = label_aggregator
        self.diar_weight = diar_weight
        self.attractor_weight = attractor_weight
        self.attractor = attractor
        self.decoder = decoder

        if self.attractor is not None:
            self.decoder = None
        elif self.decoder is not None:
            self.num_spk = decoder.num_spk
        else:
            raise NotImplementedError

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        spk_labels: torch.Tensor = None,
        spk_labels_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Process input speech through the model and compute the loss.

        This method combines the frontend, encoder, and decoder to
        calculate the diarization loss. It also computes various
        statistics related to speaker diarization performance.

        Args:
            speech: Tensor of shape (Batch, samples) representing input speech.
            speech_lengths: Optional; Tensor of shape (Batch,) indicating the
                lengths of each input sequence. Default is None, which is
                useful for chunk iterators that do not provide lengths.
            spk_labels: Tensor of shape (Batch, ...) containing speaker labels.
            spk_labels_lengths: Optional; Tensor of shape (Batch,) indicating
                the lengths of each speaker label sequence.
            kwargs: Additional arguments; "utt_id" is among the inputs.

        Returns:
            Tuple containing:
                - loss: Computed loss value.
                - stats: Dictionary of statistics including loss components and
                    diarization metrics.
                - weight: Weight of the current batch.

        Raises:
            AssertionError: If the number of speech samples does not match the
            number of speaker labels.

        Examples:
            >>> model = ESPnetDiarizationModel(...)
            >>> speech = torch.randn(10, 16000)  # 10 samples of 1 second audio
            >>> speech_lengths = torch.tensor([16000] * 10)  # All 1 second long
            >>> spk_labels = torch.randint(0, 2, (10, 20, 3))  # Example labels
            >>> loss, stats, weight = model.forward(speech, speech_lengths, spk_labels)

        Note:
            Ensure that the input tensors are on the same device as the model
            for proper computation.
        """
        assert speech.shape[0] == spk_labels.shape[0], (speech.shape, spk_labels.shape)
        batch_size = speech.shape[0]

        # 1. Encoder
        # Use bottleneck_feats if exist. Only for "enh + diar" task.
        bottleneck_feats = kwargs.get("bottleneck_feats", None)
        bottleneck_feats_lengths = kwargs.get("bottleneck_feats_lengths", None)
        encoder_out, encoder_out_lens = self.encode(
            speech, speech_lengths, bottleneck_feats, bottleneck_feats_lengths
        )

        if self.attractor is None:
            # 2a. Decoder (baiscally a predction layer after encoder_out)
            pred = self.decoder(encoder_out, encoder_out_lens)
        else:
            # 2b. Encoder Decoder Attractors
            # Shuffle the chronological order of encoder_out, then calculate attractor
            encoder_out_shuffled = encoder_out.clone()
            for i in range(len(encoder_out_lens)):
                encoder_out_shuffled[i, : encoder_out_lens[i], :] = encoder_out[
                    i, torch.randperm(encoder_out_lens[i]), :
                ]
            attractor, att_prob = self.attractor(
                encoder_out_shuffled,
                encoder_out_lens,
                to_device(
                    self,
                    torch.zeros(
                        encoder_out.size(0), spk_labels.size(2) + 1, encoder_out.size(2)
                    ),
                ),
            )
            # Remove the final attractor which does not correspond to a speaker
            # Then multiply the attractors and encoder_out
            pred = torch.bmm(encoder_out, attractor[:, :-1, :].permute(0, 2, 1))
        # 3. Aggregate time-domain labels
        spk_labels, spk_labels_lengths = self.label_aggregator(
            spk_labels, spk_labels_lengths
        )

        # If encoder uses conv* as input_layer (i.e., subsampling),
        # the sequence length of 'pred' might be slighly less than the
        # length of 'spk_labels'. Here we force them to be equal.
        length_diff_tolerance = 2
        length_diff = spk_labels.shape[1] - pred.shape[1]
        if length_diff > 0 and length_diff <= length_diff_tolerance:
            spk_labels = spk_labels[:, 0 : pred.shape[1], :]

        if self.attractor is None:
            loss_pit, loss_att = None, None
            loss, perm_idx, perm_list, label_perm = self.pit_loss(
                pred, spk_labels, encoder_out_lens
            )
        else:
            loss_pit, perm_idx, perm_list, label_perm = self.pit_loss(
                pred, spk_labels, encoder_out_lens
            )
            loss_att = self.attractor_loss(att_prob, spk_labels)
            loss = self.diar_weight * loss_pit + self.attractor_weight * loss_att
        (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        ) = self.calc_diarization_error(pred, label_perm, encoder_out_lens)

        if speech_scored > 0 and num_frames > 0:
            sad_mr, sad_fr, mi, fa, cf, acc, der = (
                speech_miss / speech_scored,
                speech_falarm / speech_scored,
                speaker_miss / speaker_scored,
                speaker_falarm / speaker_scored,
                speaker_error / speaker_scored,
                correct / num_frames,
                (speaker_miss + speaker_falarm + speaker_error) / speaker_scored,
            )
        else:
            sad_mr, sad_fr, mi, fa, cf, acc, der = 0, 0, 0, 0, 0, 0, 0

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_pit=loss_pit.detach() if loss_pit is not None else None,
            sad_mr=sad_mr,
            sad_fr=sad_fr,
            mi=mi,
            fa=fa,
            cf=cf,
            acc=acc,
            der=der,
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spk_labels: torch.Tensor = None,
        spk_labels_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Collects features from the input speech signal.

        This method extracts features from the provided speech input and returns
        them along with their lengths. It can also handle speaker labels and their
        lengths, although they are not mandatory for this operation.

        Args:
            speech (torch.Tensor): The input speech signal of shape (Batch, Length).
            speech_lengths (torch.Tensor): A tensor indicating the lengths of each
                speech signal in the batch, of shape (Batch,).
            spk_labels (torch.Tensor, optional): A tensor containing speaker labels
                of shape (Batch, ...). Defaults to None.
            spk_labels_lengths (torch.Tensor, optional): A tensor containing the
                lengths of speaker labels of shape (Batch, ...). Defaults to None.
            **kwargs: Additional keyword arguments for future extensions.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'feats': Extracted features of shape (Batch, NFrames, Dim).
                - 'feats_lengths': Lengths of the extracted features of shape
                  (Batch,).

        Examples:
            >>> model = ESPnetDiarizationModel(...)
            >>> speech = torch.randn(2, 16000)  # Batch of 2, 16000 samples
            >>> speech_lengths = torch.tensor([16000, 16000])
            >>> features = model.collect_feats(speech, speech_lengths)
            >>> print(features['feats'].shape)  # Expected shape: (2, NFrames, Dim)

        Note:
            This function primarily uses the frontend module to process the input
            speech and generate features. If no frontend is specified, the
            raw speech input will be returned as features.
        """
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        bottleneck_feats: torch.Tensor,
        bottleneck_feats_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            speech (torch.Tensor): Input tensor of shape (Batch, Length, ...).
            speech_lengths (torch.Tensor): Lengths of the input sequences, shape (Batch,).
            bottleneck_feats (torch.Tensor): Optional tensor for enhancement and
                diarization, shape (Batch, Length, ...).
            bottleneck_feats_lengths (torch.Tensor): Lengths of the bottleneck features,
                shape (Batch,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - encoder_out (torch.Tensor): Output tensor from the encoder,
                  shape (Batch, Length2, Dim).
                - encoder_out_lens (torch.Tensor): Lengths of the output sequences,
                  shape (Batch,).

        Note:
            The `autocast` context is used to enable mixed precision
            training if available.

        Examples:
            >>> model = ESPnetDiarizationModel(...)
            >>> speech = torch.randn(32, 16000)  # Example batch of 32 audio samples
            >>> speech_lengths = torch.tensor([16000] * 32)  # Lengths of each sample
            >>> bottleneck_feats = torch.randn(32, 100, 40)  # Example bottleneck features
            >>> bottleneck_feats_lengths = torch.tensor([100] * 32)  # Lengths of bottleneck features
            >>> encoder_out, encoder_out_lens = model.encode(speech, speech_lengths,
            ... bottleneck_feats, bottleneck_feats_lengths)
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

            # 4. Forward encoder
            # feats: (Batch, Length, Dim)
            # -> encoder_out: (Batch, Length2, Dim)
            if bottleneck_feats is None:
                encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
            elif self.frontend is None:
                # use only bottleneck feature
                encoder_out, encoder_out_lens, _ = self.encoder(
                    bottleneck_feats, bottleneck_feats_lengths
                )
            else:
                # use both frontend and bottleneck feats
                # interpolate (copy) feats frames
                # to match the length with bottleneck_feats
                feats = F.interpolate(
                    feats.transpose(1, 2), size=bottleneck_feats.shape[1]
                ).transpose(1, 2)
                # concatenate frontend LMF feature and bottleneck feature
                encoder_out, encoder_out_lens, _ = self.encoder(
                    torch.cat((bottleneck_feats, feats), 2), bottleneck_feats_lengths
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

    def pit_loss_single_permute(self, pred, label, length):
        """
        Calculates the PIT loss for a single permutation of predictions.

        This method computes the Binary Cross Entropy (BCE) loss between
        the predicted values and the ground truth labels for a given
        permutation. It also applies a length mask to ignore padding
        in the labels.

        Args:
            pred (torch.Tensor): The predicted values of shape (Batch, Time, Output).
            label (torch.Tensor): The ground truth labels of shape (Batch, Time, Output).
            length (torch.Tensor): The lengths of each sequence in the batch.

        Returns:
            torch.Tensor: The calculated loss for the given permutation,
                          of shape (Batch, 1).

        Examples:
            >>> pred = torch.tensor([[[0.1, 0.2], [0.5, 0.6]],
            ...                       [[0.3, 0.4], [0.7, 0.8]]])
            >>> label = torch.tensor([[[1, 0], [0, 1]],
            ...                        [[0, 1], [1, 0]]])
            >>> length = torch.tensor([2, 2])
            >>> loss = pit_loss_single_permute(pred, label, length)
            >>> print(loss)
            tensor([[0.3567], [0.6124]])
        """
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        mask = self.create_length_mask(length, label.size(1), label.size(2))
        loss = bce_loss(pred, label)
        loss = loss * mask
        loss = torch.sum(torch.mean(loss, dim=2), dim=1)
        loss = torch.unsqueeze(loss, dim=1)
        return loss

    def pit_loss(self, pred, label, lengths):
        """
        Calculate the permutation-invariant training (PIT) loss.

        This method computes the PIT loss for a given set of predictions and
        corresponding labels by considering all possible permutations of the
        labels. The minimum loss across all permutations is returned, along
        with the corresponding permutation indices and the permuted labels.

        Args:
            pred (torch.Tensor): The predicted outputs with shape
                (Batch, Length, num_output).
            label (torch.Tensor): The ground truth labels with shape
                (Batch, Length, num_output).
            lengths (torch.Tensor): The lengths of the sequences,
                with shape (Batch,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[np.ndarray], torch.Tensor]:
                - loss: The calculated PIT loss.
                - min_idx: The indices of the minimum loss permutation for each
                  sample in the batch.
                - permute_list: A list containing all permutations of labels.
                - label_permute: The permuted labels corresponding to the
                  minimum loss.

        Note:
            Credit to https://github.com/hitachi-speech/EEND for the
            implementation of this method.

        Examples:
            >>> pred = torch.rand(2, 5, 3)  # Example predictions
            >>> label = torch.rand(2, 5, 3)  # Example labels
            >>> lengths = torch.tensor([5, 5])  # Lengths of each sequence
            >>> loss, min_idx, permute_list, label_permute = pit_loss(pred, label, lengths)
        """
        # Note (jiatong): Credit to https://github.com/hitachi-speech/EEND
        num_output = label.size(2)
        permute_list = [np.array(p) for p in permutations(range(num_output))]
        loss_list = []
        for p in permute_list:
            label_perm = label[:, :, p]
            loss_perm = self.pit_loss_single_permute(pred, label_perm, lengths)
            loss_list.append(loss_perm)
        loss = torch.cat(loss_list, dim=1)
        min_loss, min_idx = torch.min(loss, dim=1)
        loss = torch.sum(min_loss) / torch.sum(lengths.float())
        batch_size = len(min_idx)
        label_list = []
        for i in range(batch_size):
            label_list.append(label[i, :, permute_list[min_idx[i]]].data.cpu().numpy())
        label_permute = torch.from_numpy(np.array(label_list)).float()
        return loss, min_idx, permute_list, label_permute

    def create_length_mask(self, length, max_len, num_output):
        """
        Creates a length mask tensor for the given input lengths, which is useful in
        ensuring that only valid entries in the label tensor are considered during
        loss computation.

        This function generates a mask of shape (batch_size, max_len, num_output),
        where `max_len` is the maximum length of the sequences in the batch, and
        `num_output` is the number of output channels. The mask is populated with
        ones for the valid lengths and zeros for the padded lengths.

        Args:
            length (torch.Tensor): A 1D tensor containing the lengths of each
                sequence in the batch.
            max_len (int): The maximum length of the sequences to be considered.
            num_output (int): The number of output channels.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, max_len, num_output)
                containing the length mask.

        Examples:
            >>> length = torch.tensor([3, 5, 2])
            >>> max_len = 5
            >>> num_output = 4
            >>> mask = create_length_mask(length, max_len, num_output)
            >>> print(mask)
            tensor([[[1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]],

                    [[1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.]],

                    [[1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]]])

        Note:
            This function assumes that the input tensor `length` is a 1D tensor
            containing the lengths of the sequences in the batch.
        """
        batch_size = len(length)
        mask = torch.zeros(batch_size, max_len, num_output)
        for i in range(batch_size):
            mask[i, : length[i], :] = 1
        mask = to_device(self, mask)
        return mask

    def attractor_loss(self, att_prob, label):
        """
        Calculate the attractor loss based on the attractor probabilities.

        The attractor loss is computed using binary cross-entropy loss between
        the predicted attractor probabilities and the ground truth attractor labels.
        The ground truth labels are created such that all speakers are labeled as
        present (1) except for an additional label which is marked as absent (0).

        Args:
            att_prob (torch.Tensor): The predicted attractor probabilities of shape
                (Batch, num_spk + 1, 1).
            label (torch.Tensor): The ground truth labels of shape
                (Batch, num_spk, 1).

        Returns:
            torch.Tensor: The computed attractor loss as a scalar tensor.

        Examples:
            >>> model = ESPnetDiarizationModel(...)
            >>> att_prob = torch.tensor([[0.9], [0.2], [0.8]])
            >>> label = torch.tensor([[1], [0]])
            >>> loss = model.attractor_loss(att_prob, label)
            >>> print(loss)
        """
        batch_size = len(label)
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        # create attractor label [1, 1, ..., 1, 0]
        # att_label: (Batch, num_spk + 1, 1)
        att_label = to_device(self, torch.zeros(batch_size, label.size(2) + 1, 1))
        att_label[:, : label.size(2), :] = 1
        loss = bce_loss(att_prob, att_label)
        loss = torch.mean(torch.mean(loss, dim=1))
        return loss

    @staticmethod
    def calc_diarization_error(pred, label, length):
        """
        Calculate the diarization error for predicted and true labels.

        This method computes various metrics to evaluate the performance of
        speaker diarization predictions. It calculates the speech activity
        detection error, speaker miss, false alarm, and overall speaker
        diarization error based on the predicted and ground truth labels.

        Args:
            pred (torch.Tensor): The predicted labels with shape
                (batch_size, max_len, num_output).
            label (torch.Tensor): The ground truth labels with shape
                (batch_size, max_len, num_output).
            length (torch.Tensor): A tensor containing the actual lengths of
                each sequence in the batch.

        Returns:
            Tuple[float, float, float, float, float, float, float, float, float]:
                A tuple containing the following metrics:
                - correct: The number of correctly predicted frames.
                - num_frames: The total number of frames.
                - speech_scored: The number of frames with detected speech.
                - speech_miss: The number of missed speech frames.
                - speech_falarm: The number of false alarm speech frames.
                - speaker_scored: The number of scored speakers.
                - speaker_miss: The number of missed speakers.
                - speaker_falarm: The number of false alarm speakers.
                - speaker_error: The total speaker error.

        Note:
            This method credits the implementation to the EEND project
            (https://github.com/hitachi-speech/EEND).

        Examples:
            >>> pred = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])
            >>> label = torch.tensor([[[1, 0], [0, 1]], [[0, 0], [1, 1]]])
            >>> length = torch.tensor([2, 2])
            >>> metrics = ESPnetDiarizationModel.calc_diarization_error(pred, label, length)
            >>> print(metrics)
            (correct, num_frames, speech_scored, speech_miss,
             speech_falarm, speaker_scored, speaker_miss,
             speaker_falarm, speaker_error)
        """
        # Note (jiatong): Credit to https://github.com/hitachi-speech/EEND

        (batch_size, max_len, num_output) = label.size()
        # mask the padding part
        mask = np.zeros((batch_size, max_len, num_output))
        for i in range(batch_size):
            mask[i, : length[i], :] = 1

        # pred and label have the shape (batch_size, max_len, num_output)
        label_np = label.data.cpu().numpy().astype(int)
        pred_np = (pred.data.cpu().numpy() > 0).astype(int)
        label_np = label_np * mask
        pred_np = pred_np * mask
        length = length.data.cpu().numpy()

        # compute speech activity detection error
        n_ref = np.sum(label_np, axis=2)
        n_sys = np.sum(pred_np, axis=2)
        speech_scored = float(np.sum(n_ref > 0))
        speech_miss = float(np.sum(np.logical_and(n_ref > 0, n_sys == 0)))
        speech_falarm = float(np.sum(np.logical_and(n_ref == 0, n_sys > 0)))

        # compute speaker diarization error
        speaker_scored = float(np.sum(n_ref))
        speaker_miss = float(np.sum(np.maximum(n_ref - n_sys, 0)))
        speaker_falarm = float(np.sum(np.maximum(n_sys - n_ref, 0)))
        n_map = np.sum(np.logical_and(label_np == 1, pred_np == 1), axis=2)
        speaker_error = float(np.sum(np.minimum(n_ref, n_sys) - n_map))
        correct = float(1.0 * np.sum((label_np == pred_np) * mask) / num_output)
        num_frames = np.sum(length)
        return (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        )
