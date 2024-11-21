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
    """Speaker embedding extraction model.

    Core model for diverse speaker-related tasks (e.g., verification, open-set
    identification, diarization)

    The model architecture comprises mainly 'encoder', 'pooling', and
    'projector'.
    In common speaker recognition field, the combination of three would be
    usually named as 'speaker_encoder' (or speaker embedding extractor).
    We splitted it into three for flexibility in future extensions:
      - 'encoder'   : extract frame-level speaker embeddings.
      - 'pooling'   : aggregate into single utterance-level embedding.
      - 'projector' : (optional) additional processing (e.g., one fully-
                      connected layer) to derive speaker embedding.

    Possibly, in the future, 'pooling' and/or 'projector' can be integrated as
    a 'decoder', depending on the extension for joint usage of different tasks
    (e.g., ASR, SE, target speaker extraction).
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
        """Feed-forward through encoder layers and aggregate into utterance-level

        feature.

        Args:
            speech: (Batch, samples)
            speech_lengths: (Batch,)
            extract_embd: a flag which doesn't go through the classification
                head when set True
            spk_labels: (Batch, )
            one-hot speaker labels used in the train phase
            task_tokens: (Batch, )
            task tokens used in case of token-based trainings
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
        frame_level_feats = self.encoder(feats)

        return frame_level_feats

    def aggregate(self, frame_level_feats: torch.Tensor) -> torch.Tensor:
        utt_level_feat = self.aggregator(frame_level_feats)

        return utt_level_feat

    def project_spk_embd(self, utt_level_feat: torch.Tensor) -> torch.Tensor:
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
        feats, feats_lengths = self.extract_feats(speech, speech_lengths)
        return {"feats": feats}
