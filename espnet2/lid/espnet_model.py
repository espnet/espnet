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


class ESPnetLIDModel(AbsESPnetModel):
    r"""ESPnet LID model

    Support for language identification and language embedding extraction.
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
        extract_feats_in_collect_stats: Optional[bool] = None,
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
        speech_lengths: torch.Tensor,
        lid_labels: Optional[torch.Tensor] = None,
        extract_embd: bool = False,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor],
        torch.Tensor,
    ]:
        r"""Forward pass of the LID model.

        Processes raw speech through frontend, encoder, pooling, and loss modules.

        Args:
            speech: Input waveform tensor (batch_size, num_samples)
            speech_lengths: Lengths of each input in the batch (batch_size,)
            lid_labels: Ground truth language labels (batch_size,)
            extract_embd: If True, return language embeddings and
                          predictions (inference mode)

        Returns:
            - If extract_embd=True (inference mode):
                Tuple(lang_embd, pred_lids)
            - If training:
                Tuple(loss, stats_dict, batch_weight)
        """

        if lid_labels is not None:
            assert speech.shape[0] == lid_labels.shape[0], (
                speech.shape,
                lid_labels.shape,
            )

        batch_size = speech.shape[0]
        stats = dict()

        # 1. Extract feats
        # Must transfer speech_lengths to extract_feats to get correct feat_lengths
        feats, feat_lengths = self.extract_feats(speech, speech_lengths)
        frame_level_feats = self.encode_frame(feats)

        # 2. Aggregation into utterance-level
        utt_level_feat = self.pooling(frame_level_feats, feat_lengths=feat_lengths)

        # 3. (Optionally) Go through further projection(s)
        lang_embd = self.project_lang_embd(utt_level_feat)

        # 4. Calculate loss
        # NOTE: If lid_labels is None, loss and accuracy are None

        loss, accuracy, pred_lids = self.loss(lang_embd, lid_labels)

        if extract_embd:
            return lang_embd, pred_lids

        stats["loss"] = loss.detach()
        if accuracy is not None:  # if not provide labels, accuracy is None
            stats["accuracy"] = accuracy.detach()

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = speech.shape[0]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else torch.ones(batch_size, device=speech.device, dtype=torch.int)
            * speech.shape[1]
        )

        # 1. Extract feats
        if self.frontend is not None:
            feats, feat_lengths = self.frontend(speech, speech_lengths)
        else:
            feats = speech
            feat_lengths = None

        # 2. Apply augmentations
        if self.specaug is not None and self.training:
            feats, _ = self.specaug(feats, feat_lengths)

        # 3. Normalize
        if self.normalize is not None:
            feats, _ = self.normalize(feats, feat_lengths)

        return feats, feat_lengths

    def encode_frame(self, feats: torch.Tensor) -> torch.Tensor:
        frame_level_feats = self.encoder(feats)

        return frame_level_feats

    def project_lang_embd(self, utt_level_feat: torch.Tensor) -> torch.Tensor:
        if self.projector is not None:
            lang_embd = self.projector(utt_level_feat)
        else:
            lang_embd = utt_level_feat

        return lang_embd

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        lid_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self.extract_feats(speech, speech_lengths)
        return {"feats": feats}
