# Copyright 2023 Jee-weon Jung
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Dict, List, Optional, Tuple, Union

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
        loss: Optional[List[AbsLoss]],
        loss_weights: Optional[List[float]] = None,
        loss_names: Optional[List[str]] = None,
    ):

        super().__init__()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.pooling = pooling
        self.projector = projector
        self.loss = loss
        self.loss_weights = loss_weights
        self.loss_names = loss_names

    @typechecked
    def forward(
        self,
        speech: torch.Tensor,
        spk_labels: Optional[torch.Tensor] = None,
        spf_labels: Optional[torch.Tensor] = None,
        pmos_labels: Optional[torch.Tensor] = None,
        task_tokens: Optional[torch.Tensor] = None,
        extract_embd: bool = False,
        precomp_frame_feats: Optional[torch.Tensor] = None,
        precomp_frame_feats_lengths: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor],
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        torch.Tensor
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
            spf_labels: (Batch, )
            pmos_labels: (Batch, )
            precomp_frame_feats: (Batch, frames, feats)
            precomp_frame_feats_lengths: (Batch,)
            one-hot spoofing labels used in the train phase
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


        if precomp_frame_feats is None or precomp_frame_feats_lengths is None:
            precomp_frame_feats = kwargs.get("frame_feats", None)
            precomp_frame_feats_lengths = kwargs.get("frame_feats_lengths", None)

        # 1. extract low-level feats (e.g., mel-spectrogram or MFCC)
        # Will do nothing for raw waveform-based models (e.g., RawNets)
        feats, _ = self.extract_feats(speech, None)

        frame_level_feats = self.encode_frame(feats, precomp_frame_feats, precomp_frame_feats_lengths)
        if isinstance(frame_level_feats, tuple):
            frame_level_feats, attention_weights = frame_level_feats

        # 2. aggregation into utterance-level
        utt_level_feat = self.pooling(frame_level_feats, task_tokens)

        # 3. (optionally) go through further projection(s)
        spk_embd = self.project_spk_embd(utt_level_feat)

        if extract_embd and not return_attention_weights:
            return spk_embd
        elif extract_embd and return_attention_weights:
            assert attention_weights is not None, "Attention weights are None, cannot return"
            return spk_embd, attention_weights

        # 4. calculate loss
        loss_names = self.loss_names
        labels = []
        for i, loss in enumerate(self.loss):
            if loss_names[i] == "spk":
                assert spk_labels is not None, "spk_labels is None, cannot compute loss"
                labels.append(spk_labels)
            elif loss_names[i] == "spf":
                assert spf_labels is not None, "spf_labels is None, cannot compute loss"
                labels.append(spf_labels)
            elif loss_names[i] == "pmos":
                assert pmos_labels is not None, "pmos_labels is None, cannot compute loss"
                labels.append(pmos_labels)
            else:
                raise NotImplementedError(f"Loss name {loss_names[i]} is not supported")
            
        assert len(self.loss) == len(labels), "Number of losses and labels do not match"
        losses = [loss_fn(spk_embd, label.squeeze()) for loss_fn, label in zip(self.loss, labels)]

        # calculate weighted sum of losses
        if self.loss_weights is not None:
            loss = sum(w * l for w, l in zip(self.loss_weights, losses))
        else:
            loss = sum(losses)

        # Prepare stats dictionary
        stats = {f"{name}_loss": loss.detach() 
                for name, loss in zip(loss_names, losses)}
        stats["loss"] = loss.detach()

        # Make gathered results
        loss, stats, weight = force_gatherable(
            (loss, stats, batch_size), loss.device
        )
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

    def encode_frame(self, feats: torch.Tensor, precomp_frame_feats: torch.Tensor, precomp_frame_feats_lengths: torch.Tensor) -> torch.Tensor:
        if precomp_frame_feats is not None:
            frame_level_feats = self.encoder(feats, precomp_frame_feats, precomp_frame_feats_lengths)
        else:
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
