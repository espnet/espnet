from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.ser.loss.abs_loss import AbsLoss
from espnet2.ser.pooling.abs_pooling import AbsPooling
from espnet2.ser.projector.abs_projector import AbsProjector
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetSERModel(ESPnetASRModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    @typechecked
    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        preencoder: Optional[AbsPreEncoder],
        encoder: Optional[AbsEncoder],
        postencoder: Optional[AbsPostEncoder],
        pooling: Optional[AbsPooling],
        projector: Optional[AbsProjector],
        loss: Optional[AbsLoss],
        extract_feats_in_collect_stats: bool = True,
        pre_postencoder_norm: bool = False,
    ):

        AbsESPnetModel.__init__(self)
        self.pre_postencoder_norm = pre_postencoder_norm
        self.frontend = frontend
        self.specaug = specaug
        self.preencoder = preencoder
        self.encoder = encoder
        self.postencoder = postencoder
        self.pooling = pooling
        self.projector = projector
        self.loss = loss
        self.softmax = torch.nn.Softmax(dim=-1)

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        get_prediction: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        # Check that batch_size is unified
        if speech_lengths is not None:
            assert speech.shape[0] == speech_lengths.shape[0], (
                speech.shape,
                speech_lengths.shape,
            )
        batch_size = speech.shape[0]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        # 2. aggregation into utterance-level
        utt_level_feat = self.pooling(encoder_out)
        # 3. go through further projection(s)
        pred = self.projector(utt_level_feat)

        if get_prediction:
            # 4. get prediction
            pred_emo = self.softmax(pred).argmax(dim=-1)
            return pred_emo
        loss = None
        stats = dict()
        loss = self.loss(pred, emotion_labels.squeeze(1))
        # Collect total loss stats
        stats["loss"] = loss.detach()
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self.extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

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

        return feats, feat_lengths

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self.extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # Encoder
        if self.encoder is not None:
            feats, feats_lengths = self.encoder(feats, feats_lengths)

        # Post-encoder
        if self.postencoder is not None:
            feats, feats_lengths = self.postencoder(feats, feats_lengths)

        return feats, feats_lengths
