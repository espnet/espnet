# Copyright 2024 WAVLab (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
from contextlib import contextmanager
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import parse as V
from typeguard import typechecked

try:
    from torcheval.metrics import functional as EvalFunction

    is_torcheval_available = True
except ImportError:
    is_torcheval_available = False

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.cls.decoder.abs_decoder import AbsDecoder
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


logger = logging.getLogger(__name__)


class ESPnetClassificationModel(AbsESPnetModel):
    """Classification model

    A simple Classification model
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
        decoder: AbsDecoder,
        text_encoder: Optional[AbsEncoder] = None,
        classification_type="multi-class",
        lsm_weight: float = 0.0,
        mixup_augmentation: bool = False,
    ):
        super().__init__()
        if not is_torcheval_available:
            raise ImportError(
                "`torcheval` is not available. Please install it "
                "via `pip install torcheval` in your environment."
                "More info at: `https://pytorch.org/torcheval/stable/`"
            )
        self.vocab_size = vocab_size
        self.token_list = token_list.copy()
        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder = decoder
        self.classification_type = classification_type
        self.lsm_weight = lsm_weight
        if classification_type == "multi-label":
            # Also includes binary classification
            self.classification_function = F.sigmoid
            self.classification_loss = nn.BCEWithLogitsLoss()
        elif classification_type == "multi-class":
            self.classification_function = partial(F.softmax, dim=-1)
            self.classification_loss = nn.CrossEntropyLoss(label_smoothing=lsm_weight)
        else:
            raise ValueError(
                "Valid classification types are 'multi-label' and 'multi-class'"
            )
        self.mixup_augmentation = mixup_augmentation
        self.metric_functions = self.setup_metrics_()

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        label: torch.Tensor,
        label_lengths: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Pass the input through the model and calculate the loss.

        Args:
            speech: (Batch, samples)
            speech_lengths: (Batch, )
            label: (Batch, Length)
            label_lengths: (Batch, )
            text: (Batch, Length): Optional, used if text_encoder is provided
            text_lengths: (Batch, ): Optional, used if text_encoder is provided
        Returns:
            loss: (1,)
            stats: dict
            weight
        """
        assert len(label.shape) == 2, label.shape
        assert speech.shape[0] == label.shape[0], (speech.shape, label.shape)
        assert text is None or (
            len(text.shape) == 2 and text.shape[0] == label.shape[0]
        ), (
            text.shape,
            label.shape,
        )
        assert (
            text is None or self.text_encoder is not None
        ), "You must provide text encoder if text is provided."
        batch_size = speech.shape[0]
        onehot_ = label_to_onehot(
            label,
            label_lengths,
            self.vocab_size,
            self.classification_type,
            lsm_weight=self.lsm_weight if self.training else 0.0,
        )
        if self.training and self.mixup_augmentation:
            assert (
                self.classification_type == "multi-label"
            ), "Mixup is only for multi-label classification"
            if speech_lengths.min() != speech_lengths.max():
                logger.warning(
                    "Mixup is not recommended for variable length input. "
                    "It may not work as expected."
                )
            speech, onehot_ = mixup_augment(speech, onehot_, mixup_prob=1.0)

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(
            speech, speech_lengths, text, text_lengths
        )
        # 2. Decoder
        logits = self.decoder(encoder_out, encoder_out_lens)

        # 3. Compute loss and acc
        loss = self.classification_loss(logits, onehot_)

        pred = self.classification_function(logits)
        stats = {"loss": loss.detach()}
        for metric_name, metric_fn in self.metric_functions.items():
            target = (
                label.squeeze(-1)
                if self.classification_type == "multi-class"
                else onehot_
            )
            val = metric_fn(pred, target).detach()
            stats[metric_name] = val

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def score(
        self, speech: torch.Tensor, speech_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass at scoring (inference)

        Args:
            speech: (Batch, samples)
            speech_lengths: (Batch, )
        Returns:
            scores: (Batch, n_classes)
        Assumes Batch=1
        """
        batch = speech.size(0)
        assert batch == 1, "Batch size must be 1 for scoring."
        if speech_lengths is None:
            speech_lengths = torch.tensor([speech.size(1)], device=speech.device)
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        logits, _ = self.decoder.score(x=encoder_out.squeeze(0), ys=None, state=None)
        scores = self.classification_function(logits)
        return scores.unsqueeze(0)

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the input speech.

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
            text: (Batch, Length) Optional
            text_lengths: (Batch,) Optional
        Returns:
            scores: (Batch, Length, n_classes)
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

        if self.text is not None:
            text_encoder_out, text_encoder_out_lens = self.text_encoder(
                text, text_lengths
            )
            encoder_out, encoder_out_lens = self.combine_encodings(
                text_encoder_out, text_encoder_out_lens, encoder_out, encoder_out_lens
            )

        return encoder_out, encoder_out_lens

    def combine_encodings(
        self,
        text_encoding: torch.Tensor,
        text_encoding_lens: torch.Tensor,
        speech_encoding: torch.Tensor,
        speech_encoding_lens: torch.Tensor,
    ):
        """Concatenate text and speech encodings

        Args:
            text_encoding: (Batch, Length1, Dim): Maybe padded
            text_encoding_lens: (Batch,) Each element is Length_i
                denoting unpadded elements.
            speech_encoding: (Batch, Length2, Dim): Maybe padded
            speech_encoding_lens: (Batch,) Each element is Length_j
                denoting unpadded elements.
        Returns:
            encoder_out: (Batch, max(Length_i+Length_j), Dim)
            encoder_out_lens: (Batch,)
        TODO(shikhar): Add support for attn, refactor
        """
        encoder_out_lens = text_encoding_lens + speech_encoding_lens
        max_len = encoder_out_lens.max()
        batch_size = text_encoding.size(0)
        dim = text_encoding.size(-1)
        assert dim == speech_encoding.size(-1), (
            "Dimensions must match for text and speech encodings",
            dim,
            speech_encoding.size(-1),
        )
        encoder_out = torch.zeros(
            (batch_size, max_len, dim),
            dtype=text_encoding.dtype,
            device=text_encoding.device,
        )

        for i in range(batch_size):
            text_len = text_encoding_lens[i].item()
            speech_len = speech_encoding_lens[i].item()

            encoder_out[i, :text_len] = text_encoding[i, :text_len]
            encoder_out[i, text_len : text_len + speech_len] = speech_encoding[
                i, :speech_len
            ]

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

    def setup_metrics_(self):
        if self.classification_type == "multi-class":
            return {
                "acc": EvalFunction.multiclass_accuracy,
                "macro_precision": partial(
                    EvalFunction.multiclass_precision,
                    average="macro",
                    num_classes=self.vocab_size,
                ),
            }
        elif self.classification_type == "multi-label":
            return {
                "acc": partial(EvalFunction.multilabel_accuracy, criteria="hamming"),
                # acc is usually high if data is imabalanced
                "mAP": partial(
                    EvalFunction.multilabel_auprc,
                    average="macro",
                    num_labels=self.vocab_size,
                ),
            }


def label_to_onehot(
    label: torch.Tensor,
    label_lengths: torch.Tensor,
    vocab_size: int,
    classification_type: str,
    lsm_weight: float = 0.0,
) -> torch.Tensor:
    """Convert label to onehot.
    Args
        label: (Batch, Length) pad value should be -1
        label_lengths: (Batch,) only used in asserts
        vocab_size: int
        classification_type: str "multi-class" or "multi-label"
        lsm_weight: float, label smoothing weight
    Returns
        onehot: (Batch, Length, vocab_size)
    """
    if classification_type == "multi-class":
        assert label_lengths.max() == 1, "Only one label per sample"
        return F.one_hot(label.squeeze(-1), vocab_size).float()
    elif classification_type == "multi-label":
        assert (
            label_lengths.min() == label_lengths.max() or label.min() == -1
        ), "Pad value should be -1"
        label = label.masked_fill(label == -1, vocab_size)
        onehot = F.one_hot(label.view(-1), vocab_size + 2)
        onehot = onehot[:, :-2]  # Remove dummy columns, blank=-2, unk=-1
        onehot = onehot.view(label.size(0), -1, vocab_size)
        onehot = onehot.sum(dim=1)
        onehot = onehot.float()
        if lsm_weight > 0.0:
            onehot = onehot * (1 - 2 * lsm_weight) + lsm_weight
        return onehot
    else:
        raise ValueError(
            "Valid classification types are 'multi-label' and 'multi-class'"
        )


def mixup_augment(speech: torch.Tensor, onehot: torch.Tensor, mixup_prob: float):
    """Mixup augmentation for multi-label classification

    Args:
        speech: (Batch, Length, Dim)
        onehot: (Batch, n_classes)
        mixup_prob: Apply mixup with this probability
    Returns:
        speech: (Batch, Length, Dim)
        onehot: (Batch, n_classes)
    """
    batch_size = speech.size(0)
    assert onehot.size(0) == batch_size
    apply_augmentation = torch.rand((batch_size), device=speech.device) < mixup_prob
    mix_lambda = (
        torch.distributions.Beta(0.8, 0.8)
        .sample(sample_shape=(batch_size, 1))
        .to(speech.device)
    )
    perm = torch.randperm(batch_size).to(speech.device)
    identity_perm = torch.arange(batch_size, device=speech.device)
    perm[~apply_augmentation] = identity_perm[~apply_augmentation]
    # speech = speech - speech.mean(dim=1, keepdim=True)
    speech = mix_lambda * speech + (1 - mix_lambda) * speech[perm]
    onehot = mix_lambda * onehot + (1 - mix_lambda) * onehot[perm]
    return speech, onehot
