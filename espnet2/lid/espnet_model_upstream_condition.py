# Copyright 2025 @Qingzheng-Wang
#
# Language Identification (LID) model with flexible lang2vec integration.
#
# This model supports multiple types of lang2vec features for conditioning, including:
#   - geo (if use geo vector, it is geolocation-aware LID)
#   - phonology_knn
#   - syntax_knn
#   - inventory_knn
#
# It enables upstream lang2vec conditioning at selected intermediate layers
# of the frontend model (e.g., MMS), and supports downstream auxiliary lang2vec
# prediction as an auxiliary learning objective.
# The model architecture allows for independent or shared conditioning modules
# across layers, and can be configured to freeze or train the conditioning
# projections.
#
# Reference:
#   Geolocation-Aware Robust Spoken Language Identification
#   TODO: add arxiv link
#
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.lid.espnet_model import ESPnetLIDModel
from espnet2.lid.loss.aamsoftmax_sc_topk_lang2vec import AAMSoftmaxSCTopKLang2Vec
from espnet2.spk.loss.abs_loss import AbsLoss
from espnet2.spk.pooling.abs_pooling import AbsPooling
from espnet2.spk.projector.abs_projector import AbsProjector
from espnet2.torch_utils.device_funcs import force_gatherable


class ESPnetLIDUpstreamConditionModel(ESPnetLIDModel):
    """ESPnet LID model with upstream lang2vec conditioning and downstream prediction.

    Args:
        frontend: feature extractor module
        specaug: optional SpecAugment module
        normalize: optional normalization module
        encoder: frame-level encoder (e.g., ECAPA-TDNN)
        pooling: pooling layer (e.g., AttnStatPooling)
        projector: embedding projection module (e.g., RawNet3)
        loss: main loss function (e.g., AAMSoftmax)
        encoder_condition: ModuleDict of condition module(s) for encoder layers,
                           key is layer index, value is encoder condition module
                           (can be shared or independent)
        pooling_condition: ModuleDict of condition module(s) for pooling layers,
                           key is layer index, value is pooling condition module
                           (can be shared or independent)
        projector_condition: ModuleDict of condition module(s) for projector layers,
                             key is layer index, value is projector condition module
                             (can be shared or independent)
        lang2vec_conditioning_layers: list of encoder layer indices to apply
                                      lang2vec conditioning
        apply_intermediate_lang2vec_loss: whether to apply auxiliary lang2vec
                                          prediction loss at intermediate layers
        apply_intermediate_lang2vec_condition: whether to apply lang2vec
                                               conditioning at intermediate layers
        inter_lang2vec_loss_weight: weight for the intermediate lang2vec loss
        cutoff_gradient_from_backbone: whether to cutoff gradient from upstream
                                       (frontend) backbone to the conditioning projection
        cutoff_gradient_before_condproj: whether to cutoff gradient before
                                          conditioning projection
        shared_conditioning_proj: if True, share conditioning projection across layers

    Reference:
        Geolocation-Aware Robust Spoken Language Identification
        TODO: add arxiv link
    """

    @typechecked
    def __init__(
        self,
        # ======== Model Architecture ========
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: Optional[AbsEncoder],
        pooling: Optional[AbsPooling],
        projector: Optional[AbsProjector],
        loss: Optional[AbsLoss],
        # ======== Conditioning Modules ========
        encoder_condition: Optional[nn.ModuleDict] = None,
        pooling_condition: Optional[nn.ModuleDict] = None,
        projector_condition: Optional[nn.ModuleDict] = None,
        # ======== Conditioning Configs ========
        lang2vec_conditioning_layers: List[int] = None,
        apply_intermediate_lang2vec_loss: bool = False,
        apply_intermediate_lang2vec_condition: bool = True,
        inter_lang2vec_loss_weight: float = 0.0,
        cutoff_gradient_from_backbone: Optional[bool] = True,
        cutoff_gradient_before_condproj: Optional[bool] = False,
        shared_conditioning_proj: Optional[bool] = False,
    ):

        super().__init__(
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            pooling=pooling,
            projector=projector,
            loss=loss,
        )

        self.frontend.upstream.upstream.model.encoder.ecapa_encoder = encoder_condition
        self.frontend.upstream.upstream.model.encoder.pooling = pooling_condition
        self.frontend.upstream.upstream.model.encoder.projector = projector_condition

        if lang2vec_conditioning_layers is not None:
            lang2vec_conditioning_layers = sorted(lang2vec_conditioning_layers)
            # lang2vec_head is the "GeoPred" layer noted in the paper if use geo vector
            lang2vec_head = nn.ModuleDict()
            for layer_idx in lang2vec_conditioning_layers:
                projector_condition_output_size = projector_condition[
                    str(layer_idx)
                ].output_size()
                lang2vec_head[str(layer_idx)] = nn.Sequential(
                    nn.Linear(projector_condition_output_size, self.loss.lang2vec_dim),
                )
            self.frontend.upstream.upstream.model.encoder.lang2vec_head = lang2vec_head

        self.frontend.upstream.upstream.model.encoder.lang2vec_conditioning_layers = (
            lang2vec_conditioning_layers
        )

        self.frontend.upstream.upstream.model.encoder.apply_intermediate_lang2vec_condition = (
            apply_intermediate_lang2vec_condition
        )

        # Each conditioning projection layer is independent
        if (
            lang2vec_conditioning_layers is not None
            and apply_intermediate_lang2vec_condition
        ):
            if shared_conditioning_proj:
                lang2vec_conditioning_projs = nn.Linear(
                    self.loss.lang2vec_dim,
                    self.frontend.upstream.upstream.model.encoder.config.hidden_size,
                )
            else:
                lang2vec_conditioning_projs = nn.ModuleDict()
                for layer_idx in lang2vec_conditioning_layers:
                    lang2vec_conditioning_projs[str(layer_idx)] = nn.Sequential(
                        nn.Linear(
                            self.loss.lang2vec_dim,
                            self.frontend.upstream.upstream.model.encoder.config.hidden_size,
                        ),
                    )
            self.frontend.upstream.upstream.model.encoder.lang2vec_conditioning_projs = (
                lang2vec_conditioning_projs
            )
            self.frontend.upstream.upstream.model.encoder.shared_conditioning_proj = (
                shared_conditioning_proj
            )

        self.frontend.upstream.upstream.model.encoder.cutoff_gradient_from_backbone = (
            cutoff_gradient_from_backbone
        )
        self.frontend.upstream.upstream.model.encoder.cutoff_gradient_before_condproj = (
            cutoff_gradient_before_condproj
        )

        assert isinstance(self.loss, AAMSoftmaxSCTopKLang2Vec), (
            f"ESPnetLIDUpstreamConditionModel only supports AAMSoftmaxSCTopKLang2Vec, "
            f"but got {type(self.loss)}"
        )
        # NOTE(qingzheng): Keep this line (assign aamsoftmax_loss to frontend)
        # for compatibility with early model checkpoints.
        self.frontend.upstream.upstream.model.encoder.aamsoftmax_loss = self.loss

        self.apply_intermediate_lang2vec_loss = apply_intermediate_lang2vec_loss
        self.inter_lang2vec_loss_weight = inter_lang2vec_loss_weight

    @typechecked
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        lid_labels: Optional[torch.Tensor] = None,
        task_tokens: Optional[torch.Tensor] = None,
        lang2vecs: Optional[torch.Tensor] = None,
        extract_embd: bool = False,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor],
        torch.Tensor,
    ]:
        """Forward through encoder layers and aggregate into utterance-level feature.

        Args:
            speech: (Batch, samples)
            speech_lengths: (Batch,)
            extract_embd: a flag which doesn't go through the classification
                head when set True
            lid_labels: (Batch, )
            one-hot speaker labels used in the train phase
            task_tokens: (Batch, )
            task tokens used in case of token-based trainings
        """

        if lid_labels is not None:
            assert speech.shape[0] == lid_labels.shape[0], (
                speech.shape,
                lid_labels.shape,
            )
        if task_tokens is not None:
            assert speech.shape[0] == task_tokens.shape[0], (
                speech.shape,
                task_tokens.shape,
            )
        batch_size = speech.shape[0]
        stats = dict()

        # 1. extract feats
        # Must transfer speech_lengths to extract_feats to get correct feat_lengths
        feats, feat_lengths, intermediate_lang2vec_preds = self.extract_feats(
            speech, speech_lengths, lid_labels
        )
        frame_level_feats = self.encode_frame(feats)

        # 2. aggregation into utterance-level
        utt_level_feat = self.pooling(frame_level_feats, feat_lengths=feat_lengths)

        # 3. (optionally) go through further projection(s)
        lang_embd = self.project_lang_embd(utt_level_feat)

        # 4. calculate loss
        # NOTE: if lid_labels is None, loss and accuracy are None
        if lang2vecs is not None:
            loss, accuracy, pred_lids, class_loss, lang2vec_loss = self.loss(
                lang_embd, lid_labels, lang2vecs
            )
            lang2vec_type = self.loss.lang2vec_type
            stats["class_loss"] = class_loss.detach()
            if (
                lang2vec_loss is not None
            ):  # lang2vec_loss is None when setting apply_last to False in the loss
                stats[f"{lang2vec_type}_loss_downstream"] = lang2vec_loss.detach()

            if (
                intermediate_lang2vec_preds is not None
                and self.inter_lang2vec_loss_weight > 0
                and self.apply_intermediate_lang2vec_loss
                and self.frontend.upstream.upstream.model.encoder.lang2vec_conditioning_layers
                is not None
            ):
                inter_lang2vec_losses = self._calc_intermediate_lang2vec_pred_loss(
                    intermediate_lang2vec_preds,
                    lang2vecs,
                )

                inter_lang2vec_loss_mean = 0.0

                for layer_idx, inter_lang2vec_loss in zip(
                    self.frontend.upstream.upstream.model.encoder.lang2vec_conditioning_layers,
                    inter_lang2vec_losses,
                ):
                    stats[f"inter_{lang2vec_type}_loss_layer{layer_idx}"] = (
                        inter_lang2vec_loss.detach()
                    )
                    inter_lang2vec_loss_mean += inter_lang2vec_loss

                inter_lang2vec_loss_mean /= len(inter_lang2vec_losses)
                stats[f"inter_{lang2vec_type}_loss_mean"] = (
                    inter_lang2vec_loss_mean.detach()
                )

                lang2vec_loss_all = 0.0
                if (
                    lang2vec_loss is not None
                ):  # which means do not apply lang2vec loss on the last layer
                    lang2vec_loss_all += (
                        1 - self.inter_lang2vec_loss_weight
                    ) * lang2vec_loss
                    lang2vec_loss_all += (
                        self.inter_lang2vec_loss_weight * inter_lang2vec_loss_mean
                    )
                else:
                    lang2vec_loss_all = inter_lang2vec_loss_mean

                stats[f"{lang2vec_type}_loss_all"] = lang2vec_loss_all.detach()
            else:
                lang2vec_loss_all = lang2vec_loss

            lid_class_loss_all = class_loss

            # NOTE(qingzheng): recaulculate the loss, loss fomula:
            # loss = (1 - lang2vec_weight) * class_loss+ lang2vec_weight * (
            #     (1 - inter_lang2vec_loss_weight) * lang2vec_loss +
            #     inter_lang2vec_loss_weight * inter_lang2vec_loss_mean
            # )
            loss = (
                1 - self.loss.lang2vec_weight
            ) * lid_class_loss_all + self.loss.lang2vec_weight * lang2vec_loss_all
        else:
            # NOTE(qingzheng): if use aamsoftmax_sc_topk_lang2vec loss but not
            # specify the lang2vec in preprocessor, it will jump to this branch,
            # but aamsoftmax_sc_topk_lang2vec default return 5 outputs, so use [:3]
            # to restrict here only retrieve the first 3 outputs
            outputs = self.loss(lang_embd, lid_labels)
            loss, accuracy, pred_lids = outputs[:3]
            if loss is not None:
                stats["class_loss"] = loss.detach()

        if extract_embd:
            return lang_embd, pred_lids

        stats["loss"] = loss.detach()
        if accuracy is not None:  # if not provide labels, accuracy is None
            stats["accuracy"] = accuracy.detach()

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def extract_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        lid_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = speech.shape[0]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else torch.ones(batch_size).int() * speech.shape[1]
        )

        # 1. extract feats
        if self.frontend is not None:
            feats, feat_lengths, intermediate_lang2vec_preds = self.frontend(
                speech, speech_lengths, lid_labels
            )
        else:
            feats = speech
            feat_lengths = None
            intermediate_lang2vec_preds = None

        # 2. apply augmentations
        if self.specaug is not None and self.training:
            feats, _ = self.specaug(feats, feat_lengths)

        # 3. normalize
        if self.normalize is not None:
            feats, _ = self.normalize(feats, feat_lengths)

        return feats, feat_lengths, intermediate_lang2vec_preds

    def _calc_intermediate_lang2vec_pred_loss(
        self,
        intermediate_lang2vec_preds: List[torch.Tensor],
        lang2vecs: torch.Tensor,
    ) -> List[torch.Tensor]:
        inter_lang2vec_losses = []
        for intermediate_lang2vec_pred in intermediate_lang2vec_preds:
            inter_lang2vec_loss = self.loss.lang2vec_loss(
                intermediate_lang2vec_pred,
                lang2vecs,
            )
            inter_lang2vec_losses.append(inter_lang2vec_loss)

        return inter_lang2vec_losses
