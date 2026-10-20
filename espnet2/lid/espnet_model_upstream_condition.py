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
#   https://arxiv.org/pdf/2508.17148
#
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.lid.conditioned_layer import Lang2VecConditionedLayer
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
        cutoff_gradient_from_backbone: if True, detach the projected residual
            before adding it to the SSL hidden states AND freeze the conditioning
            projection parameters. The SSL backbone and auxiliary prediction
            heads remain trainable.
        cutoff_gradient_before_condproj: if True, detach the predicted lang2vec
            vector before projection. This alone does not freeze the projection.
        shared_conditioning_proj: if True, share conditioning projection across
                                  layers

    Reference:
        Geolocation-Aware Robust Spoken Language Identification
        https://arxiv.org/pdf/2508.17148
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

        assert isinstance(self.frontend, S3prlFrontend), (
            f"ESPnetLIDUpstreamConditionModel only supports S3prlFrontend "
            f"as frontend, but got {type(self.frontend)}"
        )

        if not isinstance(self.loss, AAMSoftmaxSCTopKLang2Vec):
            raise ValueError("Upstream conditioning requires AAMSoftmaxSCTopKLang2Vec.")
        if lang2vec_conditioning_layers:
            if any(
                value is None
                for value in (
                    self.loss.lang2vec_dim,
                    self.loss.lang2vec_type,
                    self.loss.lang2vec_weight,
                )
            ):
                raise ValueError(
                    "Conditioning requires lang2vec_dim, lang2vec_type "
                    "and lang2vec_weight in loss_conf."
                )
            num_layers = self.frontend.upstream.upstream.model.config.num_hidden_layers
            if len(set(lang2vec_conditioning_layers)) != len(
                lang2vec_conditioning_layers
            ) or any(i < 0 or i >= num_layers for i in lang2vec_conditioning_layers):
                raise ValueError(
                    f"Conditioning layers must be unique indices in [0, {num_layers})."
                )
        if not 0 <= inter_lang2vec_loss_weight <= 1:
            raise ValueError("inter_lang2vec_loss_weight must be in [0, 1].")

        self.lang2vec_conditioning_layers = sorted(lang2vec_conditioning_layers or [])
        upstream = self.frontend.upstream.upstream.model
        # Conditioned layers must never be skipped during training.
        upstream.config.layerdrop = 0.0
        shared_projection = None
        if shared_conditioning_proj and apply_intermediate_lang2vec_condition:
            shared_projection = nn.Linear(
                self.loss.lang2vec_dim, frontend.output_size()
            )
        for index in self.lang2vec_conditioning_layers:
            key = str(index)
            projection = shared_projection
            if apply_intermediate_lang2vec_condition and projection is None:
                projection = nn.Linear(self.loss.lang2vec_dim, frontend.output_size())
            if projection is not None and cutoff_gradient_from_backbone:
                projection.requires_grad_(False)
            upstream.encoder.layers[index] = Lang2VecConditionedLayer(
                layer=upstream.encoder.layers[index],
                encoder=encoder_condition[key],
                pooling=pooling_condition[key],
                projector=projector_condition[key],
                prediction=nn.Linear(
                    projector_condition[key].output_size(), self.loss.lang2vec_dim
                ),
                conditioning_layer=projection,
                detach_prediction=cutoff_gradient_before_condproj,
                detach_condition=cutoff_gradient_from_backbone,
            )

        # Update the stock frontend's initialization snapshot for the wrappers so
        # reload_pretrained_parameters() still works with an explicit --init.
        pretrained = self.frontend.pretrained_params
        updated = {}
        for name, value in self.frontend.upstream.state_dict().items():
            original_name = name
            for index in self.lang2vec_conditioning_layers:
                prefix = f"upstream.model.encoder.layers.{index}.layer."
                if name.startswith(prefix):
                    original_name = prefix[:-6] + name[len(prefix) :]
                    break
            updated[name] = (
                pretrained[original_name]
                if original_name in pretrained
                else value.detach().clone()
            )
        self.frontend.pretrained_params = updated
        self.apply_intermediate_lang2vec_loss = apply_intermediate_lang2vec_loss
        self.inter_lang2vec_loss_weight = inter_lang2vec_loss_weight

    @property
    def conditioned_layers(self):
        """Return wrappers in execution order without registering duplicates."""
        layers = self.frontend.upstream.upstream.model.encoder.layers
        return [layers[i] for i in self.lang2vec_conditioning_layers]

    @typechecked
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        lid_labels: Optional[torch.Tensor] = None,
        task_tokens: Optional[torch.Tensor] = None,
        lang2vecs: Optional[torch.Tensor] = None,
        extract_embd: bool = False,
        lang2vec_mask: Optional[torch.Tensor] = None,
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
            lang2vec_mask: (Batch, 1), optional mask for languages with auxiliary
                supervision. Excluded languages still contribute classification loss.
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
        if lang2vecs is not None and (lang2vec_mask is None or lang2vec_mask.any()):
            class_loss, accuracy, pred_lids, lang2vec_loss = self.loss(
                lang_embd, lid_labels, lang2vecs, lang2vec_mask
            )
            lang2vec_type = self.loss.lang2vec_type
            stats["class_loss"] = class_loss.detach()
            if lang2vec_loss is not None:
                stats[f"{lang2vec_type}_loss_downstream"] = lang2vec_loss.detach()

            if (
                intermediate_lang2vec_preds is not None
                and self.inter_lang2vec_loss_weight > 0
                and self.apply_intermediate_lang2vec_loss
                and self.lang2vec_conditioning_layers
            ):
                inter_lang2vec_losses = self._calc_intermediate_lang2vec_pred_loss(
                    intermediate_lang2vec_preds,
                    lang2vecs,
                    lang2vec_mask,
                )

                inter_lang2vec_loss_mean = 0.0

                for layer_idx, inter_lang2vec_loss in zip(
                    self.lang2vec_conditioning_layers,
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
                if lang2vec_loss is not None:
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

            # Weight classification and auxiliary supervision once, here.
            if lang2vec_loss_all is None:
                loss = class_loss
            else:
                loss = (
                    1 - self.loss.lang2vec_weight
                ) * class_loss + self.loss.lang2vec_weight * lang2vec_loss_all
        else:
            loss, accuracy, pred_lids, _ = self.loss(lang_embd, lid_labels)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        batch_size = speech.shape[0]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else speech.new_full((batch_size,), speech.shape[1], dtype=torch.long)
        )

        # Collect predictions as interCTC collects intermediate_outs. Clear both
        # before and after forward so graphs cannot leak between minibatches.
        for layer in self.conditioned_layers:
            layer.intermediate_out = None
        try:
            feats, feat_lengths = self.frontend(speech, speech_lengths)
            intermediate_lang2vec_preds = tuple(
                layer.intermediate_out for layer in self.conditioned_layers
            )
            if any(pred is None for pred in intermediate_lang2vec_preds):
                raise RuntimeError("An SSL conditioning layer did not execute.")
        finally:
            for layer in self.conditioned_layers:
                layer.intermediate_out = None

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
        lang2vec_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        inter_lang2vec_losses = []
        for intermediate_lang2vec_pred in intermediate_lang2vec_preds:
            inter_lang2vec_loss = self.loss.compute_lang2vec_loss(
                intermediate_lang2vec_pred,
                lang2vecs,
                lang2vec_mask,
            )
            inter_lang2vec_losses.append(inter_lang2vec_loss)

        return inter_lang2vec_losses

    def collect_feats(self, speech, speech_lengths, **kwargs):
        feats, _, _ = self.extract_feats(speech, speech_lengths)
        return {"feats": feats}
