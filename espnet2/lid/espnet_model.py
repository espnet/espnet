# Copyright 2025 Qingzheng Wang
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

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
from espnet2.spk.loss.abs_loss import AbsLoss
from espnet2.spk.pooling.abs_pooling import AbsPooling
from espnet2.spk.projector.abs_projector import AbsProjector
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


class ESPnetLIDModel(AbsESPnetModel):
    """ESPnet LID model with optional lang2vec conditioning and auxiliary prediction.

    Args:
        frontend: feature extractor module
        specaug: optional SpecAugment module
        normalize: optional normalization module
        encoder: frame-level encoder (e.g., ECAPA-TDNN)
        pooling: pooling layer (e.g., AttnStatPooling)
        projector: embedding projection module (e.g., RawNet3)
        loss: main loss function (e.g., AAMSoftmax)
        extract_feats_in_collect_stats: Retained for existing LID configurations.
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
        lang2vec_dim: Dimension of the recipe's per-language target vectors.
        lang2vec_loss: "mse" for continuous targets (e.g. geo), or "bce" for
            binary targets (e.g. syntax_knn), using logits before the sigmoid.
        lang2vec_weight: Weight of the auxiliary losses, disabled by default.
            The classification loss receives weight 1 - lang2vec_weight on
            supervised batches. Empty conditioning layers and zero weight retain
            the ordinary LID architecture and checkpoint keys.

    The classification head uses the configured LID loss. When lang2vec is
    enabled, a linear head predicts its targets from the final embedding, and
    additional heads predict the same targets at selected SSL layers. These
    predictions can condition the speech representation. Language metadata is
    used only as a training target, never as an inference input.

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
        extract_feats_in_collect_stats: Optional[bool] = None,
        # ======== Conditioning Modules ========
        encoder_condition: Optional[nn.ModuleDict] = None,
        pooling_condition: Optional[nn.ModuleDict] = None,
        projector_condition: Optional[nn.ModuleDict] = None,
        # ======== Conditioning Configs ========
        lang2vec_conditioning_layers: Optional[List[int]] = None,
        apply_intermediate_lang2vec_loss: bool = False,
        apply_intermediate_lang2vec_condition: bool = True,
        inter_lang2vec_loss_weight: float = 0.0,
        cutoff_gradient_from_backbone: Optional[bool] = True,
        cutoff_gradient_before_condproj: Optional[bool] = False,
        shared_conditioning_proj: Optional[bool] = False,
        lang2vec_dim: int = 299,
        lang2vec_loss: str = "mse",
        lang2vec_weight: float = 0.0,
    ):

        super().__init__()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.pooling = pooling
        self.projector = projector
        self.loss = loss

        if lang2vec_dim <= 0:
            raise ValueError("lang2vec_dim must be positive.")
        if not 0 <= lang2vec_weight <= 1:
            raise ValueError("lang2vec_weight must be in [0, 1].")
        if lang2vec_loss not in ("mse", "bce"):
            raise ValueError("lang2vec_loss must be 'mse' or 'bce'.")
        if not 0 <= inter_lang2vec_loss_weight <= 1:
            raise ValueError("inter_lang2vec_loss_weight must be in [0, 1].")
        self.lang2vec_conditioning_layers = sorted(lang2vec_conditioning_layers or [])
        self.lang2vec_weight = lang2vec_weight
        self.apply_intermediate_lang2vec_loss = apply_intermediate_lang2vec_loss
        self.inter_lang2vec_loss_weight = inter_lang2vec_loss_weight
        self.lang2vec_head = None
        self.lang2vec_loss = None
        if lang2vec_weight > 0 or self.lang2vec_conditioning_layers:
            embedding_size = (
                projector.output_size()
                if projector is not None
                else pooling.output_size()
            )
            self.lang2vec_head = nn.Linear(embedding_size, lang2vec_dim)
            self.lang2vec_loss = (
                nn.MSELoss() if lang2vec_loss == "mse" else nn.BCEWithLogitsLoss()
            )

        # Other frontends, including precomputed features, need no SSL wrappers.
        if not self.lang2vec_conditioning_layers:
            return
        if not isinstance(self.frontend, S3prlFrontend):
            raise ValueError("lang2vec conditioning layers require S3prlFrontend.")
        upstream = self.frontend.upstream.upstream.model
        num_layers = upstream.config.num_hidden_layers
        if len(set(self.lang2vec_conditioning_layers)) != len(
            self.lang2vec_conditioning_layers
        ) or any(i < 0 or i >= num_layers for i in self.lang2vec_conditioning_layers):
            raise ValueError(
                f"Conditioning layers must be unique indices in [0, {num_layers})."
            )
        # Conditioned layers must never be skipped during training.
        upstream.config.layerdrop = 0.0
        shared_projection = None
        if shared_conditioning_proj and apply_intermediate_lang2vec_condition:
            shared_projection = nn.Linear(lang2vec_dim, frontend.output_size())
        for index in self.lang2vec_conditioning_layers:
            key = str(index)
            projection = shared_projection
            if apply_intermediate_lang2vec_condition and projection is None:
                projection = nn.Linear(lang2vec_dim, frontend.output_size())
            if projection is not None and cutoff_gradient_from_backbone:
                projection.requires_grad_(False)
            wrapper = Lang2VecConditionedLayer(
                layer=upstream.encoder.layers[index],
                encoder=encoder_condition[key],
                pooling=pooling_condition[key],
                projector=projector_condition[key],
                prediction=nn.Linear(
                    projector_condition[key].output_size(), lang2vec_dim
                ),
                conditioning_layer=projection,
                detach_prediction=cutoff_gradient_before_condproj,
                detach_condition=cutoff_gradient_from_backbone,
            )

            self.frontend.wrap_upstream_layer(
                f"upstream.model.encoder.layers.{index}", wrapper
            )

    @property
    def conditioned_layers(self):
        """Return wrappers in execution order without registering duplicates."""
        if not self.lang2vec_conditioning_layers:
            return []
        layers = self.frontend.upstream.upstream.model.encoder.layers
        return [layers[i] for i in self.lang2vec_conditioning_layers]

    @typechecked
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        lid_labels: Optional[torch.Tensor] = None,
        extract_embd: bool = False,
        lang2vecs: Optional[torch.Tensor] = None,
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
            extract_embd: Return embeddings and predicted language labels.
            lid_labels: (Batch, )
                Integer language labels used during training.
            lang2vecs: (Batch, lang2vec_dim), auxiliary training targets.
            lang2vec_mask: (Batch, 1), optional mask for languages with auxiliary
                supervision. Excluded languages still contribute classification loss.
        """

        if lid_labels is not None:
            assert speech.shape[0] == lid_labels.shape[0], (
                speech.shape,
                lid_labels.shape,
            )
        batch_size = speech.shape[0]
        stats = dict()

        # 1. extract feats
        # Must transfer speech_lengths to extract_feats to get correct feat_lengths
        feats, feat_lengths, intermediate_lang2vec_preds = self.extract_feats(
            speech, speech_lengths, return_intermediate=True
        )
        frame_level_feats = self.encode_frame(feats)

        # 2. aggregation into utterance-level
        utt_level_feat = self.pooling(frame_level_feats, feat_lengths=feat_lengths)

        # 3. (optionally) go through further projection(s)
        lang_embd = self.project_lang_embd(utt_level_feat)

        class_loss, accuracy, pred_lids = self.loss(lang_embd, lid_labels)
        if extract_embd:
            return lang_embd, pred_lids
        if class_loss is None:
            raise ValueError("lid_labels are required when extract_embd is False")
        loss = class_loss
        if self.lang2vec_head is not None:
            stats["class_loss"] = class_loss.detach()
            # The trainer reduces each statistic in insertion order across ranks.
            # Keep the same keys even when one rank has no auxiliary targets; NaN
            # tells recursive_average to exclude that rank from auxiliary reporting.
            missing_stat = class_loss.detach().new_full((), float("nan"))
            stats["lang2vec_loss_downstream"] = missing_stat
            use_intermediate_loss = (
                self.apply_intermediate_lang2vec_loss
                and self.inter_lang2vec_loss_weight > 0
                and bool(self.lang2vec_conditioning_layers)
            )
            if use_intermediate_loss:
                for index in self.lang2vec_conditioning_layers:
                    stats[f"inter_lang2vec_loss_layer{index}"] = missing_stat
                stats["inter_lang2vec_loss_mean"] = missing_stat
                stats["lang2vec_loss_all"] = missing_stat
            if lang2vecs is not None and (lang2vec_mask is None or lang2vec_mask.any()):
                auxiliary = self.compute_lang2vec_loss(
                    self.lang2vec_head(lang_embd), lang2vecs, lang2vec_mask
                )
                stats["lang2vec_loss_downstream"] = auxiliary.detach()
                if use_intermediate_loss:
                    intermediate_losses = self._calc_intermediate_lang2vec_pred_loss(
                        intermediate_lang2vec_preds, lang2vecs, lang2vec_mask
                    )
                    for index, value in zip(
                        self.lang2vec_conditioning_layers, intermediate_losses
                    ):
                        stats[f"inter_lang2vec_loss_layer{index}"] = value.detach()
                    intermediate = torch.stack(intermediate_losses).mean()
                    stats["inter_lang2vec_loss_mean"] = intermediate.detach()
                    auxiliary = (
                        1 - self.inter_lang2vec_loss_weight
                    ) * auxiliary + self.inter_lang2vec_loss_weight * intermediate
                    stats["lang2vec_loss_all"] = auxiliary.detach()
                loss = (
                    1 - self.lang2vec_weight
                ) * class_loss + self.lang2vec_weight * auxiliary

        stats["loss"] = loss.detach()
        if accuracy is not None:  # if not provide labels, accuracy is None
            stats["accuracy"] = accuracy.detach()

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def extract_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]],
    ]:
        """Extract features, optionally returning auxiliary SSL predictions."""
        batch_size = speech.shape[0]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else torch.ones(batch_size, device=speech.device, dtype=torch.int)
            * speech.shape[1]
        )

        # Collect predictions as interCTC collects intermediate_outs. Clear both
        # before and after forward so graphs cannot leak between minibatches.
        for layer in self.conditioned_layers:
            layer.intermediate_out = None
        try:
            if self.frontend is not None:
                feats, feat_lengths = self.frontend(speech, speech_lengths)
            else:
                feats, feat_lengths = speech, None
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

        if return_intermediate:
            return feats, feat_lengths, intermediate_lang2vec_preds
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

    def _calc_intermediate_lang2vec_pred_loss(
        self,
        intermediate_lang2vec_preds: List[torch.Tensor],
        lang2vecs: torch.Tensor,
        lang2vec_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        inter_lang2vec_losses = []
        for intermediate_lang2vec_pred in intermediate_lang2vec_preds:
            inter_lang2vec_loss = self.compute_lang2vec_loss(
                intermediate_lang2vec_pred,
                lang2vecs,
                lang2vec_mask,
            )
            inter_lang2vec_losses.append(inter_lang2vec_loss)

        return inter_lang2vec_losses

    def compute_lang2vec_loss(self, prediction, target, mask=None):
        """Reduce auxiliary loss over rows with observed targets only."""
        if mask is not None:
            mask = mask.reshape(-1).bool()
            prediction, target = prediction[mask], target[mask]
        if prediction.shape[0] == 0:
            return prediction.sum()
        return self.lang2vec_loss(prediction, target)

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        lid_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self.extract_feats(speech, speech_lengths)
        return {"feats": feats}
