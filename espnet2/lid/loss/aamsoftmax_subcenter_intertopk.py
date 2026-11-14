"""Soft-target and multi-label variants of the sub-center Inter-TopK ArcFace loss.

Both keep the constructor arguments and state-dict keys (``weight``) of the
speaker loss ``ArcMarginProduct_intertopk_subcenter`` and apply the angular
margin to every target class instead of a single one.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.spk.loss.aamsoftmax import sine_from_cosine
from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
    ArcMarginProduct_intertopk_subcenter,
)


def _subcenter_cosine(
    module: ArcMarginProduct_intertopk_subcenter, input: torch.Tensor
) -> torch.Tensor:
    """Return the cosine to the closest sub-center of each class, (B, nclasses)."""
    cosine = F.linear(F.normalize(input), F.normalize(module.weight))
    cosine = torch.reshape(cosine, (-1, module.out_features, module.K))
    cosine, _ = torch.max(cosine, 2)
    return cosine


def _margin_logits(
    module: ArcMarginProduct_intertopk_subcenter,
    input: torch.Tensor,
    cosine: torch.Tensor,
    one_hot: torch.Tensor,
) -> torch.Tensor:
    """Return the scaled ArcFace + Inter-TopK logits, as in the upstream forward.

    ``one_hot`` (same shape and dtype as ``cosine``) may have several 1s per
    row; each receives the angular margin. ``module.k_top`` should not exceed
    the number of non-target classes.
    """
    sine = sine_from_cosine(cosine)
    phi = cosine * module.cos_m - sine * module.sin_m  # cos(theta + m)
    phi_mp = cosine * module.cos_mp + sine * module.sin_mp  # cos(theta - mp)
    if module.easy_margin:
        phi = torch.where(cosine > 0, phi, cosine)
    else:
        phi = torch.where(cosine > module.th, phi, cosine - module.mmm)

    if module.k_top > 0:
        _, top_k_index = torch.topk(cosine - 2 * one_hot, module.k_top)
        top_k_one_hot = input.new_zeros(cosine.size()).scatter_(1, top_k_index, 1)
        output = (
            (one_hot * phi)
            + (top_k_one_hot * phi_mp)
            + ((1.0 - one_hot - top_k_one_hot) * cosine)
        )
    else:
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

    output *= module.scale
    return output


def _dense_target(label: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Return ``label`` as a dense target shaped and typed like ``ref``.

    Integer class ids of shape (B,) or (B, 1) become one-hot rows; a dense
    target of shape (B, nclasses) is only cast.
    """
    if label.dim() == 2 and label.size(1) == ref.size(1):
        target = label.to(dtype=ref.dtype)
    else:
        if label.dim() == 2:
            label = label.squeeze(1)
        target = torch.zeros_like(ref).scatter_(1, label.long().view(-1, 1), 1.0)
    if target.shape != ref.shape:
        raise ValueError(f"target shape {target.shape} does not match {ref.shape}")
    return target


class ArcMarginProduct_intertopk_subcenter_softtarget(
    ArcMarginProduct_intertopk_subcenter
):
    """ArcFace loss with sub-center and Inter-TopK penalty for soft targets.

    Accepts hard integer labels or a dense target distribution of shape
    (batch_size, nclasses), e.g. 0.5/0.5 for an utterance with two languages.
    Dense targets are normalized per row, every class with positive mass gets
    the angular margin, and the loss is KL(target || softmax(logits)). Hard
    labels give the same loss as ``ArcMarginProduct_intertopk_subcenter``.
    """

    def forward(
        self, input: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Compute the soft-target loss.

        Args:
            input: Input embeddings, shape (batch_size, embedding_dim)
            label: Class ids of shape (batch_size,) or (batch_size, 1), or a
                dense target of shape (batch_size, nclasses), or None

        Returns:
            loss: KL(target || softmax) with batchmean reduction, or None
            accuracy: Fraction of rows whose top-1 prediction has positive
                target mass, or None
            pred_lids: Predicted class indices, shape (batch_size,)
        """
        cosine = _subcenter_cosine(self, input)
        preds = torch.argmax(cosine, dim=1)
        if label is None:
            return None, None, preds

        target = _dense_target(label, cosine)
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        positive_mask = target > 0
        accuracy = positive_mask.gather(1, preds.view(-1, 1)).float().mean()

        output = _margin_logits(self, input, cosine, positive_mask.to(cosine.dtype))
        loss = F.kl_div(F.log_softmax(output, dim=1), target, reduction="batchmean")
        return loss, accuracy, preds


class ArcMarginProduct_intertopk_subcenter_multilabel(
    ArcMarginProduct_intertopk_subcenter
):
    """Sub-center cosine classifier trained with multi-label BCE.

    Training logits apply the angular margin to every positive label and the
    Inter-TopK penalty to the hardest negatives. Inference scores are the
    label-independent scaled cosines; predictions are their argmax.

    Args:
        nout, nclasses, scale, margin, easy_margin, K, mp, k_top, do_lm: As in
            ``ArcMarginProduct_intertopk_subcenter``
        threshold: Sigmoid probability at which a class counts as predicted in
            the reported (subset) accuracy
        pos_weight: Positive class weight of the BCE loss, shared by all classes
    """

    def __init__(
        self,
        nout: int,
        nclasses: int,
        scale: float = 32.0,
        margin: float = 0.2,
        easy_margin: bool = False,
        K: int = 3,
        mp: float = 0.06,
        k_top: int = 5,
        do_lm: bool = False,
        threshold: float = 0.5,
        pos_weight: Optional[float] = None,
    ):
        """Initialize the head; see the class docstring for the arguments."""
        super().__init__(
            nout=nout,
            nclasses=nclasses,
            scale=scale,
            margin=margin,
            easy_margin=easy_margin,
            K=K,
            mp=mp,
            k_top=k_top,
            do_lm=do_lm,
        )
        self.threshold = float(threshold)
        weight = None
        if pos_weight is not None:
            weight = torch.full((nclasses,), float(pos_weight))
        # Saved in checkpoints as the persistent buffer ``bce.pos_weight``
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward(
        self, input: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Compute the multi-label BCE loss.

        Args:
            input: Input embeddings, shape (batch_size, embedding_dim)
            label: Class ids of shape (batch_size,) or (batch_size, 1), or a
                multi-hot target of shape (batch_size, nclasses), or None

        Returns:
            loss: BCE-with-logits loss (mean over batch and classes), or None
            accuracy: Exact-match (subset) accuracy of the thresholded
                inference scores, or None
            pred_lids: Argmax of the inference scores, shape (batch_size,)
        """
        cosine = _subcenter_cosine(self, input)
        scores = cosine * self.scale
        preds = torch.argmax(scores, dim=1)
        if label is None:
            return None, None, preds

        target = _dense_target(label, cosine)
        logits = _margin_logits(self, input, cosine, (target > 0).to(cosine.dtype))
        loss = self.bce(logits, target)

        pred = torch.sigmoid(scores) >= self.threshold
        accuracy = pred.eq(target >= 0.5).all(dim=1).float().mean()
        return loss, accuracy, preds
