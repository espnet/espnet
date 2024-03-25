# github.com/pytorch/audio/blob/main/src/torchaudio/models/wav2vec2/components.py#L1075
# github.com/pytorch/audio/blob/main/examples/self_supervised_learning/losses/_hubert_loss.py

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from espnet2.ssl.loss.abs_loss import AbsLoss
from espnet.nets.pytorch_backend.nets_utils import th_accuracy


# https://github.com/pytorch/pytorch/issues/104564
def cosine_similarity(t1, t2, dim=-1, eps=1e-8):
    # get normalization value
    t1_div = torch.linalg.vector_norm(t1, dim=dim, keepdims=True)
    t2_div = torch.linalg.vector_norm(t2, dim=dim, keepdims=True)

    t1_div = t1_div.clone()
    t2_div = t2_div.clone()
    with torch.no_grad():
        t1_div.clamp_(math.sqrt(eps))
        t2_div.clamp_(math.sqrt(eps))

    # normalize, avoiding division by 0
    t1_norm = t1 / t1_div
    t2_norm = t2 / t2_div

    return (t1_norm * t2_norm).sum(dim=dim)


def _compute_logits(
    proj_x: Tensor,
    target: Tensor,
    label_embeddings: nn.Parameter,
) -> Tensor:
    """Compute the logits of the embeddings.
    Args:
        proj_x (Tensor): The projected masked representations `[batch, frame, final_dim]`.
        target (Tensor): The target Tensor  `[batch, frame, final_dim]`.
        label_embeddings (Parameter): The trainable embeddings of target  `[num_class, final_dim]`.

    Returns:
        (Tensor): The logits of the inputs.
    """
    logit_temp = 0.1
    pos = torch.index_select(label_embeddings, 0, target.long())
    negs = label_embeddings.unsqueeze(1).expand(-1, proj_x.size(0), -1)
    neg_is_pos = (pos == negs).all(-1)
    pos = pos.unsqueeze(0)
    targets = torch.cat([pos, negs], dim=0)
    logits = torch.cosine_similarity(proj_x.float(), targets.float(), dim=-1).type_as(
        proj_x
    )
    logits /= logit_temp
    if neg_is_pos.any():
        logits[1:][neg_is_pos] = float("-inf")
    logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
    return logits


class LogitGenerator(nn.Module):
    """Generate the logits of masked and unmasked inputs.
    Args:
        encoder_embed_dim (int): The dimension of the transformer embedding output.
        num_classes (int): The number of classes in the labels.
        final_dim (int): Project final representations and targets to `final_dim`.
        skip_masked (bool): If True, skip computing losses over masked frames.
        skip_nomask (bool): If True, skip computing losses over unmasked frames.
    """

    def __init__(
        self,
        encoder_embed_dim: int,
        num_classes: int,
        final_dim: int,
        skip_masked: bool,
        skip_nomask: bool,
    ):
        super().__init__()
        self.label_embeddings = nn.Parameter(torch.FloatTensor(num_classes, final_dim))
        torch.nn.init.uniform_(self.label_embeddings)
        self.final_proj = torch.nn.Linear(encoder_embed_dim, final_dim)
        self.skip_masked = skip_masked
        self.skip_nomask = skip_nomask

    def forward(
        self, x: Tensor, label: Tensor, mask_m: Tensor, mask_u: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): The feature representation of the last transformer layer.
            label (Tensor): The label Tensor of dimension `[batch, frame]`.
            mask_m (Tensor): The masked indices of dimension `[batch, frame]`.
            mask_u (Tensor): The unmasked indices of dimension `[batch, frame]`.

        Returns:
            Tensor: The logits of masked frames `[masked_frame, final_dim]`.
            Tensor: The logits of unmasked frames `[unmasked_frame, final_dim]`.
        """
        proj_x = self.final_proj(x)

        if self.skip_masked:
            logit_m = None
        else:
            proj_x_m = proj_x[mask_m]
            label_m = label[mask_m]
            logit_m = _compute_logits(proj_x_m, label_m, self.label_embeddings)

        if self.skip_nomask:
            logit_u = None
        else:
            proj_x_u = proj_x[mask_u]
            label_u = label[mask_u]
            logit_u = _compute_logits(proj_x_u, label_u, self.label_embeddings)

        return logit_m, logit_u


class HuBERTLoss(AbsLoss):
    def __init__(
        self,
        encoder_embed_dim: int,
        num_classes: int,
        final_dim: int,
        masked_weight: float = 1.0,
        unmasked_weight: float = 0.0,
        layers=[-1],
    ):
        super().__init__()
        self.masked_weight = masked_weight
        self.unmasked_weight = unmasked_weight

        self.layers = layers

        self.decoder = LogitGenerator(
            encoder_embed_dim,
            num_classes,
            final_dim,
            False,
            False,
        )

    def _compute_correct(
        self,
        logits,
    ):
        if logits.numel() == 0:
            corr, count = 0, 0
        else:
            assert logits.dim() > 1, logits.shape
            max = logits.argmax(-1) == 0
            min = logits.argmin(-1) == 0
            both = max & min
            corr = max.long().sum().item() - both.long().sum().item()
            count = max.numel()
        return corr, count

    def _calc_hubert_loss(
        self,
        logit_m: Optional[torch.Tensor],
        logit_u: Optional[torch.Tensor],
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute the cross-entropy loss on HuBERT masked and non-masked logits.

        Args:
            logit_m (Tensor or None): The masked logit Tensor of dimension
                `(masked_frames, final_dim)`.
            logit_u (Tensor or None): The non-masked logit Tensor of dimension
                `(unmasked_frames, final_dim)`.
            feature_penalty (Tensor): The feature mean value for additional penalty
                loss.
            masked_weight (float, optional): The weight for masked cross-entropy loss
                (Default: ``1.0``).
            unmasked_weight (float, optional): The weight for non-masked cross-entropy
                loss (Default: ``0.0``).
            feature_weight (float, optional): The weight for feature penalty loss
                (Default: ``10.0``).
            reduction (str, optional): The reduction method for cross-entropy loss
                (Default: ``"sum"``).
        Ref:
            torchaudio: examples/hubert/loss/hubert_loss.py
        """
        loss_m = 0
        if logit_m is not None:
            target_m = torch.zeros(
                logit_m.shape[0], dtype=torch.long, device=logit_m.device
            )
            loss_m = torch.nn.functional.cross_entropy(
                logit_m, target_m, reduction=reduction
            )
            loss_m *= self.masked_weight

        loss_u = 0
        if logit_u is not None:
            target_u = torch.zeros(
                logit_u.shape[0], dtype=torch.long, device=logit_m.device
            )
            loss_u = torch.nn.functional.cross_entropy(
                logit_u, target_u, reduction=reduction
            )
            loss_u *= self.unmasked_weight

        return loss_m, loss_u

    def forward(
        self,
        xs_pad: torch.Tensor,
        ys_pad: torch.Tensor,
        mask_info,
        feature_penalty,
        feature_weight=10,
    ):

        mask_m = mask_info["mask_m"]
        mask_u = mask_info["mask_u"]

        layer_loss = 0
        for layer in self.layers:
            logit_m, logit_u = self.decoder(xs_pad[layer], ys_pad, mask_m, mask_u)

            loss_m, loss_u = self._calc_hubert_loss(
                logit_m,
                logit_u,
            )

            layer_loss = layer_loss + loss_m + loss_u

        total_loss = feature_penalty * feature_weight * logit_m.shape[0] + layer_loss

        # log accuracies of masked and unmasked frames
        correct_m, count_m = self._compute_correct(logit_m)
        correct_u, count_u = self._compute_correct(logit_u)

        # for now only log stats of last layer
        stats = dict(
            hubert_loss=total_loss.detach(),
            hubert_correct_m=correct_m,
            hubert_count_m=count_m,
            hubert_acc_m=0 if count_m == 0 else correct_m / count_m,
            hubert_correct_u=correct_u,
            hubert_count_u=count_u,
            hubert_acc_u=0 if count_u == 0 else correct_u / count_u,
        )

        return total_loss, stats
