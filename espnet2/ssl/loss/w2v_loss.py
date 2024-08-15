# https://github.com/pytorch/audio/blob/main/examples/self_supervised_learning/losses/_wav2vec2_loss.py

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from espnet2.ssl.loss.abs_loss import AbsLoss


class Wav2vec2Loss(AbsLoss):
    def __init__(
        self,
        encoder_embed_dim: int,
        final_dim: int,
        layers=[-1],
    ):
        super().__init__()
        self.layers = layers

    def compute_contrastive_loss(
        x: Tensor,
        mask_indices: Tensor,
        targets: Tensor,
        neg_is_pos: Tensor,
        reduction: str = "none",
        logit_temp: float = 0.1,
    ):
        """
        Computes the contrastive loss used in Wav2Vec2 loss function.

        Args:
            x (Tensor): Input embeddings of shape `(batch_size, sequence_length, hidden_size)`.
            mask_indices (Tensor): Indices to mask negative samples of shape `(batch_size, sequence_length)`.
            targets (Tensor): Labels indicating positive samples.
                Tensor of shape `(num_negative + 1, batch, sequence_length, hidden_size)`.
            neg_is_pos (Tensor): Boolean tensor indicating whether negative samples should be treated as positives.
                Tensor of shape `(batch, sequence_length)`.
            reduction (str): Reduction type ("sum" or "none").
            logit_temp (float, optional): Temperature scaling factor for logits, defaults to 0.1.

        Returns:
            The computed contrastive loss and sample size
        """

        x = (
            x[mask_indices]
            .view(x.size(0), -1, x.size(-1))
            .unsqueeze(0)
            .expand(targets.shape)
        )
        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).float()
        logits /= logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        target = logits.new_zeros(
            logits.size(1) * logits.size(2), dtype=torch.long, device=logits.device
        )
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        loss = F.cross_entropy(
            logits,
            target,
            reduction=reduction,
        )
        sample_size = target.numel()
        return loss, sample_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ys_pad: torch.Tensor,
        mask_info,
        feature_penalty,
        feature_weight=10,
    ):
        """
        mask_info:
            - positives -> (batch_size, masked_sequence_length, hidden_size)
            - negatives -> (batch_size, masked_sequence_length, hidden_size)
            - mask_indices -> (batch_size, sequence_length)
        """

        neg_is_pos = (positives == negatives).all(-1)
        positives = positives.unsqueeze(0)
        targets = torch.cat([positives, negatives], dim=0)
