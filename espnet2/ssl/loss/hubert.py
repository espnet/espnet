#!/usr/bin/env python3

# Adapted parts from torchaudio/wav2vec2/components.py - LogitGenerator
# Copyright 2025 William Chen
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from espnet2.ssl.loss.abs_loss import AbsSSLLoss


class HuBERTLoss(AbsSSLLoss):
    def __init__(
        self,
        encoder_output_size: int,
        num_classes: int,
        final_dim: int,
        loss_type: str = "cross_entropy",
        layers: List = [-1],
        loss_weights: List = [1.0],
    ):
        """HuBERT MLM Loss

        Args:
            encoder_output_size (int): input dimension
            num_classes (int): vocab size
            final_dim (int): final projection dim
            loss_type (str): TODO, unused for now
            layers (List): encoder output layers for loss
            loss_weights (List): weight of each layer for loss
        """

        super(HuBERTLoss, self).__init__()

        self.layers = layers
        self.loss_weights = loss_weights

        assert len(self.layers) == len(self.loss_weights)

        self.util_attributes = ["mask"]
        self.required_inputs = [
            "encoder_output",
            "encoder_output_lengths",
            "text",
            "text_lengths",
            "mask_info",
        ]

        self.decoder = HuBERTDecoder(
            encoder_output_size,
            num_classes,
            final_dim,
        )

    def _compute_correct(
        self,
        logits,
        targets,
    ):
        if logits.numel() == 0:
            correct, count = 0, 0
        else:
            assert logits.dim() > 1, logits.shape
            max_idx = logits.argmax(-1)
            correct = (max_idx == targets).sum().item()
            count = max_idx.numel()
        return correct, count

    def forward(
        self,
        encoder_output: List,
        encoder_output_lengths: torch.Tensor = None,
        text: torch.Tensor = None,
        text_lengths: torch.Tensor = None,
        mask_info: Dict = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """HuBERT forward

        Args:
            encoder_output (List): List of encoded sequences (B, T, D) from each layer.
            encoder_output_lengths (Tensor): Lengths of batched encoder sequences (B,).
            text (Tensor): text targets (B, T)
            text_lengths (Tensor): Lengths of text targets (B,).
            mask_info (Dict): Contains masked/unmasked indices
        """

        mask_m = mask_info["mask_m"]
        mask_u = mask_info["mask_u"]
        y_m = text[mask_m]
        y_u = text[mask_u]

        total_loss = 0.0
        stats = {}
        for layer, weight in zip(self.layers, self.loss_weights):
            if layer < 0:
                layer = len(encoder_output) + layer
            x = encoder_output[layer]
            x_m, x_u = self.decoder(x, mask_m, mask_u)

            loss = (
                F.cross_entropy(x_m, y_m.long(), reduction="mean", ignore_index=-1)
                * weight
            )
            total_loss += loss
            stats[f"hubert_loss_m_{layer}"] = loss.detach().item()

            correct_m, count_m = self._compute_correct(x_m, y_m)
            correct_u, count_u = self._compute_correct(x_u, y_u)
            stats[f"hubert_acc_m_{layer}"] = 0 if count_m == 0 else correct_m / count_m
            stats[f"hubert_acc_u_{layer}"] = 0 if count_u == 0 else correct_m / count_m

        return total_loss, stats


class HuBERTDecoder(nn.Module):
    def __init__(
        self,
        encoder_embed_dim: int,
        num_classes: int,
        final_dim: int,
    ):
        """Generate the logits of masked and unmasked inputs.
        Args:
            encoder_embed_dim (int): The dimension of the transformer embedding output.
            num_classes (int): The number of classes in the labels.
            final_dim (int): Project final representations and targets to `final_dim`.
        """

        super().__init__()
        self.final_proj = torch.nn.Linear(encoder_embed_dim, final_dim)
        self.label_embeddings = torch.nn.Linear(final_dim, num_classes)

    def forward(
        self, x: torch.Tensor, mask_m: torch.Tensor, mask_u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (Tensor): The feature representation of the last transformer layer.
            mask_m (Tensor): The masked indices of dimension `[batch, frame]`.
            mask_u (Tensor): The unmasked indices of dimension `[batch, frame]`.

        Returns:
            Tensor: The logits of masked frames. `[masked_frame, final_dim]`.
            Tensor: The logits of unmasked frames. `[unmasked_frame, final_dim]`.
        """
        logit_temp = 0.1
        proj_x = self.final_proj(x)

        proj_x_m = proj_x[mask_m]
        logit_m = self.label_embeddings(proj_x_m) / logit_temp

        proj_x_u = proj_x[mask_u]
        logit_u = self.label_embeddings(proj_x_u) / logit_temp

        return logit_m, logit_u
