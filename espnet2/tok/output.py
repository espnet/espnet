"""Shared output types for trainable speech tokenizers."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SpeechTokenizerOutput:
    """Outputs of a trainable speech tokenizer.

    Args:
        continuous: Continuous frontend features of shape ``(B, T, D)``.
        assignment: Hard straight-through cluster assignments of shape
            ``(B, T, K)``.  The forward values are one-hot during training,
            while gradients propagate through their soft counterparts.
        token_ids: Integer cluster indices of shape ``(B, T)``.
        lengths: Valid token lengths of shape ``(B,)``.
        soft_assignment: Optional soft cluster assignments of shape
            ``(B, T, K)`` for analysis and auxiliary regularization.
        quantized_features: Optional differentiable quantized features of shape
            ``(B, T, D)``. A future VQ implementation can expose its
            straight-through vectors here so downstream losses can reach the
            frontend.
    """

    continuous: torch.Tensor
    assignment: torch.Tensor
    token_ids: torch.Tensor
    lengths: torch.Tensor
    soft_assignment: Optional[torch.Tensor] = None
    quantized_features: Optional[torch.Tensor] = None
