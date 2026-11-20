"""Shared base class and output record of the compression models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn


@dataclass
class CompressionOutput:
    """Return type for all compression model forward passes.

    Shapes (B = batch, T = frames, D = feature dim, S = segments):

    - ``boundary_soft`` (B, T): float boundary indicators (0 or 1); 1 marks
      the first frame of a new segment, and frame 0 is always 0.
    - ``segment_idx`` (B, T): int segment index per frame.
    - ``reconstructed_features`` (B, T, D): segment-averaged features
      upsampled back to frame resolution.
    - ``expected_length`` (B, 1): number of segments (boundaries + 1).

    Example:
        .. code-block:: python

            >>> compression = CosineSimilarityCompression()
            >>> output = compression(torch.randn(2, 50, 128), rate=0.5)
            >>> output.reconstructed_features.shape
            torch.Size([2, 50, 128])
            >>> output.expected_length.shape
            torch.Size([2, 1])
    """

    boundary_soft: torch.Tensor
    segment_idx: torch.Tensor
    reconstructed_features: torch.Tensor
    expected_length: torch.Tensor


class BaseCompressionModel(nn.Module, ABC):
    """Abstract base class for all compression models.

    A compression model segments a (B, T, D) frame sequence into
    variable-length segments controlled by a ``rate`` and replaces every
    frame with the mean of its segment, so the sequence keeps its frame
    resolution while carrying fewer distinct vectors.

    Subclasses must implement ``forward``, which segments an input feature
    sequence and returns a ``CompressionOutput``. The per-frame identity
    output (``rate=None``) and the segment-averaging math are shared here.

    Example:
        A minimal subclass that never merges frames:

        .. code-block:: python

            >>> class PerFrameCompression(BaseCompressionModel):
            ...     def forward(self, x, rate=None, padding_mask=None, **kwargs):
            ...         return self._build_no_rate_output(x, padding_mask)
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        rate: Optional[Union[float, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressionOutput:
        """Segment the input sequence and return compressed representations.

        Args:
            x: (B, T, D) input features.
            rate: Compression rate or budget (interpretation is
                model-specific): a float, or a (B,) tensor with one rate per
                batch item. When ``None``, every frame is its own segment
                (identity).
            padding_mask: (B, T) bool tensor; ``True`` marks padding positions.
            **kwargs: Model-specific options. The RVQ wrapper always passes
                ``anchor_boundary`` (a (B, T) boundary tensor or ``None``);
                models that do not support anchoring accept and ignore it.

        Returns:
            ``CompressionOutput`` for the batch.

        Example:
            .. code-block:: python

                >>> model = CosineSimilarityCompression()
                >>> output = model(torch.randn(2, 50, 128), rate=0.5)
                >>> output.boundary_soft[:, 0]
                tensor([0., 0.])
        """

    def _build_no_rate_output(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]
    ) -> CompressionOutput:
        """Return the per-frame segmentation used when ``rate`` is ``None``."""
        B, T, _ = x.shape
        device = x.device
        boundary = torch.ones(B, T, device=device, dtype=torch.long)
        boundary[:, 0] = 0
        if padding_mask is not None:
            boundary = boundary.masked_fill(padding_mask, 0)
        boundary_soft = boundary.float()
        segment_idx = torch.cumsum(boundary, dim=1)
        expected_length = boundary_soft.sum(dim=1, keepdim=True) + 1
        return CompressionOutput(
            boundary_soft=boundary_soft,
            segment_idx=segment_idx,
            reconstructed_features=x,
            expected_length=expected_length,
        )

    def _average_segments(
        self,
        x: torch.Tensor,
        segment_idx: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Average ``x`` within each segment, upsampled back to (B, T, D).

        Uses scatter_add + gather instead of a dense (B, S, T) assignment
        matrix, keeping peak memory at O(B*T*D).
        """
        B, T, D = x.shape
        device = x.device
        max_segments = int(segment_idx.max().item()) + 1

        x_masked = (
            x
            if padding_mask is None
            else x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        )

        # Scatter-add frames into segments -> (B, S, D)
        expanded_idx = segment_idx.unsqueeze(-1).expand(B, T, D)  # (B, T, D)
        segment_sum = torch.zeros(B, max_segments, D, device=device, dtype=x.dtype)
        segment_sum.scatter_add_(1, expanded_idx, x_masked)

        # Count frames per segment (use ones, zero out padding)
        ones = torch.ones(B, T, 1, device=device, dtype=x.dtype)
        if padding_mask is not None:
            ones = ones.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        segment_count = torch.zeros(B, max_segments, 1, device=device, dtype=x.dtype)
        segment_count.scatter_add_(1, segment_idx.unsqueeze(-1), ones)

        segment_means = segment_sum / (segment_count + 1e-6)  # (B, S, D)

        # Gather back to frame resolution -> (B, T, D)
        return segment_means.gather(1, expanded_idx)
