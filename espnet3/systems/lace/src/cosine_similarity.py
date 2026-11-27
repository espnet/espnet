"""Cosine-similarity threshold compression model."""

from typing import Optional, Union

import torch
import torch.nn.functional as F

from espnet3.systems.lace.src.compression import (
    BaseCompressionModel,
    CompressionOutput,
)


class CosineSimilarityCompression(BaseCompressionModel):
    """Threshold segmentation on consecutive-frame cosine similarity.

    A new segment starts at every frame whose cosine similarity to the
    previous frame is below ``rate``, i.e. the two frames are dissimilar
    enough that the latter is judged to belong to a new segment. ``rate``
    therefore acts directly as the similarity threshold: high rate (-> 1.0)
    means many boundaries and little compression; low rate (-> 0.0) means
    few boundaries and heavy compression.

    When ``anchor_boundary`` is given (anchored layers of the RVQ wrapper),
    boundaries are additionally restricted to the anchor set, so later
    layers can only segment at positions already chosen by earlier layers.

    Example:
        .. code-block:: python

            >>> model = CosineSimilarityCompression(max_tokens_per_group=4)
            >>> output = model(torch.randn(2, 50, 128), rate=0.5)
            >>> output.segment_idx.shape
            torch.Size([2, 50])
    """

    def __init__(self, max_tokens_per_group: Optional[int] = 4):
        """Store the segment-length cap.

        Args:
            max_tokens_per_group: Hard cap on segment length in frames
                (FlexiCodec's ``max_tokens_per_group``). A segment that
                would naturally be longer is split into chunks of this size,
                with a possibly shorter leftover at its end. ``None``
                disables the cap.
        """
        super().__init__()
        self.max_tokens_per_group = max_tokens_per_group

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_cosine_similarity(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Return the (B, T-1) cosine similarities between consecutive frames.

        Padded positions are filled with 1.0 so they never fall below the
        threshold (and ``_build_boundary`` masks them again anyway).
        """
        cosine_similarity = F.cosine_similarity(x[:, :-1], x[:, 1:], dim=-1)  # (B, T-1)
        if padding_mask is not None:
            cosine_similarity = cosine_similarity.masked_fill(padding_mask[:, 1:], 1.0)
        return cosine_similarity

    def _build_boundary(
        self,
        stop_tokens: torch.Tensor,
        batch_size: int,
        device: torch.device,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Prepend a t=0 zero and mask padding into a (B, T) boundary tensor."""
        boundary = torch.cat(
            [
                torch.zeros(batch_size, 1, device=device, dtype=stop_tokens.dtype),
                stop_tokens,
            ],
            dim=1,
        )
        if padding_mask is not None:
            boundary = boundary.masked_fill(padding_mask, 0)
        return boundary

    def _apply_max_segment_length(
        self, boundary: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Split segments longer than ``self.max_tokens_per_group`` frames.

        Forces a boundary at every position where the current segment would
        otherwise exceed the cap, and returns a boundary tensor of the same
        shape and dtype as ``boundary`` (1 marks the start of a new segment;
        position 0 stays 0). No-op when ``self.max_tokens_per_group`` is
        ``None``.

        Mirrors FlexiCodec's vectorised cummax trick: compute the in-segment
        index of every frame, then add a boundary wherever this index is a
        positive multiple of the cap. The forced boundaries are OR'd with the
        natural ones, so the natural splits are preserved exactly, and any
        forced boundary falling on a padded position is cleared.
        """
        if self.max_tokens_per_group is None:
            return boundary

        B, T = boundary.shape
        device = boundary.device

        # FlexiCodec convention: treat position 0 as an implicit segment
        # start when computing the in-segment index. The output boundary's
        # position 0 is left untouched (it stays 0 by convention).
        is_segment_start = boundary.clone().long()
        is_segment_start[:, 0] = 1

        arange_t = (
            torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, T)
        )
        segment_start_markers = arange_t * is_segment_start
        last_segment_start = torch.cummax(segment_start_markers, dim=1).values
        in_segment_idx = arange_t - last_segment_start

        # Force a boundary at every n*max_tokens position WITHIN a segment.
        # Skip in_segment_idx == 0 (already a boundary or implicit start).
        forced = (in_segment_idx > 0) & (
            (in_segment_idx % self.max_tokens_per_group) == 0
        )
        new_boundary = (boundary.bool() | forced).to(boundary.dtype)

        if padding_mask is not None:
            new_boundary = new_boundary.masked_fill(padding_mask, 0)
        return new_boundary

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        rate: Optional[Union[float, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_boundary: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressionOutput:
        """Segment the input sequence by thresholding frame similarity.

        Args:
            x: (B, T, D) input features.
            rate: Similarity threshold in (0, 1]; a float, a per-sample
                (B,) tensor, or None for trivial per-frame segmentation.
            padding_mask: (B, T) bool tensor; ``True`` marks padding positions.
            anchor_boundary: (B, T) binary tensor; when given, boundaries
                are restricted to these anchor positions.
            **kwargs: Ignored; accepted so all compression models share one
                call signature.

        Returns:
            ``CompressionOutput`` for the batch.

        Example:
            .. code-block:: python

                >>> model = CosineSimilarityCompression(max_tokens_per_group=None)
                >>> x = torch.ones(1, 4, 8)
                >>> x[0, 2] = -1.0
                >>> model(x, rate=0.5).boundary_soft
                tensor([[0., 0., 1., 1.]])
        """
        B, T, _ = x.shape
        device = x.device

        if rate is None:
            return self._build_no_rate_output(x, padding_mask)

        cosine_similarity = self._compute_cosine_similarity(x, padding_mask)
        if isinstance(rate, torch.Tensor) and rate.dim() == 1:
            rate = rate.unsqueeze(-1)  # (B,) -> (B, 1), broadcast over T-1
        stop_tokens = (cosine_similarity < rate).long()
        if anchor_boundary is not None:
            # Anchored layers may only place boundaries inside the anchor set.
            stop_tokens = stop_tokens * anchor_boundary[:, 1:].bool().long()
        boundary = self._build_boundary(stop_tokens, B, device, padding_mask)
        # Optional FlexiCodec-style hard cap on segment length. No-op when
        # ``self.max_tokens_per_group is None``.
        boundary = self._apply_max_segment_length(boundary, padding_mask)
        boundary_soft = boundary.float()
        segment_idx = torch.cumsum(boundary, dim=1)
        average_vectors = self._average_segments(x, segment_idx, padding_mask)
        expected_length = boundary_soft.sum(dim=1, keepdim=True) + 1

        return CompressionOutput(
            boundary_soft=boundary_soft,
            segment_idx=segment_idx,
            reconstructed_features=average_vectors,
            expected_length=expected_length,
        )
