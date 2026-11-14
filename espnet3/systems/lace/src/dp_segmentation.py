"""Dynamic-programming segmentation compression model."""

import math
from typing import Optional, Tuple, Union

import torch

from espnet3.systems.lace.src.compression import (
    BaseCompressionModel,
    CompressionOutput,
)


class DPSegmentationCompression(BaseCompressionModel):
    """Optimal frame boundary detection via Dynamic Programming.

    Based on CodecSlime (arXiv:2506.21074). Finds the segmentation that
    minimises total L2 distortion between original frames and their
    segment-mean reconstruction:

        L(j, s) = Σ_{t=j−s+1}^{j} ‖h_t − mean(h[j−s+1 … j])‖₂

    The DP maximises the cumulative negative distortion:

        d[j, i] = max_{1 ≤ s ≤ T}  { d[j−s, i−1] − L(j, s) }
        d[0, 0] = 0

    targeting exactly T' = clamp(round(rate × T), 1, T) segments.

    Optimal boundaries are recovered by backtracking through stored parent
    pointers. The cost table and DP run under ``torch.no_grad()`` because
    boundaries are hard (non-differentiable); gradients still flow through
    the subsequent segment-averaging step.

    Example:
        .. code-block:: python

            >>> model = DPSegmentationCompression(max_span=4)
            >>> output = model(torch.randn(2, 50, 128), rate=0.5)
            >>> output.expected_length
            tensor([[25.], [25.]])
    """

    def __init__(
        self,
        max_span: Optional[int] = 4,
    ):
        """Store the segment-length bound.

        Args:
            max_span: Hard upper bound U on the segment length (the paper
                uses 4). It is an algorithmic parameter, part of the DP
                constraint set ``S = {s | Σsᵢ = T, 1 ≤ sᵢ ≤ U}``, so keep it
                the same between training and inference. Larger values allow
                more aggressive compression but grow the cost table from
                O(T·U·D) toward O(T²·D). ``None`` removes the constraint
                (U = T).
        """
        super().__init__()
        self.max_span = max_span

    # ------------------------------------------------------------------
    # Cost-table computation (vectorised over windows for each span s)
    # ------------------------------------------------------------------

    def _compute_cost_table(self, x: torch.Tensor) -> torch.Tensor:
        """Return the (T, U) L2 distortion of every (end-frame, span) pair.

        For ``x`` of shape (T, D), ``cost[j, s−1]`` is the cost of a segment
        of length *s* whose last frame is 0-indexed frame *j*:

            cost[j, s−1] = Σ_{t=j−s+1}^{j} ‖h_t − mean(h[j−s+1 … j])‖₂

        with U = min(max_span, T), or T when ``max_span`` is ``None``.
        Invalid entries (j < s−1) are ``+inf``.
        """
        T, D = x.shape
        U = min(self.max_span, T) if self.max_span is not None else T
        device, dtype = x.device, x.dtype

        cost = torch.full((T, U), float("inf"), device=device, dtype=dtype)

        for s in range(1, U + 1):
            num_windows = T - s + 1
            if num_windows <= 0:
                break

            # Build sliding-window index: idx[w, t] = w + t
            window_idx = torch.arange(num_windows, device=device)  # (num_windows,)
            offset_idx = torch.arange(s, device=device)  # (s,)
            idx = window_idx.unsqueeze(1) + offset_idx.unsqueeze(0)  # (num_windows, s)

            frames = x[idx]  # (num_windows, s, D)
            segment_mean = frames.mean(dim=1, keepdim=True)  # (num_windows, 1, D)
            l2_sum = (frames - segment_mean).norm(dim=-1).sum(dim=1)  # (num_windows,)

            # 0-indexed end frames for windows of length s: s−1, s, …, T−1
            j_vals = torch.arange(s - 1, T, device=device)  # (num_windows,)
            cost[j_vals, s - 1] = l2_sum

        return cost  # (T, U)

    # ------------------------------------------------------------------
    # DP forward pass + backtracking
    # ------------------------------------------------------------------

    def _run_dp_forward(
        self,
        cost: torch.Tensor,
        T: int,
        max_segments: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the forward DP pass over the (j, i) states up to ``max_segments``.

        Returns ``(d, parent)``, both (T+1, max_segments+1): ``d[j, i]`` is
        the best objective (negative distortion) covering j frames with i
        segments, ``-inf`` where infeasible, and ``parent[j, i]`` is the
        1-indexed span of the last segment chosen to reach state (j, i).
        """
        U = cost.shape[1]
        device = cost.device
        neg_inf = float("-inf")

        d = torch.full(
            (T + 1, max_segments + 1), neg_inf, device=device, dtype=cost.dtype
        )
        d[0, 0] = 0.0

        parent = torch.zeros((T + 1, max_segments + 1), dtype=torch.long, device=device)

        # For span s_idx (0-indexed, span length = s_idx+1):
        #   candidates[s_idx, j] = d_prev[j - s_idx] - cost[j, s_idx]
        # Build a padded view of d_prev so shifting is a plain slice:
        #   padded = [-inf×U, d_prev[0..T-1]]  (length U+T)
        #   padded[U - s_idx + j] = d_prev[j - s_idx]  when j >= s_idx (else -inf)
        neg_inf_pad = cost.new_full((U,), neg_inf)

        for i in range(1, max_segments + 1):
            d_prev = d[:, i - 1]  # (T+1,)
            padded = torch.cat([neg_inf_pad, d_prev[:T]])  # (U+T,)

            shifted = torch.stack(
                [padded[U - s_idx : U - s_idx + T] for s_idx in range(U)], dim=0
            )  # (U, T)

            candidates = shifted - cost.T

            valid = (shifted > neg_inf) & (cost.T < float("inf"))
            candidates = torch.where(
                valid, candidates, cost.new_full(candidates.shape, neg_inf)
            )

            best_vals, best_s_idx = candidates.max(dim=0)
            d[1 : T + 1, i] = best_vals
            parent[1 : T + 1, i] = best_s_idx + 1

        return d, parent

    def _backtrack_dp(
        self,
        parent: torch.Tensor,
        T: int,
        num_segments: int,
    ) -> torch.Tensor:
        """Return the (T,) segment index backtracked from state (T, num_segments)."""
        device = parent.device
        segment_lengths: list[int] = []
        j, segment_count = T, num_segments
        while segment_count > 0:
            s = int(parent[j, segment_count].item())
            segment_lengths.append(s)
            j -= s
            segment_count -= 1
        segment_lengths.reverse()

        segment_idx = torch.zeros(T, dtype=torch.long, device=device)
        position = 0
        for segment_id, length in enumerate(segment_lengths):
            segment_idx[position : position + length] = segment_id
            position += length
        return segment_idx

    def _run_dp(
        self,
        cost: torch.Tensor,
        T: int,
        num_segments: int,
    ) -> torch.Tensor:
        """Return the (T,) optimal segment index for exactly ``num_segments`` segments.

        ``cost`` is the (T, U) table from ``_compute_cost_table`` and
        ``num_segments`` is already clamped for feasibility by the caller;
        the uniform fallback only guards an infeasible state.
        """
        d, parent = self._run_dp_forward(cost, T, num_segments)
        if d[T, num_segments].item() == float("-inf"):
            # Should not happen after feasibility clamping, but guard anyway.
            return self._segment_uniformly(T, num_segments, cost.device)
        return self._backtrack_dp(parent, T, num_segments)

    # ------------------------------------------------------------------
    # Anchor-restricted DP: boundaries are a subset of anchor positions
    # ------------------------------------------------------------------

    def _run_dp_anchor_subset(
        self,
        x_b: torch.Tensor,
        anchor_positions: torch.Tensor,
        num_segments: int,
    ) -> torch.Tensor:
        """Run the DP with boundaries restricted to the anchor positions.

        ``x_b`` is one unpadded (valid_length, D) sequence and
        ``anchor_positions`` its (K,) sorted anchors in [1, valid_length).
        Only segments that start and end on an anchor (or on the sequence
        ends) keep a finite cost, so the optimal segmentation can cut only
        at a subset of ``anchor_positions``.
        """
        valid_length = x_b.shape[0]
        device = x_b.device

        # Augmented candidate positions: 0, a_1, …, a_K, valid_length
        positions = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                anchor_positions,
                torch.tensor([valid_length], dtype=torch.long, device=device),
            ]
        )  # (M,) with M = K + 2

        cost_table = self._compute_cost_table(x_b)  # (valid_length, max_span)

        # Initialize everything as invalid (infinite cost)
        constrained_cost = torch.full_like(cost_table, float("inf"))

        # Create a matrix of all possible spans: (M, M)
        p_end = positions.view(-1, 1)  # (M, 1)
        p_start = positions.view(1, -1)  # (1, M)
        spans = p_end - p_start  # (M, M) matrix of all spans

        # Build a boolean mask for valid spans
        mask = (spans > 0) & (spans <= cost_table.shape[1])
        if self.max_span is not None:
            mask &= spans <= self.max_span

        # Extract the valid end positions and valid spans
        valid_ends = p_end.expand_as(spans)[mask]
        valid_spans = spans[mask]

        # Vectorized assignment: copy only valid anchor-to-anchor costs
        constrained_cost[valid_ends - 1, valid_spans - 1] = cost_table[
            valid_ends - 1, valid_spans - 1
        ]

        # Run constrained DP
        segment_idx_b = self._run_dp(constrained_cost, valid_length, num_segments)
        return segment_idx_b

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _segment_uniformly(
        T: int, num_segments: int, device: Union[str, torch.device]
    ) -> torch.Tensor:
        """Return the (T,) segment index spreading T frames evenly over segments."""
        segment_idx = torch.zeros(T, dtype=torch.long, device=device)
        for t in range(T):
            segment_idx[t] = min(int(t * num_segments / T), num_segments - 1)
        return segment_idx

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
        """Segment the input sequence using DP-optimal boundary placement.

        Args:
            x: (B, T, D) input features.
            rate: Compression rate in (0, 1]; a float or a per-sample (B,)
                tensor. Target segments T' = clamp(round(rate * T), 1, T).
                Pass ``None`` to return a trivial per-frame identity
                segmentation.
            padding_mask: (B, T) bool tensor; ``True`` marks padding positions.
            anchor_boundary: (B, T) binary tensor of boundary positions
                inherited from prior codebooks (1 = boundary). When
                provided, boundaries may only be placed on anchor positions:
                the DP selects the subset of anchors that minimises the
                total L2 distortion for the target segment count. When no
                anchor falls inside a sequence, that sequence uses the
                plain unconstrained DP; when the target count cannot be
                reached with the anchors, it falls back to uniform
                segmentation.
            **kwargs: Ignored; accepted so all compression models share one
                call signature.

        Returns:
            ``CompressionOutput`` for the batch.

        Example:
            .. code-block:: python

                >>> model = DPSegmentationCompression(max_span=4)
                >>> output = model(torch.randn(2, 50, 128), rate=0.5)
                >>> output.expected_length
                tensor([[25.], [25.]])
        """
        B, T, _ = x.shape
        device = x.device

        if rate is None:
            return self._build_no_rate_output(x, padding_mask)

        use_anchors = anchor_boundary is not None

        valid_lengths = (
            (~padding_mask).sum(dim=1).float() if padding_mask is not None else None
        )

        if isinstance(rate, torch.Tensor):
            rate_b = rate.squeeze(-1)
        else:
            rate_b = torch.full((B,), rate, device=device, dtype=torch.float)

        all_segment_idx = torch.zeros(B, T, dtype=torch.long, device=device)

        for b in range(B):
            valid_length = (
                int(valid_lengths[b].item()) if valid_lengths is not None else T
            )
            if valid_length <= 1:
                continue

            x_b = x[b, :valid_length]  # (valid_length, D)

            rate_b_item = (
                rate_b[b] if rate_b.dim() > 0 else rate_b
            )  # float32 scalar tensor

            # Replicate CosineSimilarityCompression topk formula exactly,
            # keeping arithmetic in float32 tensors to match its rounding:
            #   no padding:   k = (r × (valid_length−1)).long()
            #   with padding: k = clamp((r × valid_length−1).long(), 0)
            # and num_segments = k+1 in both cases.
            if valid_lengths is None:
                num_segments = max(
                    1, int((rate_b_item * (valid_length - 1)).long().item()) + 1
                )
            else:
                num_segments = max(
                    1, int((rate_b_item * valid_length - 1).long().item()) + 1
                )

            # Feasibility: with a finite max_span, we need at least
            # ceil(valid_length / max_span) segments to cover all frames.
            if self.max_span is not None:
                num_segments = max(
                    num_segments, math.ceil(valid_length / self.max_span)
                )
            num_segments = min(num_segments, valid_length)

            # Cost table and DP run without gradient tracking: boundaries are
            # hard decisions; gradients flow through _average_segments instead.
            with torch.no_grad():
                # Anchor-augmented path: select the best subset of anchor
                # positions as boundaries, minimizing total L2 distortion.
                anchors_b = None
                if use_anchors:
                    # Anchors live at positions 1..valid_length-1 (position 0
                    # is never a boundary by convention).
                    anchors_b = (
                        anchor_boundary[b, 1:valid_length].nonzero(as_tuple=True)[0] + 1
                    )
                    if anchors_b.numel() > 0:
                        segment_idx_b = self._run_dp_anchor_subset(
                            x_b,
                            anchors_b,
                            num_segments,
                        )
                    else:
                        cost = self._compute_cost_table(x_b)
                        segment_idx_b = self._run_dp(cost, valid_length, num_segments)
                else:
                    cost = self._compute_cost_table(x_b)
                    segment_idx_b = self._run_dp(cost, valid_length, num_segments)

            all_segment_idx[b, :valid_length] = segment_idx_b
            if valid_length < T:
                all_segment_idx[b, valid_length:] = int(segment_idx_b[-1].item())

        # Derive boundary tensor from segment indices.
        boundary = torch.zeros(B, T, device=device, dtype=torch.long)
        boundary[:, 1:] = (all_segment_idx[:, 1:] != all_segment_idx[:, :-1]).long()
        if padding_mask is not None:
            boundary = boundary.masked_fill(padding_mask, 0)

        boundary_soft = boundary.float()
        # Recompute cumsum-based segment_idx for pipeline consistency.
        segment_idx = torch.cumsum(boundary, dim=1)
        reconstructed = self._average_segments(x, segment_idx, padding_mask)
        expected_length = boundary_soft.sum(dim=1, keepdim=True) + 1

        return CompressionOutput(
            boundary_soft=boundary_soft,
            segment_idx=segment_idx,
            reconstructed_features=reconstructed,
            expected_length=expected_length,
        )
