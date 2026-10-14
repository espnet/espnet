"""Density-peak clustering compression model."""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from espnet3.systems.lace.src.compression import (
    BaseCompressionModel,
    CompressionOutput,
)


class DensityPeakCompression(BaseCompressionModel):
    """Frame boundary detection via Temporal-Aware Density Peak Clustering.

    Based on VARSTok (arXiv:2509.04685). A frame is a good cluster seed if it
    is *locally dense* (surrounded by similar neighbours) **and** *isolated* from
    denser regions. Concretely, each frame is scored as

        s[i] = rho[i] * delta[i]

    where

    * ``rho[i] = exp( mean of k-NN cosine similarities )``  - local density,
    * ``delta[i]`` - min distance to any frame j with rho[j] > rho[i];
      for the global density peak, ``delta = max distance to any frame``.

    Segmentation is the full VARSTok greedy bidirectional expansion: each
    iteration picks the highest-scored unassigned frame as a seed, then
    expands forward/backward up to *max_span* frames subject to a
    similarity threshold. ``rate`` is used directly as that threshold:
    high rate (-> 1.0) makes merging hard (many segments, little
    compression); low rate (-> 0.0) makes merging easy (few segments,
    heavy compression).

    Example:
        .. code-block:: python

            >>> model = DensityPeakCompression(k=5, beta=0.2, max_span=4)
            >>> output = model(torch.randn(2, 50, 128), rate=0.5)
            >>> output.segment_idx.shape
            torch.Size([2, 50])
    """

    def __init__(
        self,
        k: int = 5,
        beta: float = 0.2,
        max_span: int = 4,
    ):
        """Store the clustering hyperparameters.

        Args:
            k: Number of nearest neighbours for the local density.
            beta: Penalty weight that discourages absorbing high-scored
                frames into a neighbouring cluster.
            max_span: Maximum number of frames per cluster.
        """
        super().__init__()
        self.k = k
        self.beta = beta
        self.max_span = max_span

    # ------------------------------------------------------------------
    # Density-peak helpers (operate on a single, unpadded sequence)
    # ------------------------------------------------------------------

    def _compute_similarity_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Return the (T, T) pairwise cosine similarity of ``x`` (T, D) in [0, 1]."""
        x_normalized = F.normalize(x, dim=-1)  # (T, D)
        cosine = torch.matmul(x_normalized, x_normalized.T)  # (T, T)  in [-1, 1]
        return (cosine + 1.0) / 2.0  # rescale to [0, 1]

    def _compute_local_density(self, similarity: torch.Tensor) -> torch.Tensor:
        """Return the (T,) local density rho = exp(mean of k-NN similarities)."""
        T = similarity.shape[0]
        k = min(self.k, T - 1)

        # Zero out self-similarity before selecting k nearest neighbours.
        similarity_no_diagonal = similarity.clone()
        similarity_no_diagonal.fill_diagonal_(0.0)

        knn_similarities, _ = torch.topk(similarity_no_diagonal, k, dim=-1)  # (T, k)
        return torch.exp(knn_similarities.mean(dim=-1))  # (T,)

    def _compute_peak_distance(
        self, similarity: torch.Tensor, rho: torch.Tensor
    ) -> torch.Tensor:
        """Return the (T,) peak distance delta of every frame.

        For frame i, delta[i] is the minimum distance to any frame j with
        rho[j] > rho[i]; for the global density peak (no such j), it is the
        maximum distance to any frame.
        """
        distance = 1.0 - similarity  # (T, T)

        # (T, T)[i, j] = True when rho[j] > rho[i]
        higher = rho.unsqueeze(0) > rho.unsqueeze(1)  # (T, T)

        # Replace non-higher-density entries with inf so min ignores them.
        distance_masked = distance.masked_fill(~higher, float("inf"))
        delta, _ = distance_masked.min(dim=-1)  # (T,)

        # Frames with no denser neighbour get max distance (global peak rule).
        is_peak = delta.isinf()
        if is_peak.any():
            max_distance = distance.max(dim=-1).values  # (T,)
            delta = torch.where(is_peak, max_distance, delta)

        return delta

    def _compute_density_scores(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the (T,) scores rho * delta of ``x`` (T, D) and its similarity.

        The (T, T) similarity matrix is returned too, for the greedy
        clustering that follows.
        """
        similarity = self._compute_similarity_matrix(x)
        rho = self._compute_local_density(similarity)
        delta = self._compute_peak_distance(similarity, rho)
        return rho * delta, similarity

    # ------------------------------------------------------------------
    # Greedy bidirectional clustering
    # ------------------------------------------------------------------

    def _cluster_greedily(
        self,
        scores: torch.Tensor,
        similarity: torch.Tensor,
        max_span: int,
        threshold: float,
    ) -> torch.Tensor:
        """Return the (T,) segment index from greedy expansion around density peaks.

        Iteratively selects the highest-scored unassigned frame (``scores``,
        (T,)) as a cluster seed and expands forwards and backwards up to
        ``max_span`` frames, merging any frame whose adjusted similarity
        exceeds ``threshold``; the clusters are then numbered in temporal
        order. The adjusted similarity penalises absorbing high-scored frames::

            similarity_score[seed, t] = similarity[seed, t] - beta * scores[t]
        """
        T = scores.shape[0]
        assigned = torch.zeros(T, dtype=torch.bool, device=scores.device)
        clusters: list[list[int]] = []

        while not assigned.all():
            # Pick the unassigned frame with the highest density score.
            masked_scores = scores.clone()
            masked_scores[assigned] = -float("inf")
            seed = int(masked_scores.argmax().item())

            cluster = [seed]
            assigned[seed] = True

            # Penalise adding other high-score frames into this cluster.
            similarity_score = similarity[seed] - self.beta * scores  # (T,)

            # Forward expansion.
            for t in range(seed + 1, min(T, seed + max_span + 1)):
                if assigned[t] or len(cluster) >= max_span:
                    break
                if similarity_score[t].item() > threshold:
                    cluster.append(t)
                    assigned[t] = True
                else:
                    break

            # Backward expansion.
            for t in range(seed - 1, max(-1, seed - max_span - 1), -1):
                if assigned[t] or len(cluster) >= max_span:
                    break
                if similarity_score[t].item() > threshold:
                    cluster.append(t)
                    assigned[t] = True
                else:
                    break

            clusters.append(sorted(cluster))

        # Sort clusters by their earliest frame to restore temporal order.
        clusters.sort(key=lambda frames: frames[0])

        segment_idx = torch.zeros(T, dtype=torch.long, device=scores.device)
        for segment_id, cluster in enumerate(clusters):
            for t in cluster:
                segment_idx[t] = segment_id

        return segment_idx

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        rate: Optional[Union[float, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressionOutput:
        """Segment the input sequence using density peak clustering.

        Args:
            x: (B, T, D) input features.
            rate: Compression rate in (0, 1], used directly as the
                similarity threshold for cluster merging; a float or a
                per-sample (B,) tensor. ``max_span`` is fixed at
                construction time. Pass ``None`` to return a trivial
                per-frame identity segmentation.
            padding_mask: (B, T) bool tensor; ``True`` marks padding positions.
            **kwargs: Ignored (e.g. ``anchor_boundary``; this model does not
                support anchoring).

        Returns:
            ``CompressionOutput`` for the batch.

        Example:
            .. code-block:: python

                >>> model = DensityPeakCompression(max_span=4)
                >>> output = model(torch.randn(2, 50, 128), rate=0.5)
                >>> output.reconstructed_features.shape
                torch.Size([2, 50, 128])
        """
        B, T, _ = x.shape
        device = x.device

        if rate is None:
            return self._build_no_rate_output(x, padding_mask)

        valid_lengths = (
            (~padding_mask).sum(dim=1).float() if padding_mask is not None else None
        )

        if isinstance(rate, torch.Tensor):
            rate_b = rate.squeeze(-1)  # (B,) or scalar tensor
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

            scores_b, similarity_b = self._compute_density_scores(x_b)

            # rate is used directly as the similarity threshold:
            #   high rate (-> 1.0) = hard to merge = many segments
            #   low  rate (-> 0.0) = easy to merge = few  segments
            threshold = float(rate_b[b].item() if rate_b.dim() > 0 else rate_b.item())

            segment_idx_b = self._cluster_greedily(
                scores_b, similarity_b, self.max_span, threshold
            )
            all_segment_idx[b, :valid_length] = segment_idx_b

            # Padding frames share the last valid segment.
            if valid_length < T:
                all_segment_idx[b, valid_length:] = int(segment_idx_b[-1].item())

        # Derive boundary tensor from segment indices.
        boundary = torch.zeros(B, T, device=device, dtype=torch.long)
        boundary[:, 1:] = (all_segment_idx[:, 1:] != all_segment_idx[:, :-1]).long()
        if padding_mask is not None:
            boundary = boundary.masked_fill(padding_mask, 0)

        boundary_soft = boundary.float()
        # Recompute cumsum-based segment_idx for consistency with the rest of
        # the pipeline (same convention as CosineSimilarityCompression).
        segment_idx = torch.cumsum(boundary, dim=1)
        reconstructed = self._average_segments(x, segment_idx, padding_mask)
        expected_length = boundary_soft.sum(dim=1, keepdim=True) + 1

        return CompressionOutput(
            boundary_soft=boundary_soft,
            segment_idx=segment_idx,
            reconstructed_features=reconstructed,
            expected_length=expected_length,
        )
