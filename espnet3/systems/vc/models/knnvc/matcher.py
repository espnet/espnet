"""k-nearest-neighbours regression on encoder features (kNN-VC's "converter").

The converter of kNN-VC is non-parametric: every source frame is replaced by
the mean of its ``k`` nearest frames (cosine distance) in a matching set made
of the target speaker's features. The same operation, applied against the
*other utterances of a training utterance's own pool*, is the "prematching"
used to build better vocoder training data.
"""

from __future__ import annotations

import torch


def compute_cosine_distances(
    source_feats: torch.Tensor, matching_pool: torch.Tensor
) -> torch.Tensor:
    """Compute pairwise cosine distances between two feature sets.

    Numerically equivalent to ``1 - cosine_similarity`` but built from a single
    ``torch.cdist`` call, which is much faster for large matching pools (this
    is the ``fast_cosine_dist`` trick from the official implementation).

    Args:
        source_feats: Query frames, shape ``(N1, dim)``.
        matching_pool: Matching set frames, shape ``(N2, dim)``.

    Returns:
        Distance matrix of shape ``(N1, N2)`` in ``[0, 2]``.

    Raises:
        ValueError: If either input is not 2-D or the feature dims differ.
    """
    if source_feats.dim() != 2 or matching_pool.dim() != 2:
        raise ValueError(
            "Expected 2-D (frames, dim) tensors, got shapes "
            f"{tuple(source_feats.shape)} and {tuple(matching_pool.shape)}"
        )
    if source_feats.size(1) != matching_pool.size(1):
        raise ValueError(
            "Feature dims differ: "
            f"{source_feats.size(1)} (source) vs {matching_pool.size(1)} (pool)"
        )
    source_norms = torch.norm(source_feats, p=2, dim=-1)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = (
        -torch.cdist(source_feats[None], matching_pool[None], p=2)[0] ** 2
        + source_norms[:, None] ** 2
        + matching_norms[None] ** 2
    )
    dotprod /= 2
    return 1 - (dotprod / (source_norms[:, None] * matching_norms[None]))


def match_features(
    query_seq: torch.Tensor,
    matching_set: torch.Tensor,
    synth_set: torch.Tensor | None = None,
    topk: int = 4,
) -> torch.Tensor:
    """Replace every query frame by the mean of its ``topk`` nearest pool frames.

    Args:
        query_seq: Source features, shape ``(N1, dim)``.
        matching_set: Features the nearest neighbours are searched in, shape
            ``(N2, dim)``. For voice conversion this is the target speaker's
            reference features.
        synth_set: Optional features of shape ``(N2, dim)`` aligned row-by-row
            with ``matching_set``; the neighbours found in ``matching_set`` are
            averaged over this set instead. ``None`` (the default, and the
            paper's setting) uses ``matching_set`` for both.
        topk: Number of neighbours to average. Clamped to ``N2`` when the pool
            is smaller than ``topk``.

    Returns:
        Converted features of shape ``(N1, dim)`` on ``query_seq.device``.

    Raises:
        ValueError: If ``matching_set`` is empty, ``topk < 1``, or ``synth_set``
            is not aligned with ``matching_set``.

    Examples:
        >>> converted = match_features(src_feats, ref_feats, topk=4)
        >>> converted.shape == src_feats.shape
        True
    """
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    if matching_set.numel() == 0 or matching_set.size(0) == 0:
        raise ValueError("matching_set must contain at least one frame.")
    if synth_set is None:
        synth_set = matching_set
    elif synth_set.shape != matching_set.shape:
        raise ValueError(
            "synth_set must be aligned with matching_set: "
            f"{tuple(synth_set.shape)} vs {tuple(matching_set.shape)}"
        )
    device = query_seq.device
    matching_set = matching_set.to(device, dtype=query_seq.dtype)
    synth_set = synth_set.to(device, dtype=query_seq.dtype)

    dists = compute_cosine_distances(query_seq, matching_set)
    k = min(int(topk), matching_set.size(0))
    best = dists.topk(k=k, dim=-1, largest=False)
    return synth_set[best.indices].mean(dim=1)
