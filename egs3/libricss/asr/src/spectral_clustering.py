"""NME spectral clustering for speaker diarization.

Port of egs/libri_css/asr1/diarization/spec_clust.py (Maxim Korenevsky,
STC-innovations Ltd, Apache 2.0), which implements:

    T. Park, K. Han, M. Kumar, and S. Narayanan, "Auto-tuning spectral
    clustering for speaker diarization using normalized maximum eigengap,"
    IEEE Signal Processing Letters, vol. 27, pp. 381-385, 2019.

Algorithm: for each candidate neighbor count p in 2..pmax, build the
symmetrized binarized p-neighbor graph of the affinity matrix, take its
Laplacian eigengap sequence, and score p by r = p / g with
g = max(eigengap) / max(eigenvalue). The best (p, k) yields the cluster
count k + 1; final labels come from scikit-learn spectral clustering on the
value-preserving thresholded affinity, symmetrized.

Documented differences from the egs1 original:

- The original crashes with ``IndexError``/``NameError`` on small affinity
  matrices (``Threshold`` index p must stay below N; the NME loop
  ``range(2, pmax+1)`` can be empty). This port guards those cases.
- The original's ``pbest != 0`` + ``num_clusters=None`` branch calls
  ``ComputeNMEParameters(A, p)`` with an undefined ``p`` and a wrong
  unpacking order (a dead code path). This port recomputes the eigengap
  parameters at the given ``pbest`` instead.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def get_kneighbors_conn(X_dist: np.ndarray, p_neighbors: int) -> np.ndarray:
    """Binarized p-neighbor connectivity (egs1 port).

    For each row i, entry (j, i) is set to 1 for the ``p_neighbors``
    positions j holding the largest values of that row.
    """
    X_dist_out = np.zeros_like(X_dist)
    for i, line in enumerate(X_dist):
        sorted_idx = np.argsort(line)[::-1][:p_neighbors]
        X_dist_out[sorted_idx, i] = 1
    return X_dist_out


def threshold_affinity(A: np.ndarray, p: int) -> np.ndarray:
    """Keep row values greater than the (p+1)-th largest (egs1 port).

    Unlike a binarization, the original values above the threshold are
    preserved. Requires ``p <= N - 1``; callers must clamp.
    """
    N = A.shape[0]
    Ap = np.zeros((N, N))
    for i in range(N):
        thr = np.sort(A[i, :])[::-1][p]
        mask = A[i, :] > thr
        Ap[i, mask] = A[i, mask]
    return Ap


def laplacian_matrix(A: np.ndarray) -> np.ndarray:
    """Return the unnormalized graph Laplacian L = D - A (egs1 port)."""
    d = np.sum(A, axis=1) - np.diag(A)
    return np.diag(d) - A


def eigengap(S: np.ndarray) -> np.ndarray:
    """Return adjacent differences of the sorted eigenvalues (egs1 port)."""
    return np.diff(np.sort(S))


def compute_nme_parameters(
    A: np.ndarray, p: int, max_num_clusters: int
) -> tuple[np.ndarray, float, int, float]:
    """NME criterion at neighbor count ``p`` (egs1 port).

    Returns:
        ``(e, g, k, r)``: eigengap sequence, normalized maximum eigengap
        ``g = max(e[:max_num_clusters]) / (max(S) + 1e-10)``, best cluster
        index ``k = argmax(e[:max_num_clusters])``, and criterion
        ``r = p / g`` (lower is better).
    """
    Ap = get_kneighbors_conn(A, p)
    Ap = (Ap + Ap.T) / 2
    Lp = laplacian_matrix(Ap)
    S = np.linalg.eigvalsh(Lp)  # symmetric matrix: real eigenvalues
    e = eigengap(S)
    g = np.max(e[:max_num_clusters]) / (np.max(S) + 1e-10)
    r = p / g
    k = int(np.argmax(e[:max_num_clusters]))
    return e, float(g), k, float(r)


def _effective_p(p: int, n: int) -> int:
    """Clamp a neighbor count into ``threshold_affinity``'s valid range."""
    return max(1, min(int(p), n - 1))


def nme_spectral_clustering(
    A: np.ndarray,
    num_clusters: Optional[int] = None,
    max_num_clusters: int = 10,
    pbest: int = 0,
    pmax: int = 20,
    random_state: int = 0,
) -> np.ndarray:
    """Spectral clustering with Normalized Maximum Eigengap (egs1 port).

    Args:
        A: Square symmetric cosine affinity matrix (N, N).
        num_clusters: Number of clusters to generate. ``None`` (default)
            determines it automatically from the NME criterion.
        max_num_clusters: Maximum allowed number of clusters.
        pbest: Best neighbor count for affinity thresholding. 0 (default)
            determines it automatically by minimizing ``r = p / g``.
        pmax: Maximum neighbor count for the automatic search (>= 2).
        random_state: Seed passed to ``SpectralClustering``.

    Returns:
        Integer cluster labels of shape (N,), values in 0..num_clusters-1.
    """
    n = A.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)

    max_k = max(1, min(int(max_num_clusters), n - 1))
    kbest: Optional[int] = None

    if pbest == 0:
        p_hi = min(int(pmax), n - 1)
        rbest: Optional[float] = None
        for p in range(2, p_hi + 1):
            _e, _g, k, r = compute_nme_parameters(A, p, max_k)
            if not np.isfinite(r):
                continue
            if rbest is None or rbest > r:
                rbest, pbest, kbest = r, p, k
        if rbest is None:
            # Degenerate case (too few points for the p >= 2 search, or all
            # criteria non-finite): fall back to a minimal threshold.
            pbest = max(1, p_hi)

    if num_clusters is None:
        if kbest is None:
            # pbest was given: the egs1 original crashes here (undefined p);
            # recompute the eigengap parameters at the given pbest instead.
            _e, _g, kbest, _r = compute_nme_parameters(
                A, _effective_p(pbest, n), max_k
            )
        num_clusters = kbest + 1

    num_clusters = max(1, min(int(num_clusters), n))
    if num_clusters == 1:
        return np.zeros(n, dtype=int)
    return _spectral_clustering_sklearn(
        A, num_clusters, _effective_p(pbest, n), random_state
    )


def _spectral_clustering_sklearn(
    A: np.ndarray, num_clusters: int, p: int, random_state: int
) -> np.ndarray:
    """Spectral clustering on the thresholded affinity (egs1 port).

    ``SpectralClustering`` keeps the ``affinity=`` argument across
    scikit-learn versions (the ``metric=`` rename only affected
    ``AgglomerativeClustering``).
    """
    try:
        from sklearn.cluster import SpectralClustering  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "The `diarize` stage requires scikit-learn: `pip install "
            "scikit-learn` (>=1.0.0, or espnet[egs2])."
        ) from e
    Ap = threshold_affinity(A, p)
    Ap = (Ap + Ap.T) / 2
    model = SpectralClustering(
        n_clusters=num_clusters,
        affinity="precomputed",
        random_state=random_state,
    )
    return model.fit_predict(Ap)
