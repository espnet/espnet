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
- The NME sweep runs with BLAS/OpenMP threads capped (``_BLAS_THREADS``):
  the small-matrix LAPACK eigendecompositions are orders of magnitude
  slower under OpenBLAS thread oversubscription (measured on a 12-core
  host: ``eigvalsh`` on 750x750 takes ~16 ms at 1-2 threads vs ~5.8 s at
  12). Clustering results are unaffected.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import numpy as np

# BLAS/OpenMP thread cap for the LAPACK-heavy parts of this module; see the
# module docstring for the measured oversubscription pathology. 2 measured
# fastest; the cap also stops idle OpenBLAS workers from spin-waiting at
# 100% CPU on every core during the single-threaded Python sections.
_BLAS_THREADS = 2


@contextmanager
def _limited_blas_threads() -> Iterator[None]:
    """Cap BLAS/OpenMP threads inside the block; no-op if unavailable.

    ``threadpoolctl`` ships with scikit-learn, which the ``diarize`` stage
    requires anyway; the lazy import keeps this module importable without
    it (e.g. for unit tests of the pure-numpy helpers).
    """
    try:
        from threadpoolctl import threadpool_limits  # noqa: PLC0415
    except ImportError:
        yield
        return
    with threadpool_limits(limits=_BLAS_THREADS):
        yield


def get_kneighbors_conn(X_dist: np.ndarray, p_neighbors: int) -> np.ndarray:
    """Binarized p-neighbor connectivity (egs1 port).

    For each row i, entry (j, i) is set to 1 for the ``p_neighbors``
    positions j holding the largest values of that row.

    Vectorized form of the egs1 per-row loop: one axis-wise ``argsort``
    (same quicksort/introsort as ``np.argsort`` on a single row, hence
    identical output including tie-breaking) replaces N Python-level
    sort-and-assign iterations; ``p_neighbors`` outside 1..N is clamped,
    matching the loop's slicing semantics.
    """
    X_dist = np.asarray(X_dist)
    n = X_dist.shape[0]
    X_dist_out = np.zeros_like(X_dist)
    if n == 0 or p_neighbors <= 0:
        return X_dist_out
    k = min(int(p_neighbors), n)
    order = np.argsort(X_dist, axis=1, kind="quicksort")
    top = order[:, ::-1][:, :k]  # per-row indices of the k largest values
    X_dist_out[top, np.arange(n)[:, None]] = 1
    return X_dist_out


def threshold_affinity(A: np.ndarray, p: int) -> np.ndarray:
    """Keep row values greater than the (p+1)-th largest (egs1 port).

    Unlike a binarization, the original values above the threshold are
    preserved. Requires ``0 <= p <= N - 1``; callers must clamp.

    Vectorized form of the egs1 per-row loop: one axis-wise ``np.sort``
    yields every row's (p+1)-th largest value at once. The threshold is
    compared as a value (not selected by index), so rows with ties give
    exactly the loop's result.
    """
    A = np.asarray(A)
    N = A.shape[0]
    if N == 0:
        return np.zeros((0, 0))
    if p < 0 or p > N - 1:
        raise IndexError(
            f"threshold index p must satisfy 0 <= p <= N-1; got p={p}, N={N}."
        )
    ascending = np.sort(A, axis=1)
    thr = ascending[:, N - 1 - int(p)]  # per-row (p+1)-th largest value
    Ap = np.zeros((N, N))
    np.copyto(Ap, A, where=A > thr[:, None])
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

    with _limited_blas_threads():
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
                # Degenerate case (too few points for the p >= 2 search, or
                # all criteria non-finite): fall back to a minimal threshold.
                pbest = max(1, p_hi)

        if num_clusters is None:
            if kbest is None:
                # pbest was given: the egs1 original crashes here (undefined
                # p); recompute the eigengap parameters at the given pbest
                # instead.
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
