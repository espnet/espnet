"""Trial scoring helpers shared by validation callbacks and metrics."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from espnet2.utils.eer import ComputeErrorRates, ComputeMinDcf, tuneThresholdfromScore


def compute_eer(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Compute the equal error rate of a set of scored trials.

    Args:
        scores: Similarity score of each trial. Higher means more likely to be
            the same speaker.
        labels: ``1`` for target trials and ``0`` for nontarget trials.

    Returns:
        Equal error rate in percent.

    Examples:
        >>> round(compute_eer([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 3)
        0.0
    """
    _, eer, _, _ = tuneThresholdfromScore(np.asarray(scores), np.asarray(labels), [1])
    return float(eer)


def compute_min_dcf(
    scores: Sequence[float],
    labels: Sequence[int],
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """Compute the normalized minimum detection cost of a set of trials.

    Args:
        scores: Similarity score of each trial.
        labels: ``1`` for target trials and ``0`` for nontarget trials.
        p_target: Prior probability of a target trial.
        c_miss: Cost of a missed detection.
        c_fa: Cost of a false alarm.

    Returns:
        Minimum detection cost normalized by the cost of the best trivial
        system, following the NIST SRE definition.

    Examples:
        >>> round(compute_min_dcf([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 3)
        0.0
    """
    fnrs, fprs, thresholds = ComputeErrorRates(list(scores), list(labels))
    min_dcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa)
    return float(min_dcf)


def score_statistics(
    scores: Sequence[float], labels: Sequence[int]
) -> Tuple[float, float, float, float]:
    """Return the mean and standard deviation of target and nontarget scores.

    Well separated distributions are the first thing to look at when a run
    produces an unexpected EER, so recipes report them next to the metric.

    Args:
        scores: Similarity score of each trial.
        labels: ``1`` for target trials and ``0`` for nontarget trials.

    Returns:
        ``(target_mean, target_std, nontarget_mean, nontarget_std)``.

    Examples:
        >>> score_statistics([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0])
        (1.0, 0.0, 0.0, 0.0)
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    target = scores[labels == 1]
    nontarget = scores[labels == 0]
    return (
        float(np.mean(target)),
        float(np.std(target)),
        float(np.mean(nontarget)),
        float(np.std(nontarget)),
    )
