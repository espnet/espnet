"""Shared helpers for classification metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

try:
    from sklearn import metrics as sklearn_metrics
except ImportError:
    sklearn_metrics = None

logger = logging.getLogger(__name__)


def require_sklearn():
    """Return ``sklearn.metrics``, raising a helpful error when unavailable.

    Returns:
        module: The ``sklearn.metrics`` module.

    Raises:
        RuntimeError: If ``scikit-learn`` is not installed.
    """
    if sklearn_metrics is None:
        raise RuntimeError(
            "scikit-learn is required for classification metrics. "
            "Please install it with `pip install scikit-learn`."
        )
    return sklearn_metrics


def load_class_labels(token_list: str | Path) -> List[str]:
    """Read a token list and return class labels in classifier output order.

    The final entry is dropped to mirror ``n_classes = len(token_list) - 1``
    in ``espnet2/tasks/cls.py``, where the trailing ``<unk>`` is not a class.

    Args:
        token_list: Path to the token list written by the ``prepare_labels``
            stage.

    Returns:
        List[str]: Class labels ordered to match the score vector.

    Raises:
        ValueError: If the token list holds fewer than two entries.

    Example:
        >>> load_class_labels("data/token_list")
        ['neutral', 'joy', 'surprise', 'anger', 'sadness', 'disgust', 'fear']
    """
    path = Path(token_list)
    lines = path.read_text(encoding="utf-8").splitlines()
    tokens = [line.strip() for line in lines if line.strip()]
    if len(tokens) < 2:
        raise ValueError(
            f"token_list must hold at least two entries, got {len(tokens)}: {path}"
        )
    return tokens[:-1]


def build_target_matrix(
    references: Sequence[str], classes: Sequence[str]
) -> np.ndarray:
    """Convert reference label strings into a multi-hot target matrix.

    Args:
        references: One SCP value per utterance. Space-separated labels are
            treated as multiple positives, matching espnet2's ``cls_score.py``.
        classes: Class labels in score-vector order.

    Returns:
        np.ndarray: Float array of shape ``(n_utterances, n_classes)``.

    Raises:
        ValueError: If a reference label is absent from ``classes``.
    """
    index = {label: i for i, label in enumerate(classes)}
    target = np.zeros((len(references), len(classes)), dtype=np.float64)
    for row, reference in enumerate(references):
        for label in reference.split():
            if label not in index:
                raise ValueError(
                    f"Reference label {label!r} is not in the token list: "
                    f"{list(classes)}"
                )
            target[row, index[label]] = 1.0
    return target


def build_score_matrix(scores: Sequence[str], n_classes: int) -> np.ndarray:
    """Parse score SCP values into a probability matrix.

    Args:
        scores: One SCP value per utterance, holding space-separated floats.
        n_classes: Expected number of classes per row.

    Returns:
        np.ndarray: Float array of shape ``(n_utterances, n_classes)``.

    Raises:
        ValueError: If any row does not hold exactly ``n_classes`` values.
    """
    matrix = np.zeros((len(scores), n_classes), dtype=np.float64)
    for row, score in enumerate(scores):
        values = score.split()
        if len(values) != n_classes:
            raise ValueError(
                f"Expected {n_classes} scores per utterance but row {row} has "
                f"{len(values)}. Was inference run with all class probabilities?"
            )
        matrix[row] = [float(value) for value in values]
    return matrix


def supported_classes(
    target: np.ndarray, classes: Sequence[str]
) -> Tuple[np.ndarray, List[str]]:
    """Select the class columns that have at least one positive reference.

    espnet2's ``cls_score.py`` scores empty classes as 0.0 and averages them
    in, which drags the macro mean down. Skipping them keeps the mean
    meaningful on label sets that a corpus does not fully cover.

    Args:
        target: Multi-hot target matrix of shape ``(n_utterances, n_classes)``.
        classes: Class labels in score-vector order.

    Returns:
        Tuple[np.ndarray, List[str]]: Indices of the kept columns and their
        labels.

    Raises:
        ValueError: If no class has a positive reference.
    """
    keep = np.flatnonzero(target.sum(axis=0) > 0)
    dropped = [classes[i] for i in range(len(classes)) if i not in set(keep)]
    if dropped:
        logger.warning("Skipping classes with no reference example: %s", dropped)
    if keep.size == 0:
        raise ValueError("No class has a positive reference example")
    return keep, [classes[i] for i in keep]


def single_labels(references: Sequence[str]) -> List[str]:
    """Validate that every SCP value holds exactly one label and return them.

    Args:
        references: One SCP value per utterance.

    Returns:
        List[str]: The single label of each utterance.

    Raises:
        ValueError: If any utterance carries zero or multiple labels.
    """
    labels = []
    for row, reference in enumerate(references):
        parts = reference.split()
        if len(parts) != 1:
            raise ValueError(
                f"Expected one label per utterance but row {row} has "
                f"{len(parts)}: {reference!r}. This metric is multi-class only."
            )
        labels.append(parts[0])
    return labels


def resolve_classes(
    labels: Sequence[str], token_list: str | Path | None = None
) -> List[str]:
    """Return the classes to macro-average over, dropping unused ones.

    Args:
        labels: Reference label of each utterance.
        token_list: Optional token list fixing the class set and its order.
            When omitted, the classes are taken from ``labels`` in sorted
            order.

    Returns:
        List[str]: Classes that have at least one reference example.

    Raises:
        ValueError: If a reference label is absent from ``token_list``, or if
            ``labels`` is empty.
    """
    present = set(labels)
    if not present:
        raise ValueError("No reference labels to score")
    if token_list is not None:
        classes = load_class_labels(token_list)
        unknown = sorted(present - set(classes))
        if unknown:
            raise ValueError(
                f"Reference labels {unknown} are not in the token list: {classes}"
            )
    else:
        classes = sorted(present)
    dropped = [label for label in classes if label not in present]
    if dropped:
        logger.warning("Skipping classes with no reference example: %s", dropped)
    return [label for label in classes if label in present]
