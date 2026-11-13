"""Area under the ROC curve metric for classification."""

from __future__ import annotations

import logging
from typing import Dict, Sequence

import numpy as np

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field
from espnet3.systems.esp2_cls.src.scoring_utils import (
    build_score_matrix,
    build_target_matrix,
    load_class_labels,
    require_sklearn,
    supported_classes,
)

logger = logging.getLogger(__name__)


class AUC(BaseMetric):
    """Compute the macro-averaged one-vs-rest AUC from class probabilities.

    The area under the ROC curve is computed per class and then macro
    averaged, matching the ``mean_auc`` column of espnet2's ``cls_score.py``.

    It reads the per-class scores in addition to the reference labels.
    Declare it in ``conf/metrics.yaml``:

    .. code-block:: yaml

        - metric:
            _target_: espnet3.systems.esp2_cls.metrics.auc.AUC
            token_list: data/token_list
          inputs:
            ref: dataset:label
            score: score
    """

    inputs = (Field("ref", "text"), Field("score", "text"))
    outputs = (Field("AUC", "number"),)

    def __init__(self, token_list: str) -> None:
        """Initialize the AUC metric.

        Args:
            token_list: Path to the recipe's token list. It defines the order
                of the columns in ``score.scp``.
        """
        self.token_list = token_list
        super().__init__()
        self.add_state("refs", [], "cat")
        self.add_state("scores", [], "cat")

    def update(self, *, ref: Sequence[str], score: Sequence[str]) -> None:
        """Keep the labels and scores of one batch; the score needs them all.

        Args:
            ref: Reference labels, one per utterance; space-separated labels
                are multiple positives.
            score: Space-separated per-class scores, aligned with ``ref``.

        Examples:
            >>> metric = AUC("data/token_list")
            >>> metric.update(ref=["joy"], score=["0.1 0.9"])
        """
        self.refs.extend(ref)
        self.scores.extend(score)

    def compute(self) -> Dict[str, float]:
        """Return the macro-averaged one-vs-rest AUC of every utterance added so far.

        Returns:
            Dict[str, float]: ``{"AUC": <percentage>}``, rounded to 2 places.

        Raises:
            RuntimeError: If ``scikit-learn`` is not installed.
            ValueError: If a reference label is missing from ``token_list``,
                a score row has the wrong length, or no class has both a
                positive and a negative reference.
        """
        sklearn_metrics = require_sklearn()
        classes = load_class_labels(self.token_list)
        target = build_target_matrix(self.refs, classes)
        output = build_score_matrix(self.scores, len(classes))
        keep, kept_labels = supported_classes(target, classes)
        # A class every utterance belongs to has no negative to rank against,
        # and sklearn returns nan for it rather than raising.
        scorable = [k for k in keep if target[:, k].sum() < target.shape[0]]
        if not scorable:
            raise ValueError(
                "AUC needs a class with both a positive and a negative "
                f"reference, but every utterance carries {kept_labels}"
            )
        dropped = [classes[k] for k in keep if k not in set(scorable)]
        if dropped:
            logger.warning("Skipping classes with no negative example: %s", dropped)
        values = [
            sklearn_metrics.roc_auc_score(target[:, k], output[:, k]) for k in scorable
        ]
        return {"AUC": round(float(np.mean(values)) * 100, 2)}
