"""Mean average precision metric for classification."""

from __future__ import annotations

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


class MAP(BaseMetric):
    """Compute mean average precision (mAP) from class probabilities.

    Average precision is computed per class in a one-vs-rest fashion and then
    macro averaged, matching the ``mAP`` column of espnet2's ``cls_score.py``.
    Because it consumes ranked probabilities rather than the arg-max label, it
    is sensitive to how well the model orders its classes.

    It reads the per-class scores in addition to the reference labels.
    Declare it in ``conf/metrics.yaml``:

    .. code-block:: yaml

        - metric:
            _target_: espnet3.systems.esp2_cls.metrics.mean_ap.MAP
            token_list: data/token_list
          inputs:
            ref: dataset:label
            score: score
    """

    inputs = (Field("ref", "text"), Field("score", "text"))
    outputs = (Field("mAP", "number"),)

    def __init__(self, token_list: str) -> None:
        """Initialize the mAP metric.

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
            >>> metric = MAP("data/token_list")
            >>> metric.update(ref=["joy"], score=["0.1 0.9"])
        """
        self.refs.extend(ref)
        self.scores.extend(score)

    def compute(self) -> Dict[str, float]:
        """Return the mean average precision of every utterance added so far.

        Returns:
            Dict[str, float]: ``{"mAP": <percentage>}``, rounded to 2 places.

        Raises:
            RuntimeError: If ``scikit-learn`` is not installed.
            ValueError: If a reference label is missing from ``token_list``,
                a score row has the wrong length, or no class has a positive reference.
        """
        sklearn_metrics = require_sklearn()
        classes = load_class_labels(self.token_list)
        target = build_target_matrix(self.refs, classes)
        output = build_score_matrix(self.scores, len(classes))
        keep, _ = supported_classes(target, classes)
        values = [
            sklearn_metrics.average_precision_score(target[:, k], output[:, k])
            for k in keep
        ]
        return {"mAP": round(float(np.mean(values)) * 100, 2)}
