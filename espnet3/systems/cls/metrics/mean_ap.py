"""Mean average precision metric for classification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.systems.cls.metrics.scoring_utils import (
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

    This metric needs ``score.scp`` in addition to ``ref.scp``, which the
    default ``ref_key``/``hyp_key`` fallback cannot express. Declare the
    inputs in ``conf/metrics.yaml``::

        - metric:
            _target_: espnet3.systems.cls.metrics.mean_ap.MAP
            token_list: data/token_list
          inputs:
            ref: ref
            score: score
    """

    def __init__(
        self,
        token_list: str,
        ref_key: str = "ref",
        score_key: str = "score",
    ) -> None:
        """Initialize the mAP metric.

        Args:
            token_list: Path to the recipe's token list. It defines the order
                of the columns in ``score.scp``.
            ref_key: Key name for reference label entries.
            score_key: Key name for per-class probability entries.
        """
        self.token_list = token_list
        self.ref_key = ref_key
        self.score_key = score_key

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        """Compute mean average precision.

        Args:
            data (Dict[str, Path]): Mapping of metric input aliases to SCP
                paths. Expects ``data[self.ref_key]`` and
                ``data[self.score_key]`` to be aligned by utterance ID.
            test_name (str): Test set name. Unused by this metric.
            inference_dir (Path): Inference output root. Unused by this metric.

        Returns:
            Dict[str, float]: ``{"mAP": <percentage>}``

        Raises:
            RuntimeError: If ``scikit-learn`` is not installed.
            ValueError: If a score row does not match the class count, if a
                reference label is missing from the token list, or if no class
                has a reference example.

        Example:
            >>> metric({"ref": Path("test/ref.scp"),
            ...         "score": Path("test/score.scp")}, "test", Path("infer"))
            {'mAP': 27.0}
        """
        sklearn_metrics = require_sklearn()
        refs = []
        scores = []
        for _, row in self.iter_inputs(data, self.ref_key, self.score_key):
            refs.append(row[self.ref_key])
            scores.append(row[self.score_key])

        classes = load_class_labels(self.token_list)
        target = build_target_matrix(refs, classes)
        output = build_score_matrix(scores, len(classes))
        keep, _ = supported_classes(target, classes)
        values = [
            sklearn_metrics.average_precision_score(target[:, k], output[:, k])
            for k in keep
        ]
        return {"mAP": round(float(np.mean(values)) * 100, 2)}
