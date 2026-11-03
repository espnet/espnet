"""Unweighted accuracy metric for classification."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field
from espnet3.systems.esp2_cls.src.scoring_utils import (
    require_sklearn,
    resolve_classes,
    single_labels,
)


class UA(BaseMetric):
    """Compute unweighted accuracy (UA) for a classification dataset.

    UA is the macro average of the per-class recall, so every class counts
    equally regardless of how many examples it has. The emotion recognition
    literature also calls it unweighted average recall (UAR), and scikit-learn
    calls it balanced accuracy. On the imbalanced label distributions typical
    of the task it is the more informative companion to
    :class:`espnet3.systems.esp2_cls.metrics.wa.WA`.

    Declare it in ``conf/metrics.yaml``:

    .. code-block:: yaml

        - metric:
            _target_: espnet3.systems.esp2_cls.metrics.ua.UA
            token_list: data/token_list  # optional; fixes the class set
          inputs:
            ref: dataset:label
            hyp: hyp
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("UA", "number"),)

    def __init__(self, token_list: Optional[str] = None) -> None:
        """Initialize the UA metric.

        Args:
            token_list: Optional path to the recipe's token list. It fixes the
                class set so unseen labels are caught; without it the classes
                are taken from the references.
        """
        self.token_list = token_list
        super().__init__()
        self.add_state("refs", [], "cat")
        self.add_state("hyps", [], "cat")

    def update(self, *, ref: Sequence[str], hyp: Sequence[str]) -> None:
        """Keep the labels of one batch; the score needs every utterance.

        Args:
            ref: Reference labels, one per utterance.
            hyp: Predicted labels, aligned with ``ref``.

        Examples:
            >>> metric = UA()
            >>> metric.update(ref=["joy", "anger"], hyp=["joy", "joy"])
        """
        self.refs.extend(ref)
        self.hyps.extend(hyp)

    def compute(self) -> Dict[str, float]:
        """Return the macro-averaged recall of every utterance added so far.

        Returns:
            Dict[str, float]: ``{"UA": <percentage>}``, rounded to 2 places.

        Raises:
            RuntimeError: If ``scikit-learn`` is not installed.
            ValueError: If no utterance was added, an utterance carries
                multiple labels, or a reference label is missing from
                ``token_list``.

        Examples:
            >>> metric = UA()
            >>> metric.update(ref=["joy", "anger"], hyp=["joy", "joy"])
            >>> metric.compute()
            {'UA': 50.0}
        """
        sklearn_metrics = require_sklearn()
        references = single_labels(self.refs)
        hypotheses = single_labels(self.hyps)
        classes = resolve_classes(references, self.token_list)
        score = sklearn_metrics.recall_score(
            references,
            hypotheses,
            labels=classes,
            average="macro",
            zero_division=0,
        )
        return {"UA": round(float(score) * 100, 2)}
