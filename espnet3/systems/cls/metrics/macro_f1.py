"""Macro-averaged F1 metric for classification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.systems.cls.metrics.scoring_utils import (
    require_sklearn,
    resolve_classes,
    single_labels,
)


class MacroF1(BaseMetric):
    """Compute the macro-averaged F1 score for a classification dataset.

    The F1 score of each class is computed independently and then averaged
    without weighting by support, so rare classes are not drowned out.
    """

    def __init__(
        self,
        ref_key: str = "ref",
        hyp_key: str = "hyp",
        token_list: Optional[str] = None,
    ) -> None:
        """Initialize the macro F1 metric.

        Args:
            ref_key: Key name for reference label entries.
            hyp_key: Key name for hypothesis label entries.
            token_list: Optional path to the recipe's token list. It fixes the
                class set so unseen labels are caught; without it the classes
                are taken from the reference file.
        """
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        self.token_list = token_list

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        """Compute the macro-averaged F1 score.

        Args:
            data (Dict[str, Path]): Mapping of metric input aliases to SCP
                paths. Expects ``data[self.ref_key]`` and ``data[self.hyp_key]``
                to be aligned by utterance ID.
            test_name (str): Test set name. Unused by this metric.
            inference_dir (Path): Inference output root. Unused by this metric.

        Returns:
            Dict[str, float]: ``{"MacroF1": <percentage>}``

        Raises:
            RuntimeError: If ``scikit-learn`` is not installed.
            ValueError: If an utterance carries multiple labels, or if a
                reference label is missing from ``token_list``.

        Example:
            >>> metric({"ref": Path("test/ref.scp"), "hyp": Path("test/hyp.scp")},
            ...        "test", Path("infer"))
            {'MacroF1': 39.84}
        """
        sklearn_metrics = require_sklearn()
        refs = []
        hyps = []
        for _, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            refs.append(row[self.ref_key])
            hyps.append(row[self.hyp_key])

        references = single_labels(refs)
        hypotheses = single_labels(hyps)
        classes = resolve_classes(references, self.token_list)
        score = sklearn_metrics.f1_score(
            references,
            hypotheses,
            labels=classes,
            average="macro",
            zero_division=0,
        )
        return {"MacroF1": round(float(score) * 100, 2)}
