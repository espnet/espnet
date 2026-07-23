"""Equal Error Rate (EER) metric for speaker verification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from espnet2.utils.eer import ComputeErrorRates, ComputeMinDcf, tuneThresholdfromScore
from espnet3.components.metrics.abs_metric import AbsMetric


class EER(AbsMetric):
    """Compute EER and minDCF for speaker verification trials.

    Expects cosine similarity scores and binary trial labels (1 = same
    speaker, 0 = different speaker). Both are read from SCP files produced
    during inference.

    Example measure config:

    .. code-block:: yaml

        metrics:
          - metric:
              _target_: espnet3.systems.spk.metrics.eer.EER
              score_key: score
              label_key: label
            inputs:
              score: score
              label: label
    """

    def __init__(
        self,
        score_key: str = "score",
        label_key: str = "label",
        p_target: float = 0.01,
        c_miss: float = 1.0,
        c_fa: float = 1.0,
    ) -> None:
        """Initialize the EER metric.

        Args:
            score_key: Key name for cosine similarity scores.
            label_key: Key name for binary trial labels (1=target, 0=non-target).
            p_target: Prior probability of target speaker for minDCF.
            c_miss: Cost of a missed detection for minDCF.
            c_fa: Cost of a false alarm for minDCF.
        """
        self.score_key = score_key
        self.label_key = label_key
        self.p_target = p_target
        self.c_miss = c_miss
        self.c_fa = c_fa

    def __call__(
        self,
        data: Dict[str, List[str]],
        test_name: str,
        output_dir: Path,
    ) -> Dict[str, float]:
        """Compute EER and minDCF from trial scores and labels.

        Args:
            data: Mapping of field names to lists of strings, e.g.
                ``{"score": ["0.92", "-0.14", ...], "label": ["1", "0", ...]}``.
            test_name: Test set name used for output directory naming.
            output_dir: Root path where result files are stored.

        Returns:
            Dict with ``EER`` (%) and ``minDCF`` values.
        """
        scores = [float(s) for s in data[self.score_key]]
        labels = [int(v) for v in data[self.label_key]]

        _, eer, _, _ = tuneThresholdfromScore(scores, labels, target_fa=[0.01])

        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        min_dcf, _ = ComputeMinDcf(
            fnrs, fprs, thresholds, self.p_target, self.c_miss, self.c_fa
        )

        result = {"EER": round(eer, 4), "minDCF": round(min_dcf, 4)}

        test_dir = Path(output_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        with open(test_dir / "eer_result.txt", "w", encoding="utf-8") as f:
            f.write(f"EER: {result['EER']}%\n")
            f.write(f"minDCF: {result['minDCF']}\n")

        return result
