"""Equal error rate metric for speaker verification trials."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.systems.spk.scoring import compute_eer, score_statistics


class EER(BaseMetric):
    """Compute the equal error rate over scored verification trials.

    The metric expects the ``infer`` stage to have written one similarity score
    and one target/nontarget label per trial. It also writes the target and
    nontarget score distributions next to the inference outputs, which is the
    usual first check when an EER looks wrong.

    Args:
        ref_key: Key of the trial label entries.
        hyp_key: Key of the trial score entries.
    """

    def __init__(self, ref_key: str = "label", hyp_key: str = "score") -> None:
        """Initialize the EER metric."""
        self.ref_key = ref_key
        self.hyp_key = hyp_key

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        """Compute the EER and write the trial score distributions.

        Args:
            data: Mapping of metric input aliases to SCP paths. Expects
                ``data[self.ref_key]`` and ``data[self.hyp_key]`` to be aligned
                by trial ID.
            test_name: Test set name used for output directory naming.
            inference_dir: Base directory holding the inference outputs.

        Returns:
            ``{"EER": <percentage>}``.

        Raises:
            AssertionError: If the score and label SCP files are not aligned.
        """
        scores = []
        labels = []
        for _, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            scores.append(float(row[self.hyp_key]))
            labels.append(int(row[self.ref_key]))

        trg_mean, trg_std, nontrg_mean, nontrg_std = score_statistics(scores, labels)
        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        with (test_dir / "score_distribution").open("w", encoding="utf-8") as f:
            f.write(f"n_trials {len(scores)}\n")
            f.write(f"target {trg_mean:.4f} +- {trg_std:.4f}\n")
            f.write(f"nontarget {nontrg_mean:.4f} +- {nontrg_std:.4f}\n")

        return {"EER": round(compute_eer(scores, labels), 3)}
