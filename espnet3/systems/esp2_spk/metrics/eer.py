"""Equal error rate metric for speaker verification trials."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from espnet3.api import Field
from espnet3.systems.esp2_spk.metrics.trials import TrialMetric
from espnet3.systems.esp2_spk.src.scoring import compute_eer, score_statistics


class EER(TrialMetric):
    """Compute the equal error rate over scored verification trials.

    Also writes the target and nontarget score distributions next to the
    inference outputs, which is the usual first check when an EER looks
    wrong.

    Examples:
        >>> EER()({"label": label_scp, "score": score_scp}, "vox1_o")
        {'EER': 0.739}
    """

    outputs = (Field("EER", "number"),)

    def compute(self) -> Dict[str, float]:
        """Return ``{"EER": <percentage>}``."""
        return {"EER": round(compute_eer(self.scores, self.labels), 3)}

    def write_artifacts(self, test_dir: Path) -> None:
        """Write ``score_distribution``: trial count, mean and std per class."""
        trg_mean, trg_std, nontrg_mean, nontrg_std = score_statistics(
            self.scores, self.labels
        )
        with (test_dir / "score_distribution").open("w", encoding="utf-8") as f:
            f.write(f"n_trials {len(self.scores)}\n")
            f.write(f"target {trg_mean:.4f} +- {trg_std:.4f}\n")
            f.write(f"nontarget {nontrg_mean:.4f} +- {nontrg_std:.4f}\n")
