"""Overall and macro-averaged language identification accuracy."""

from collections import Counter
from pathlib import Path

from espnet3.components.metrics.base_metric import BaseMetric


class Accuracy(BaseMetric):
    """Compute accuracy over utterances and average recall over reference languages."""

    def __init__(self, ref_key: str = "ref", hyp_key: str = "hyp") -> None:
        """Select the reference and prediction SCP aliases.

        Args:
            ref_key: Reference SCP alias.
            hyp_key: Predicted language SCP alias.
        """
        self.ref_key = ref_key
        self.hyp_key = hyp_key

    def __call__(
        self, data: dict[str, Path], test_name: str, inference_dir: Path
    ) -> dict[str, float]:
        """Return overall and macro accuracy percentages.

        Args:
            data: SCP files indexed by reference and prediction aliases.
            test_name: Evaluation split name.
            inference_dir: Output root, unused by this scalar metric.

        Returns:
            A dictionary with Accuracy and Macro Accuracy, rounded to two decimals.

        Example:
            >>> Accuracy()({"ref": ref_path, "hyp": hyp_path}, "dev", output_dir)
        """
        total, correct = Counter(), Counter()
        for _, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            reference, hypothesis = row[self.ref_key].strip(), row[self.hyp_key].strip()
            if not reference or not hypothesis:
                raise ValueError("Empty LID label")
            total[reference] += 1
            correct[reference] += reference == hypothesis
        if not total:
            raise ValueError("No LID predictions found")
        return {
            "Accuracy": round(100 * sum(correct.values()) / sum(total.values()), 2),
            "Macro Accuracy": round(
                100 * sum(correct[k] / n for k, n in total.items()) / len(total), 2
            ),
        }
