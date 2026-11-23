"""Utterance-group speaker-counting accuracy.

This is the percentage of utterance groups in which the hypothesis has as
many speaker blocks as the reference. The blocks are counted after the same
text cleaning that cpWER uses, so a block that becomes empty after cleaning
does not count as a speaker.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from espnet2.text.cleaner import TextCleaner
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field
from espnet3.systems.esp2_s2t.metrics.cpwer import split_speakers


class UtteranceGroupSpeakerCountAccuracy(BaseMetric):
    """Speaker-counting accuracy over utterance groups, as ``ug_spk_count_acc``.

    Args:
        speaker_change_symbol: The separator the checkpoint was trained with.
        clean_types: TextCleaner pipeline, the same one cpWER uses.

    Examples:
        >>> metric = UtteranceGroupSpeakerCountAccuracy("????")
        >>> metric.update(ref=["a ???? b", "c"], hyp=["a b", "c"])
        >>> metric.compute()
        {'ug_spk_count_acc': 50.0}
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("ug_spk_count_acc", "number"),)

    def __init__(self, speaker_change_symbol: str, clean_types=None) -> None:
        """Build the metric. See the class docstring for the arguments."""
        if not speaker_change_symbol:
            raise ValueError(
                "speaker_change_symbol is required and must be the symbol the "
                "checkpoint was trained with"
            )
        self.cleaner = TextCleaner(list(clean_types) if clean_types else None)
        self.speaker_change_symbol = speaker_change_symbol
        super().__init__()
        self.add_state("correct", 0, "sum")
        self.add_state("total", 0, "sum")
        # [num_ref_speakers, num_hyp_speakers] per group, kept only while a
        # report will be written.
        self.add_state("pairs", [], "cat")

    def update(self, *, ref, hyp) -> None:
        """Count the groups whose speaker-block counts match."""
        for reference, hypothesis in zip(ref, hyp):
            num_ref = len(
                split_speakers(reference, self.cleaner, self.speaker_change_symbol)
            )
            num_hyp = len(
                split_speakers(hypothesis, self.cleaner, self.speaker_change_symbol)
            )
            self.correct += int(num_ref == num_hyp)
            self.total += 1
            if self.collect_artifacts:
                self.pairs.append([num_ref, num_hyp])

    def compute(self):
        """Return the matching groups over all groups, in percent.

        Raises:
            ValueError: When no group was scored.
        """
        if self.total == 0:
            raise ValueError("speaker-counting accuracy needs at least one group")
        return {"ug_spk_count_acc": round(100.0 * self.correct / self.total, 2)}

    def write_artifacts(self, test_dir: Path) -> None:
        """Write accuracy by reference speaker count and the count confusion.

        Args:
            test_dir: Directory ``speaker_count.json`` is written into.
        """
        by_num_ref = defaultdict(lambda: [0, 0])
        for num_ref, num_hyp in self.pairs:
            by_num_ref[num_ref][0] += int(num_ref == num_hyp)
            by_num_ref[num_ref][1] += 1
        confusion = Counter(f"{num_ref}->{num_hyp}" for num_ref, num_hyp in self.pairs)
        report = {
            "accuracy_by_num_ref_speakers": {
                str(n): {"accuracy": 100.0 * correct / total, "num_groups": total}
                for n, (correct, total) in sorted(by_num_ref.items())
            },
            "confusion_ref_to_hyp": dict(sorted(confusion.items())),
        }
        with (test_dir / "speaker_count.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
