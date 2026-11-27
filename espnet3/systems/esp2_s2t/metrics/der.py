"""Utterance-group diarization error rate (DER).

The hypothesis segments come from the Whisper timestamps in the model
output, and the reference segments come from the timestamps in the reference
SOT text. Each speaker block between two speaker-change symbols is one
speaker. Within a block, each pair of consecutive ``<|start|> ... <|end|>``
timestamps is one segment.

The timestamps are relative to the start of each utterance group, so each
group is scored separately and the totals are summed over all groups. A
meeting is never scored as one long recording, so these numbers cannot be
compared with published session-level results. The class name and the
reported key include ``utterance group`` for this reason.

The scoring uses pyannote.metrics, which needs no external binary. Two of
its conventions differ from ``md-eval.pl`` of SCTK, and this module handles
both.

- pyannote removes ``collar / 2`` on each side of a reference boundary, while
  the ``-c`` option of md-eval removes the full value on each side.
  ``collar`` below follows md-eval and is doubled before it is passed to
  pyannote.
- By default, pyannote scores the union of the reference and hypothesis
  extents, so hypothesis speech after the end of the reference counts as
  false alarm. Each group is scored from ``0`` to the end of its reference
  instead, as md-eval does.

With both changes, the reported rate agrees with md-eval to two decimals on
the AMI SDM test set. The error components do not agree. False alarm and
confusion each differ by a few percent, in opposite directions, and the cause
was not found.
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

try:
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate
except ImportError:
    DiarizationErrorRate = None

from espnet2.text.cleaner import TextCleaner
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field
from espnet3.systems.esp2_s2t.metrics.cpwer import split_speakers

logger = logging.getLogger(__name__)

_TS_RE = re.compile(r"<\|(\d+(?:\.\d+)?)\|>")


def _parse_sot_segments(
    text: str, speaker_change_symbol: str
) -> List[Tuple[int, float, float]]:
    """Parse SOT text into ``(speaker_index, start, end)`` segments.

    Each block between speaker-change symbols is one speaker, numbered by its
    position in the text from 0. Within a block, the timestamps are paired as
    ``(start, end)`` in the order they appear. A pair with zero or negative
    length is dropped. A last timestamp without a partner is also dropped. A
    hypothesis has one when the model starts a segment and reaches its length
    limit before it ends the segment. A prepared reference has none.
    """
    segments = []
    for idx, block in enumerate(text.split(speaker_change_symbol)):
        timestamps = [float(x) for x in _TS_RE.findall(block)]
        for k in range(0, len(timestamps) - 1, 2):
            start, end = timestamps[k], timestamps[k + 1]
            if end > start:
                segments.append((idx, start, end))
    return segments


def _build_annotation(segments: List[Tuple[int, float, float]]) -> "Annotation":
    """Convert the segments of one group to a pyannote Annotation.

    Args:
        segments: ``(speaker_index, start, end)`` triples, as
            :func:`_parse_sot_segments` returns them.

    Returns:
        An Annotation with the labels ``spk<index>``. Zero-length segments
        are dropped, because pyannote would keep them as reference speech that
        no hypothesis can cover.
    """
    annotation = Annotation()
    for index, start, end in segments:
        if end > start:
            annotation[Segment(start, end)] = f"spk{index}"
    return annotation


class UtteranceGroupDER(BaseMetric):
    """DER computed for each utterance group, reported as ``ug_DER``.

    Each utterance group has one text block per speaker, separated by
    ``speaker_change_symbol``, with Whisper timestamps in the text. Each group
    is scored as its own recording and the errors are summed, as for cpWER. A
    meeting is never scored as one long recording.

    Args:
        speaker_change_symbol: The symbol that the model writes between
            speakers. It is required because it depends on the checkpoint.
            With the wrong symbol, every group would stay one block, and no
            error would be raised.
        collar: Seconds removed on each side of a reference boundary, as in
            md-eval and ``scripts/utils/score_der.sh``. The collar argument
            of pyannote is the total width, so this value is doubled before
            it is passed to pyannote.
        clean_types: TextCleaner pipeline. It is used only to count the
            reference speakers for the results by speaker count, so that these
            results cover the same groups as those of cpWER.

    Examples:
        >>> metric = UtteranceGroupDER("????", collar=0.0)
        >>> metric.update(ref=["<|0.00|> a<|2.00|>"], hyp=["<|0.00|> a<|1.00|>"])
        >>> metric.compute()
        {'ug_DER': 50.0}
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("ug_DER", "number"),)

    def __init__(
        self,
        speaker_change_symbol: str,
        collar: float = 0.25,
        clean_types=None,
    ) -> None:
        """Build the metric. See the class docstring for the arguments."""
        if not speaker_change_symbol:
            raise ValueError(
                "speaker_change_symbol is required and must be the symbol that "
                "the checkpoint was trained with. Without it, each utterance "
                "group is scored as one speaker, which gives bad scores and no "
                "error."
            )
        self.collar = collar
        self.cleaner = TextCleaner(list(clean_types) if clean_types else None)
        self.speaker_change_symbol = speaker_change_symbol
        super().__init__()
        # Seconds of error (missed, false alarm and confusion) and seconds of
        # scored reference speech. The rate is their ratio, so both are summed
        # over batches and ranks.
        self.add_state("error", 0.0, "sum")
        self.add_state("total", 0.0, "sum")
        self.add_state("num_dropped", 0, "sum")
        # [num_ref_speakers, error, total] for each group, kept only when the
        # report files are written.
        self.add_state("groups", [], "cat")

    def _ensure_pyannote(self) -> None:
        """Raise an error if the optional pyannote dependency is missing.

        Raises:
            RuntimeError: If ``pyannote.metrics`` is not installed.
        """
        if DiarizationErrorRate is None:
            raise RuntimeError(
                "pyannote.metrics is required to compute DER. "
                "Please install it with `pip install espnet[s2t]`."
            )

    def update(self, *, ref, hyp) -> None:
        """Add the error and reference seconds of each group in the batch."""
        self._ensure_pyannote()
        scorer = DiarizationErrorRate(collar=2 * self.collar)
        for reference_text, hypothesis_text in zip(ref, hyp):
            ref_segments = _parse_sot_segments(
                reference_text, self.speaker_change_symbol
            )
            if not ref_segments:
                # md-eval needs reference speech for a file to be scored.
                self.num_dropped += 1
                continue
            reference = _build_annotation(ref_segments)
            hypothesis = _build_annotation(
                _parse_sot_segments(hypothesis_text, self.speaker_change_symbol)
            )
            # md-eval scores each file from 0 to the end of its reference.
            # Otherwise pyannote would also score hypothesis speech after that
            # point and count it as false alarm.
            extent = reference.get_timeline().extent()
            parts = scorer.compute_components(
                reference, hypothesis, uem=Timeline([Segment(0.0, extent.end)])
            )
            error = (
                parts["confusion"] + parts["missed detection"] + parts["false alarm"]
            )
            self.error += error
            self.total += parts["total"]
            if self.collect_artifacts:
                # Speakers are counted with split_speakers, as in cpWER, so the
                # results by speaker count cover the same groups.
                num_speakers = len(
                    split_speakers(
                        reference_text, self.cleaner, self.speaker_change_symbol
                    )
                )
                self.groups.append([num_speakers, error, parts["total"]])

    def compute(self):
        """Return the summed error seconds over the summed reference seconds.

        Raises:
            ValueError: When no group has reference speech to score.
        """
        if self.num_dropped:
            logger.warning(
                "%d utterance group(s) had no reference segments, because the "
                "reference has no valid <start,end> timestamp pair. They are "
                "not included in DER.",
                self.num_dropped,
            )
        if self.total == 0:
            raise ValueError(
                "DER cannot be computed because no utterance group has reference "
                "speech. Check that the ref input is bound to the SOT "
                "reference text with timestamps, for example dataset:text."
            )
        return {"ug_DER": round(100.0 * self.error / self.total, 2)}

    def write_artifacts(self, test_dir: Path) -> None:
        """Write DER by reference speaker count.

        Args:
            test_dir: Directory for ``der_by_num_speakers.json``.
        """
        by_nspk = defaultdict(lambda: [0.0, 0.0, 0])
        for num_speakers, error, total in self.groups:
            sums = by_nspk[num_speakers]
            sums[0] += error
            sums[1] += total
            sums[2] += 1
        report = {
            str(n): {
                "der": 100.0 * error / total if total else None,
                "num_groups": count,
            }
            for n, (error, total, count) in sorted(by_nspk.items())
        }
        with (test_dir / "der_by_num_speakers.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
