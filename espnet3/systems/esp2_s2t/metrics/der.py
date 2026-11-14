"""Utterance-group Diarization Error Rate.

Hypothesis timing comes from the model's own inline Whisper ``<|t|>``
timestamps; reference timing comes from the inline timestamps in the
reference SOT text. Each speaker block (delimited by the speaker-change
symbol) is treated as one speaker; within a block, consecutive
``<|start|> ... <|end|>`` pairs become segments.

Timestamps are relative to each utterance-group window, so **every group is
scored on its own** and the totals are accumulated across groups. A meeting
split across many groups is never scored as one long recording, so these
numbers are not comparable with published session-level ones. The class and
the reported key both say ``utterance group`` for that reason.

Scoring is pyannote.metrics, which needs no external binary. Two of its
conventions differ from SCTK's ``md-eval.pl`` and are handled here rather
than left to the caller:

- pyannote extrudes ``collar / 2`` around each reference boundary, while
  md-eval's ``-c`` forgives that much on each side. ``collar`` below keeps
  md-eval's meaning and is doubled on the way in.
- pyannote's default scored region is the union of the reference and
  hypothesis extents, which counts hypothesis speech past the end of the
  reference. Each group is scored over ``0`` to the end of its reference
  instead, which is md-eval's region.

With both in place the reported rate agrees with md-eval to two decimals on
AMI SDM test. The components do not: false alarm and confusion each differ
by a few percent, in opposite directions, and that residue was not traced.
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

# The speaker-change symbol belongs to the checkpoint, so it is a constructor
# argument rather than a constant: a metric cannot know which symbol a model
# was trained with.
_SEP = "▁SPKCHANGE▁"  # internal marker unlikely to occur in text
_TS_RE = re.compile(r"<\|(\d+(?:\.\d+)?)\|>")


def parse_sot_segments(
    text: str, speaker_change_symbol: str
) -> List[Tuple[int, float, float]]:
    """Parse SOT text into ``(speaker_index, start, end)`` segments.

    Each speaker-change-delimited block is one synthetic speaker, numbered by
    its position in the text (0-based). Within a block, timestamps are
    paired ``(start, end)`` in order of appearance. A zero/negative-length
    pair is dropped, and so is an unpaired trailing timestamp, which a
    decode produces when the model opens a segment and reaches its length
    limit before closing it. That tolerance is aimed at hypotheses, not
    references, which a prepared corpus does not produce.
    """
    text = text.replace(speaker_change_symbol, _SEP)
    segments = []
    for idx, block in enumerate(text.split(_SEP)):
        timestamps = [float(x) for x in _TS_RE.findall(block)]
        for k in range(0, len(timestamps) - 1, 2):
            start, end = timestamps[k], timestamps[k + 1]
            if end > start:
                segments.append((idx, start, end))
    return segments


def build_annotation(segments: List[Tuple[int, float, float]]) -> "Annotation":
    """Turn one group's segments into a pyannote Annotation.

    Args:
        segments: ``(speaker_index, start, end)`` triples, as
            :func:`parse_sot_segments` returns them.

    Returns:
        An Annotation labelled ``spk<index>``. Zero-length segments are
        dropped: pyannote keeps them, and they would count as reference
        speech that no hypothesis can ever cover.
    """
    annotation = Annotation()
    for index, start, end in segments:
        if end > start:
            annotation[Segment(start, end)] = f"spk{index}"
    return annotation


class UtteranceGroupDER(BaseMetric):
    """DER over one utterance group at a time, reported as ``ug_DER``.

    Each utterance group holds one text block per speaker, separated by
    ``speaker_change_symbol``, with inline Whisper timestamps. Every group is
    scored as its own recording and the errors are accumulated, matching
    cpWER's utterance-group scope: a meeting split across many groups is never
    scored as one long recording.

    Args:
        speaker_change_symbol: The symbol the model separates speakers
            with. Required: it belongs to the checkpoint, and the wrong one
            would leave every group as a single block instead of failing.
        collar: Seconds forgiven on **each side** of a reference boundary,
            which is md-eval's convention and the one
            ``scripts/utils/score_der.sh`` uses. pyannote's own argument is
            the total width, so this is doubled on the way in.
        clean_types: TextCleaner pipeline used only to count reference
            speakers for the by-speaker-count breakdown, so that breakdown
            covers the same groups as cpWER's.

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
                "speaker_change_symbol is required and must be the symbol the "
                "checkpoint was trained with. Splitting on nothing scores a "
                "whole utterance group as one speaker, which reads as a poor "
                "result rather than as a misconfiguration."
            )
        self.collar = collar
        self.cleaner = TextCleaner(list(clean_types) if clean_types else None)
        self.speaker_change_symbol = speaker_change_symbol
        super().__init__()
        # Seconds of missed, false-alarm and confused speech, and of scored
        # reference speech: the rate is their ratio, so they sum across
        # batches and ranks.
        self.add_state("error", 0.0, "sum")
        self.add_state("total", 0.0, "sum")
        self.add_state("num_dropped", 0, "sum")
        # [num_ref_speakers, error, total] per group, kept only while a
        # report will be written.
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
            ref_segments = parse_sot_segments(
                reference_text, self.speaker_change_symbol
            )
            if not ref_segments:
                # md-eval needs reference speech for a file to be scored.
                self.num_dropped += 1
                continue
            reference = build_annotation(ref_segments)
            hypothesis = build_annotation(
                parse_sot_segments(hypothesis_text, self.speaker_change_symbol)
            )
            # md-eval scores each file from 0 to the end of its reference.
            # pyannote would otherwise widen the region to cover hypothesis
            # speech past that end and charge it as false alarm.
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
                # Same speaker-count definition as cpWER (split_speakers), so
                # the two by-speaker-count tables describe the same groups.
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
                "%d utterance group(s) had no reference segments (the "
                "reference carried no usable <start,end> timestamp pair) and "
                "were dropped from DER.",
                self.num_dropped,
            )
        if self.total == 0:
            raise ValueError(
                "DER cannot be computed: no utterance group has reference "
                "speech. Check that the ref input is bound to the SOT "
                "reference text with timestamps, for example dataset:text."
            )
        return {"ug_DER": round(100.0 * self.error / self.total, 2)}

    def write_artifacts(self, test_dir: Path) -> None:
        """Write DER by reference speaker count.

        Args:
            test_dir: Directory ``der_by_num_speakers.json`` is written into.
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
