"""Serialize a cut's supervisions into a single SOT training/reference string.

Serialized Output Training flattens a multi-speaker recording into one
sequence: each speaker's words are gathered into a block, and the blocks are
concatenated with a speaker-change symbol between them. Three choices decide
the string, and the AMI splits do not agree on them, so all three are
arguments rather than constants:

``ordering``
    Which block comes first. ``start_time`` leads with the earliest onset.
    ``longest_first`` leads with the longest block, measured as the CHARACTER
    length of the rendered text INCLUDING its timestamp markup, not word count
    and not seconds. The sort is stable, so equal-length blocks keep the
    alphabetical speaker order they were built in.
``separator``
    The speaker-change symbol. Whisper tokenizes ``????`` as a single id
    (25629); ``<sc>`` is id 51865 and is only safe in a reference file that is
    split on rather than tokenized.
``text_norm``
    Applied to each segment's text before the timestamps wrap it, so the
    separator and the markup never reach it. See dataset/text_norm.py.

``lowercase``
    Whether to case-fold. AMI's dev split keeps its original case and
    punctuation; train and test do not.

The functions take a sequence of supervisions, not a lhotse cut, so they can
be exercised without building a manifest. Anything with ``speaker``, ``start``,
``duration`` and ``text`` attributes works.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

# Whisper emits timestamps on a 20 ms grid, so segment bounds snap to it.
TIMESTAMP_RESOLUTION = 0.02

# Sort keys for the ``ordering`` argument. Both are used with ``sorted``,
# which is stable, so ties fall back to the order blocks were built in
# (alphabetical by speaker).
_ORDERINGS = {
    "start_time": lambda block: block["start"],
    # Negating the length rather than passing reverse=True keeps ties in
    # alphabetical order under the same stable sort.
    "longest_first": lambda block: -len(block["text"]),
}


def round_nearest(value: float, resolution: float = TIMESTAMP_RESOLUTION) -> float:
    """Snap ``value`` to the nearest multiple of ``resolution``."""
    return round(value / resolution) * resolution


def merge_supervisions(
    supervisions: Sequence[Any], max_timestamp_pause: float
) -> List[Dict[str, Any]]:
    """Merge one speaker's supervisions that are separated by a short pause.

    Args:
        supervisions: Supervisions for a single speaker, in any order.
        max_timestamp_pause: Gaps of at most this many seconds are absorbed
            into the preceding segment instead of starting a new one.

    Returns:
        Segment dicts with ``start``, ``end`` and ``text``, ordered by start.
    """
    merged: List[Dict[str, Any]] = []
    for sup in sorted(supervisions, key=lambda s: s.start):
        segment = {
            "start": sup.start,
            "end": sup.start + sup.duration,
            "text": sup.text,
        }
        if merged and (sup.start - merged[-1]["end"]) <= max_timestamp_pause:
            # A short supervision can sit wholly inside a longer one. Taking
            # the later end keeps the merge from truncating the segment and
            # dropping reference speech.
            merged[-1]["end"] = max(merged[-1]["end"], segment["end"])
            merged[-1]["text"] = merged[-1]["text"] + " " + segment["text"]
        else:
            merged.append(segment)
    return merged


def speaker_blocks(
    supervisions: Sequence[Any],
    *,
    max_timestamp_pause: float = 2.0,
    lowercase: bool = True,
    use_timestamps: bool = True,
    text_norm: Optional[Callable[[str], str]] = None,
) -> List[Dict[str, Any]]:
    """Build one rendered text block per speaker.

    Blocks are produced in alphabetical speaker order, which is what a stable
    ``ordering`` sort falls back to on ties.

    Args:
        supervisions: Every supervision in the cut.
        max_timestamp_pause: Passed to :func:`merge_supervisions`.
        lowercase: Case-fold the transcript text. Ignored when ``text_norm``
            is set, because a normalizer case-folds on its own.
        use_timestamps: Wrap each segment in ``<|start|> text<|end|>``.
        text_norm: Applied to a segment's text before the timestamps wrap it,
            so the separator and the timestamp markup never reach the
            normalizer. A segment is dropped when it normalizes to nothing.

    Returns:
        Block dicts with ``speaker``, ``text`` and ``start``. Speakers whose
        text is empty are omitted, so they contribute no separator.
    """
    blocks: List[Dict[str, Any]] = []
    joiner = "" if use_timestamps else " "

    for speaker in sorted({sup.speaker for sup in supervisions}):
        segments = merge_supervisions(
            [sup for sup in supervisions if sup.speaker == speaker],
            max_timestamp_pause,
        )

        rendered: List[str] = []
        starts: List[float] = []
        for segment in segments:
            text = segment["text"].strip()
            if text_norm is not None:
                # Emptiness is judged after normalizing, not before: a segment
                # that holds only symbols normalizes away, and an empty
                # segment must not leave a bare timestamp pair behind.
                text = text_norm(text)
            elif lowercase:
                text = text.lower()
            if not text:
                continue
            if use_timestamps:
                start_ts = f"<|{round_nearest(segment['start']):.2f}|>"
                end_ts = f"<|{round_nearest(segment['end']):.2f}|>"
                text = f"{start_ts} {text}{end_ts}"
            rendered.append(text)
            starts.append(segment["start"])

        if not rendered:
            continue
        blocks.append(
            {
                "speaker": speaker,
                "text": joiner.join(rendered),
                "start": min(starts),
            }
        )
    return blocks


def build_sot_text(
    supervisions: Sequence[Any],
    *,
    max_timestamp_pause: float = 2.0,
    ordering: str = "start_time",
    separator: str = "????",
    lowercase: bool = True,
    use_timestamps: bool = True,
    eos: Optional[str] = "<|endoftext|>",
    prompt: Optional[str] = None,
    text_norm: Optional[Callable[[str], str]] = None,
) -> str:
    """Serialize a cut's supervisions into one SOT string.

    Args:
        supervisions: Every supervision in the cut.
        max_timestamp_pause: Passed to :func:`merge_supervisions`.
        ordering: Key name from ``_ORDERINGS``; see the module docstring.
        separator: Speaker-change symbol placed between blocks.
        lowercase: Case-fold the transcript text.
        use_timestamps: Wrap each segment in ``<|start|> text<|end|>``.
        eos: End-of-sequence token appended to the line, or None to omit it.
        prompt: Text prepended verbatim, with no separator, when set. The S2T
            data format expects the language and task symbols here, as in
            ``"<|en|><|transcribe|>"``. ``<sos>`` and ``<eos>`` are added
            during preprocessing, so they do not belong in it.
        text_norm: Passed to :func:`speaker_blocks`.

    Returns:
        The serialized line. A cut with no usable text yields ``eos`` alone.

    Raises:
        ValueError: When ``ordering`` is not a known key.
    """
    if ordering not in _ORDERINGS:
        raise ValueError(
            f"Unknown ordering {ordering!r}; expected one of " f"{sorted(_ORDERINGS)}."
        )

    blocks = speaker_blocks(
        supervisions,
        max_timestamp_pause=max_timestamp_pause,
        lowercase=lowercase,
        use_timestamps=use_timestamps,
        text_norm=text_norm,
    )
    blocks = sorted(blocks, key=_ORDERINGS[ordering])

    body = f" {separator} ".join(block["text"] for block in blocks)
    if not body:
        # No usable text. A prompt alone would be a target with no content.
        return "" if eos is None else eos
    if prompt:
        body = f"{prompt}{body}"
    if eos is None:
        return body
    return f"{body} {eos}"
