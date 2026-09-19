"""Inference output helpers for the SLURP SLU recipe."""

from __future__ import annotations

from typing import Tuple


def split_intent(text: str) -> Tuple[str, str]:
    """Split ``"<intent> <transcript>"`` into its two parts.

    Args:
        text: A hypothesis or reference whose first token is the intent label.

    Returns:
        ``(intent, transcript)``. The transcript is empty when the text holds
        only a label, and both parts are empty for empty text -- a short
        hypothesis must not make the ``measure`` stage fail.

    Examples:
        >>> split_intent("news_query read the news")
        ('news_query', 'read the news')
        >>> split_intent("news_query")
        ('news_query', '')
    """
    parts = text.strip().split(maxsplit=1)
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def build_output(data, model_output, idx):
    """Build the output dict(s) the ``infer`` stage writes to SCP files.

    Wired in from ``inference.yaml`` as ``output_fn`` and called by
    ``InferenceRunner`` with one dataset item, its model output and its index,
    or -- when the config sets ``batch_size`` -- with a list of each, in which
    case one dict per item is returned.

    Besides the raw hypothesis and reference, the intent label and the
    transcript are written as their own fields, so ``metrics.yaml`` can score
    intent accuracy and transcript WER/CER without either contaminating the
    other. The dataset carries no ``utt_id``, so the item index is the ID.

    Returns:
        A dict with ``utt_id``, ``hyp``, ``ref``, ``hyp_intent``,
        ``ref_intent``, ``hyp_transcript`` and ``ref_transcript``, or a list of
        such dicts for batched inference.
    """
    if isinstance(data, list):
        return [
            build_output(item, output, index)
            for item, output, index in zip(data, model_output, idx)
        ]

    hypothesis = model_output[0][0]
    reference = data.get("text", "")
    hyp_intent, hyp_transcript = split_intent(hypothesis)
    ref_intent, ref_transcript = split_intent(reference)

    return {
        "utt_id": data.get("utt_id", str(idx)),
        "hyp": hypothesis,
        "ref": reference,
        "hyp_intent": hyp_intent,
        "ref_intent": ref_intent,
        "hyp_transcript": hyp_transcript,
        "ref_transcript": ref_transcript,
    }
