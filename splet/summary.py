#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""How each result key is summarized, and the summarizer itself.

VERSA keeps the same kind of table in ``versa/result_summary.py``: the summarizer
has to know which keys are numbers and which are strings. SPLET needs one
more distinction, because its headline numbers are error rates.

An error rate must not be summarized by averaging the per-utterance rates.
SCTK, sclite and every published WER pool the counts: the corpus rate is
``sum(errors) / sum(reference length)``, which is not the mean of the
per-utterance rates unless every utterance has the same length. A metric
therefore reports its counts next to its rate::

    {"wer": 0.25, "wer_errors": 1, "wer_ref_len": 4,
     "wer_sub": 1, "wer_del": 0, "wer_ins": 0}

and :func:`splet.summary.summarize` pools any key ``X`` for which
both ``X_errors`` and ``X_ref_len`` are present. Nothing has to be registered
here for that to work, so a new error-rate metric (orc_wer, cpwer, ...) is
pooled correctly by construction.
"""

from typing import Any, Dict, Sequence

# Keys whose value is text rather than a number. They are carried through to
# the per-utterance JSONL output and skipped by the summarizer.
STR_METRIC = [
    "ref_text",
    "hyp_text",
    "ref_normalized",
    "hyp_normalized",
    "alignment",
    "speaker_assignment",
]

# Suffixes that make a key a raw count. Counts are summed over the corpus,
# never averaged.
COUNT_SUFFIX = ("_errors", "_ref_len", "_hyp_len", "_sub", "_del", "_ins", "_hit")

# Suffixes that identify the two counts an error rate is pooled from.
ERROR_SUFFIX = "_errors"
REF_LEN_SUFFIX = "_ref_len"


def is_count_key(key: str) -> bool:
    """Return True if ``key`` holds a raw count that should be summed."""
    return key.endswith(COUNT_SUFFIX)


def is_str_key(key: str) -> bool:
    """Return True if ``key`` holds text that the summarizer must skip."""
    return key in STR_METRIC


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize per-utterance results into corpus figures.

    Counts are summed. An error rate is recomputed from the summed counts --
    ``sum(errors) / sum(ref_len)`` -- which is what SCTK reports and is not
    the mean of the per-utterance rates unless every utterance is the same
    length. Anything else numeric is averaged. Text keys are skipped.

    Args:
        results: Per-utterance results from :func:`measure_utterances`.

    Returns:
        The corpus figures, plus ``num_utterances``.
    """
    if not results:
        return {"num_utterances": 0}

    keys = [key for key in results[0] if key != "key"]
    summary: Dict[str, Any] = {"num_utterances": len(results)}

    for key in keys:
        if is_str_key(key):
            continue
        values = [result[key] for result in results if key in result]
        if is_count_key(key):
            summary[key] = sum(values)
        elif f"{key}{ERROR_SUFFIX}" in keys and f"{key}{REF_LEN_SUFFIX}" in keys:
            errors = sum(result[f"{key}{ERROR_SUFFIX}"] for result in results)
            ref_len = sum(result[f"{key}{REF_LEN_SUFFIX}"] for result in results)
            summary[key] = errors / ref_len if ref_len else 0.0
        else:
            summary[key] = sum(values) / len(values)
    return summary
