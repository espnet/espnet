#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Declarations describing how each result key is summarized.

VERSA keeps the same kind of table in ``versa/metrics.py``: the summarizer
has to know which keys are numbers and which are strings. SPLET needs one
more distinction, because its headline numbers are error rates.

An error rate must not be summarized by averaging the per-utterance rates.
SCTK, sclite and every published WER pool the counts: the corpus rate is
``sum(errors) / sum(reference length)``, which is not the mean of the
per-utterance rates unless every utterance has the same length. A metric
therefore reports its counts next to its rate::

    {"wer": 0.25, "wer_errors": 1, "wer_ref_len": 4,
     "wer_sub": 1, "wer_del": 0, "wer_ins": 0}

and :func:`splet.scorer_shared.load_summary` pools any key ``X`` for which
both ``X_errors`` and ``X_ref_len`` are present. Nothing has to be registered
here for that to work, so a new error-rate metric (orc_wer, cpwer, ...) is
pooled correctly by construction.
"""

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
