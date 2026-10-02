"""SPLET: Spoken Language Evaluation Toolkit.

SPLET evaluates the *outputs* of spoken language systems: text and
structured text. Audio is evaluated by VERSA (https://github.com/wavlab-speech/versa);
SPLET is its text-side counterpart and deliberately mirrors its interface.

SPLET does not import ESPnet or torch. It lives in the ESPnet repository for
now, but it is a self-contained package so that it can be split out into its
own repository and PyPI distribution without changing a single import path.
See ``splet/README.md``.
"""

from splet.metric_registry import (  # noqa: F401
    load_corpus_metrics,
    load_metrics,
    load_session_metrics,
    measure_corpus,
    measure_sessions,
    measure_utterances,
    summarize,
)

__all__ = [
    "measure_corpus",
    "measure_utterances",
    "load_corpus_metrics",
    "load_metrics",
    "load_session_metrics",
    "summarize",
    "measure_sessions",
]
