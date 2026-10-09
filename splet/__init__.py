"""SPLET: Spoken Language Evaluation Toolkit.

SPLET evaluates the *outputs* of spoken language systems: text and
structured text. Audio is evaluated by VERSA (https://github.com/wavlab-speech/versa);
SPLET is its text-side counterpart and deliberately mirrors its interface.

SPLET does not import ESPnet or torch. It lives in the ESPnet repository for
now, but it is a self-contained package so that it can be split out into its
own repository and PyPI distribution without changing a single import path.
See ``splet/README.md``.
"""

#: The package version, recorded in every summary's ``metadata`` block.
__version__ = "0.1.0"

from splet.metadata import metadata  # noqa: E402,F401
from splet.metric_registry import (  # noqa: E402,F401
    METRIC_CHOICES,
    MetricSpec,
    load_corpus_metrics,
    load_metrics,
    load_session_metrics,
    measure_corpus,
    measure_sessions,
    measure_utterances,
    validate_requirements,
)
from splet.summary import summarize  # noqa: E402,F401

__all__ = [
    "METRIC_CHOICES",
    "MetricSpec",
    "__version__",
    "load_corpus_metrics",
    "load_metrics",
    "load_session_metrics",
    "measure_corpus",
    "measure_sessions",
    "measure_utterances",
    "metadata",
    "summarize",
    "validate_requirements",
]
