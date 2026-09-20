"""SPLET: Spoken Language Evaluation Toolkit.

SPLET evaluates the *outputs* of spoken language systems: text and
structured text. Audio is evaluated by VERSA (https://github.com/wavlab-speech/versa);
SPLET is its text-side counterpart and deliberately mirrors its interface.

SPLET does not import ESPnet or torch. It lives in the ESPnet repository for
now, but it is a self-contained package so that it can be split out into its
own repository and PyPI distribution without changing a single import path.
See ``splet/README.md``.
"""

from splet.scorer_shared import (  # noqa: F401
    corpus_scoring,
    list_scoring,
    load_corpus_modules,
    load_score_modules,
    load_session_modules,
    load_summary,
    session_scoring,
)

__all__ = [
    "corpus_scoring",
    "list_scoring",
    "load_corpus_modules",
    "load_score_modules",
    "load_session_modules",
    "load_summary",
    "session_scoring",
]
