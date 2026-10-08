#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""BLEU, chrF and TER, computed by sacrebleu.

These are what ``egs2/TEMPLATE/st1/st.sh`` reports: stage 13 runs
``sacrebleu -m bleu chrf ter`` over the detokenized hypotheses. They are
called here rather than reimplemented, and the summary's ``metadata`` carries
the signature sacrebleu prints next to each score, e.g.
``nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0``. Two BLEU
numbers are only comparable when their signatures match, so a score written
out without one cannot be checked against anything.

Keyword arguments in the metrics config go straight to the sacrebleu metric
class (``sacrebleu.metrics.BLEU``, ``CHRF`` or ``TER``) and so use its names,
not the command line's::

    - name: bleu
      tokenize: zh        # sacrebleu --tokenize zh
      lowercase: true     # sacrebleu -lc

Note that ``-lc`` on the command line only lowercases BLEU. chrF stays case
sensitive and TER is case insensitive unless ``case_sensitive: true`` is
given, so st.sh's ``sacrebleu -lc`` call is ``lowercase: true`` on ``bleu``
alone. st.sh also runs ``remove_punctuation.pl`` before that call. The
``remove_punctuation`` normalizer with ``keep: "'"`` agrees with it on ASCII
text but also removes non-ASCII symbols (``°``, ``€``, ``©``) that the Perl
script keeps, so that pass is not reproduced exactly.

Scores are on sacrebleu's 0-100 scale, not the 0-1 scale of the error rates
in :mod:`splet.utterance_metrics`, so that they read the same as the numbers
in RESULTS.md and in papers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from splet.normalizers import build_normalizer

try:
    from sacrebleu.metrics import BLEU, CHRF, TER
except ImportError:  # pragma: no cover - exercised by the no-extras CI job
    BLEU = CHRF = TER = None

#: Result keys by the suffix after the configured id, and how the summary
#: reduces them (see ``splet/summary.py``). There is one result per corpus,
#: so ``sum`` leaves each figure as it is.
OUTPUTS: Dict[str, str] = {"": "sum"}

#: BLEU also reports the brevity penalty and the two lengths behind it.
BLEU_OUTPUTS: Dict[str, str] = {
    "": "sum",
    "_bp": "sum",
    "_ratio": "sum",
    "_hyp_len": "sum",
    "_ref_len": "sum",
}


def _setup(name, metric_class, metric_id, normalize, kwargs) -> Dict[str, Any]:
    """Build the sacrebleu metric object and the normalizer shared by all three."""
    if metric_class is None:
        raise ImportError(f"{name} requires sacrebleu>=2.0: pip install sacrebleu")
    return {
        "name": name,
        "metric_id": metric_id,
        "metric": metric_class(**kwargs),
        "normalizer": build_normalizer(normalize),
    }


def bleu_setup(
    metric_id: str = "bleu", normalize: Optional[list] = None, **kwargs
) -> Dict[str, Any]:
    """Prepare corpus BLEU.

    Args:
        metric_id: The configured id, and so the prefix of every reported key.
        normalize: Normalization pipeline config, applied to both sides
            before sacrebleu sees them.
        **kwargs: Passed to ``sacrebleu.metrics.BLEU``.

    Returns:
        The state passed back into :func:`sacrebleu_metric`.
    """
    return _setup("bleu", BLEU, metric_id, normalize, kwargs)


def chrf_setup(
    metric_id: str = "chrf", normalize: Optional[list] = None, **kwargs
) -> Dict[str, Any]:
    """Prepare corpus chrF. ``word_order: 2`` gives chrF++.

    Args:
        metric_id: The configured id, and so the prefix of every reported key.
        normalize: Normalization pipeline config, applied to both sides.
        **kwargs: Passed to ``sacrebleu.metrics.CHRF``.

    Returns:
        The state passed back into :func:`sacrebleu_metric`.
    """
    return _setup("chrf", CHRF, metric_id, normalize, kwargs)


def translation_edit_rate_setup(
    metric_id: str = "translation_edit_rate",
    normalize: Optional[list] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Prepare corpus TER (translation edit rate; lower is better).

    Args:
        metric_id: The configured id, and so the prefix of every reported key.
        normalize: Normalization pipeline config, applied to both sides.
        **kwargs: Passed to ``sacrebleu.metrics.TER``.

    Returns:
        The state passed back into :func:`sacrebleu_metric`.
    """
    return _setup("translation_edit_rate", TER, metric_id, normalize, kwargs)


def sacrebleu_metric(
    state: Dict[str, Any],
    pred_texts: List[str],
    gt_texts: Optional[List[str]],
) -> Dict[str, Any]:
    """Measure a whole corpus with one sacrebleu metric.

    Args:
        state: State from :func:`bleu_setup`, :func:`chrf_setup` or
            :func:`translation_edit_rate_setup`.
        pred_texts: Hypotheses, one per utterance.
        gt_texts: References, in the same order as ``pred_texts``.

    Returns:
        The score, under the configured id. BLEU also reports the brevity
        penalty and the two lengths it was computed from, which is what
        st.sh's result file shows after the score. The signature goes into
        ``state["signature"]``, where the summary's metadata picks it up.

    Raises:
        ValueError: If there are no references, or no utterances to measure.
    """
    name = state["metric_id"]
    if gt_texts is None:
        raise ValueError(f"{name} needs references (--gt)")
    if not pred_texts:
        raise ValueError(f"{name} has nothing to measure")

    normalizer = state["normalizer"]
    metric = state["metric"]
    score = metric.corpus_score(
        [normalizer(text) for text in pred_texts],
        [[normalizer(text) for text in gt_texts]],
    )

    # Not known before the first corpus_score: it includes nrefs.
    state["signature"] = str(metric.get_signature())

    result: Dict[str, Any] = {name: score.score}
    if state["name"] == "bleu":
        result.update(
            {
                f"{name}_bp": score.bp,
                f"{name}_ratio": score.ratio,
                f"{name}_hyp_len": score.sys_len,
                f"{name}_ref_len": score.ref_len,
            }
        )
    return result
