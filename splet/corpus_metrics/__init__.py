"""Metrics that need the whole corpus at once.

BLEU is the reason this tier exists: corpus BLEU is not the average of
sentence BLEUs, so it cannot be computed one utterance at a time and then
summarized. VERSA draws the same line, in ``versa/corpus_metrics``.

The contract mirrors VERSA's corpus tier::

    def bleu_setup(**kwargs) -> Any:
        '''Build whatever state this metric needs.'''

    def bleu_metric(state, pred_texts, gt_texts) -> dict:
        '''Return a flat dict of corpus-level result keys.'''

``pred_texts`` and ``gt_texts`` are lists in the same utterance order;
:func:`splet.metric_registry.measure_corpus` does the matching by key.

BLEU, chrF and TER call sacrebleu rather than reimplement it, and record the
signature sacrebleu reports -- the signature is what makes the number
comparable with a published one, and a BLEU without it is not reproducible.
sacrebleu can only give the signature once it has scored something (it
includes the number of references), so the metric puts it in its state under
``"signature"`` after measuring, and :func:`splet.metadata.metadata` writes it
into the summary. The spec registers ``tier="corpus"``,
``requires=("reference",)`` and ``outputs`` declaring its keys as ``sum``: a
corpus metric returns one result per corpus, so the sum over that one result
is the figure itself. See :mod:`splet.corpus_metrics.bleu` and the discussion
in espnet/espnet#6735.
"""

from splet.corpus_metrics.bleu import (  # noqa: F401
    bleu_setup,
    chrf_setup,
    sacrebleu_metric,
    ter_setup,
)

__all__ = ["bleu_setup", "chrf_setup", "sacrebleu_metric", "ter_setup"]
