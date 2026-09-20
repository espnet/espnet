"""Metrics that need the whole corpus at once. No metric lives here yet.

BLEU is the reason this tier exists: corpus BLEU is not the average of
sentence BLEUs, so it cannot be computed one utterance at a time and then
summarized. VERSA draws the same line, in ``versa/corpus_metrics``.

The contract mirrors VERSA's corpus tier::

    def bleu_setup(**kwargs) -> Any:
        '''Build whatever scoring this metric needs.'''

    def bleu_scoring(scorer, pred_texts, gt_texts) -> dict:
        '''Return a flat dict of corpus-level result keys.'''

For SacreBLEU, chrF and TER the implementation should call sacrebleu rather
than reimplement it, and record the signature sacrebleu reports -- the
signature is what makes the number comparable with a published one, and a
BLEU without it is not reproducible. See the discussion in
espnet/espnet#6735.
"""
