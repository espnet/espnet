"""Metrics that need the whole corpus at once. No metric lives here yet.

BLEU is the reason this tier exists, but not quite for the reason usually
given. Corpus BLEU is indeed not the average of sentence BLEUs -- and neither
is corpus WER the average of per-utterance WERs, so that alone does not tell
the two apart. The line is who does the pooling.

An error rate's sufficient statistics are two integers per utterance, and
``load_summary`` pools any metric reporting ``X_errors`` and ``X_ref_len``
without being told how, which is why WER sits in the utterance tier and
nothing is lost. BLEU's are ten -- four clipped n-gram counts, four candidate
counts, and two lengths -- combined as a brevity penalty times the geometric
mean of the pooled precisions. Generic summing cannot discover that rule, and
writing it out here would mean reimplementing sacrebleu, which is the failure
espnet/espnet#6735 is cited below for. So the pooling stays inside sacrebleu,
behind a ``corpus_score()`` that takes the whole corpus at once, and this tier
is what hands it that corpus. A per-sentence BLEU is not worth emitting
either: 4-gram precision is frequently zero on a short sentence, which is why
sentence BLEU needs smoothing that corpus BLEU does not.

VERSA draws the same line, in ``versa/corpus_metrics``.

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
