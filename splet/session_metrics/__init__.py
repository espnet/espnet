"""Metrics scored on one recording at a time. No metric lives here yet.

The tier VERSA does not have. Audio metrics are per utterance or per corpus;
long-form and multi-speaker text metrics are neither. cpWER, ORC-WER, DER
and JER are all defined over a whole recording -- a speaker permutation only
means anything within one -- and are then pooled across recordings.

The tier exists in the skeleton because espnet/espnet#6760 asks for long-form
and multi-speaker evaluation to be first class rather than bolted onto the
utterance path later, and that is a decision about the shape of the package,
not about any one metric.

The contract is the same as the utterance tier, with sessions in place of
strings::

    def cpwer_setup(**kwargs) -> Any:
        '''Build whatever scoring this metric needs.'''

    def cpwer_metric(scorer, pred_session: Session, gt_session: Session) -> dict:
        '''Return a flat dict of result keys for one session.'''

:class:`splet.structures.Session` is what both sides arrive as. A metric here
reports its counts (``*_errors``, ``*_ref_len``) alongside its rate, so that
:func:`splet.scorer_shared.load_summary` pools them over recordings instead
of averaging rates -- see ``splet/metrics.py``.

Two things are worth settling before the first metric is written here, rather
than after:

- cpWER's assignment step is a linear assignment problem, not a search over
  permutations, because a session's cost is the sum of its pairs' costs.
- Every one of these metrics has an existing implementation that ESPnet
  recipes and CHiME evaluations already use (meeteval, dscore). Matching them
  is the requirement; a reimplementation that is merely reasonable is worse
  than useless here.
"""
