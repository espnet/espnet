"""Metrics measured on one recording at a time. No metric lives here yet.

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
        '''Build whatever state this metric needs.'''

    def cpwer_metric(state, pred_session: Session, gt_session: Session) -> dict:
        '''Return a flat dict of result keys for one session.'''

:class:`splet.structures.Session` is what both sides arrive as. A metric here
registers a :class:`~splet.metric_registry.MetricSpec` with
``tier="session"``, ``requires`` naming what its sessions must carry
(``speakers`` for cpWER, ``timestamps`` and ``speakers`` for DER and JER;
checked before anything is measured), and ``outputs`` declaring its counts
(``_errors``, ``_ref_len``, or the DER numerator and scored time) with the
rule that pools them over recordings -- see ``splet/summary.py``. A rate is
never averaged over sessions.

Two things are worth settling before the first metric is written here, rather
than after:

- cpWER's assignment step is a linear assignment problem, not a search over
  permutations, because a session's cost is the sum of its pairs' costs.
- Every one of these metrics has an existing implementation that ESPnet
  recipes and CHiME evaluations already use (meeteval, dscore). Matching them
  is the requirement; a reimplementation that is merely reasonable is worse
  than useless here.
"""
