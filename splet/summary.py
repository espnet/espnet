#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""How per-item results become corpus figures.

VERSA keeps the same kind of table in ``versa/result_summary.py``: the
summarizer has to know which keys are numbers and which are strings. SPLET
needs one more distinction, because its headline numbers are error rates.

An error rate must not be summarized by averaging the per-utterance rates.
SCTK, sclite and every published WER pool the counts: the corpus rate is
``sum(errors) / sum(reference length)``, which is not the mean of the
per-utterance rates unless every utterance has the same length. A metric
therefore reports its counts next to its rate::

    {"wer": 0.25, "wer_errors": 1, "wer_ref_len": 4,
     "wer_sub": 1, "wer_del": 0, "wer_ins": 0}

How each key is reduced is **declared** by the metric, in the ``outputs`` of
its :class:`~splet.metric_registry.MetricSpec`, keyed by the suffix the key
carries after the metric's configured id. Nothing is inferred from a key's
name: a key no metric declared is an error, so a metric cannot report a
number that the summary then averages by accident. The rules are the
:data:`RULES` below.

The reduction is incremental: :class:`Accumulator` folds results in one at
a time and keeps only the counts, so a training loop can add a batch per
step and read the corpus figure at the end of the epoch (and sum the
counts across workers first). :func:`summarize` is the same thing over a
complete list.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

#: The reductions a metric may declare for a result key.
#:
#: ``sum``
#:     A raw count, added over the corpus (``wer_errors``, ``wer_ref_len``).
#: ``rate``
#:     An error rate: recomputed as ``<id>_errors / <id>_ref_len`` from the
#:     two summed counts with :func:`error_rate_from_counts`, never averaged.
#: ``text``
#:     Text carried through to the per-item output and skipped here
#:     (a rendered alignment, a normalized transcript).
RULES = ("sum", "rate", "text")


def error_rate_from_counts(errors: int, ref_len: int) -> float:
    """Return the one error rate SPLET reports for a pair of counts.

    The same function serves one utterance and a whole corpus, so the two
    cannot disagree. The zero-denominator policy is the one every toolkit
    applies in practice: with no reference tokens, any error (necessarily an
    insertion) is a rate of 1.0, and no error at all is 0.0. The counts are
    reported next to the rate either way, so the two insertions behind a
    1.0 are never hidden.

    Args:
        errors: Substitutions, deletions and insertions together.
        ref_len: Reference tokens.

    Returns:
        ``errors / ref_len``, or 1.0 / 0.0 when ``ref_len`` is zero.
    """
    if ref_len == 0:
        return 1.0 if errors > 0 else 0.0
    return errors / ref_len


def declared_rules(
    metrics: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Tuple[str, str]]:
    """Map every result key the loaded metrics may produce to its owner and rule.

    Args:
        metrics: From :func:`splet.metric_registry.load_metrics`.

    Returns:
        Result key (``<id><suffix>``) to ``(metric id, rule)``. The owner is
        recorded rather than recovered from the key, because ids may be
        prefixes of one another (``wer`` and ``wer_norm``).

    Raises:
        ValueError: If a rule is unknown or two metrics declare the same key.
    """
    rules: Dict[str, Tuple[str, str]] = {}
    for metric_id, module in metrics.items():
        for suffix, rule in module["spec"].outputs.items():
            if rule not in RULES:
                raise ValueError(f"unknown summary rule '{rule}'; expected {RULES}")
            key = f"{metric_id}{suffix}"
            if key in rules:
                raise ValueError(
                    f"result key '{key}' is declared by both '{rules[key][0]}' "
                    f"and '{metric_id}'"
                )
            rules[key] = (metric_id, rule)
    return rules


class Accumulator:
    """Corpus figures built up one result at a time.

    ``add`` folds a per-item result into the running counts by the rules the
    metrics declared; ``result`` recomputes every rate from them. That is
    what a validation loop needs - one call per batch, one at the end of the
    epoch - and because the state is plain counts (:meth:`state`), it can be
    summed across workers before the rates are computed.
    :func:`summarize` is this class applied to a complete list.
    """

    def __init__(self, metrics: Mapping[str, Mapping[str, Any]]) -> None:
        """Prepare for the keys the metrics declare."""
        self._rules = declared_rules(metrics)
        self.reset()

    def reset(self) -> None:
        """Forget everything added so far."""
        self._num_items = 0
        self._sums: Dict[str, int] = {}

    def add(self, result: Mapping[str, Any]) -> "Accumulator":
        """Fold one per-item result into the running counts.

        Args:
            result: One result dict from a ``*_metric`` call or a row of
                :func:`measure_utterances`; its ``key`` is ignored, and a
                rate is skipped because it is recomputed from its counts.

        Returns:
            ``self``, so calls chain.

        Raises:
            ValueError: If the result holds a key no loaded metric declared.
        """
        self._num_items += 1
        for key, value in result.items():
            if key == "key":
                continue
            if key not in self._rules:
                raise ValueError(
                    f"result key '{key}' is not declared by any loaded metric; "
                    "declare it in the metric's spec outputs"
                )
            if self._rules[key][1] == "sum":
                self._sums[key] = self._sums.get(key, 0) + value
        return self

    def add_all(self, results: Sequence[Mapping[str, Any]]) -> "Accumulator":
        """Fold every result of a sequence; see :meth:`add`."""
        for result in results:
            self.add(result)
        return self

    def state(self) -> Dict[str, Any]:
        """Return the running counts as plain numbers.

        What a distributed caller sums across workers, field by field, before
        :meth:`result`; :meth:`load_state` takes the sum back.
        """
        return {"num_utterances": self._num_items, "sums": dict(self._sums)}

    def load_state(self, state: Mapping[str, Any]) -> "Accumulator":
        """Replace the running counts with ``state`` (from :meth:`state`)."""
        self._num_items = int(state["num_utterances"])
        self._sums = dict(state["sums"])
        return self

    def result(self) -> Dict[str, Any]:
        """Return the corpus figures for everything added so far.

        Counts are their sums and every rate is
        :func:`error_rate_from_counts` of its metric's summed ``_errors`` and
        ``_ref_len``; a rate whose counts were never seen is left out.
        """
        summary: Dict[str, Any] = {"num_utterances": self._num_items}
        if self._num_items == 0:
            return summary
        for key, (metric_id, rule) in self._rules.items():
            if rule == "sum" and key in self._sums:
                summary[key] = self._sums[key]
            elif rule == "rate":
                errors = self._sums.get(f"{metric_id}_errors")
                ref_len = self._sums.get(f"{metric_id}_ref_len")
                if errors is not None and ref_len is not None:
                    summary[key] = error_rate_from_counts(errors, ref_len)
        return summary


def summarize(
    results: Sequence[Dict[str, Any]], metrics: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    """Reduce per-item results to corpus figures, by the metrics' own rules.

    Args:
        results: Per-item results from :func:`measure_utterances`.
        metrics: The loaded metrics the results came from.

    Returns:
        The corpus figures, plus ``num_utterances``.

    Raises:
        ValueError: If a result holds a key no loaded metric declared.
    """
    return Accumulator(metrics).add_all(results).result()
