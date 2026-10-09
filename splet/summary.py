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
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

#: The reductions a metric may declare for a result key.
#:
#: ``sum``
#:     A raw count, added over the corpus (``wer_errors``, ``wer_ref_len``).
#: ``mean``
#:     A per-item number whose corpus figure is its mean. Never an error rate.
#: ``text``
#:     Text carried through to the per-item output and skipped here
#:     (a rendered alignment, a normalized transcript).
#: ``pool:<errors suffix>/<reference length suffix>``
#:     An error rate, recomputed from two summed counts with
#:     :func:`error_rate_from_counts`; ``pool:_errors/_ref_len`` is what WER
#:     and CER declare for their headline key.
RULES = ("sum", "mean", "text", "pool")


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


def _parse_rule(rule: str) -> tuple:
    """Split a declared rule into its kind and arguments."""
    kind, _, argument = rule.partition(":")
    if kind not in RULES:
        raise ValueError(f"unknown summary rule '{rule}'; expected one of {RULES}")
    if kind == "pool":
        errors_suffix, _, ref_len_suffix = argument.partition("/")
        if not errors_suffix or not ref_len_suffix:
            raise ValueError(
                f"a pool rule names its two count suffixes, "
                f"'pool:_errors/_ref_len'; got '{rule}'"
            )
        return kind, errors_suffix, ref_len_suffix
    if argument:
        raise ValueError(f"rule '{kind}' takes no argument; got '{rule}'")
    return (kind,)


def declared_rules(
    metrics: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Tuple[str, str]]:
    """Map every result key the loaded metrics may produce to its owner and rule.

    Args:
        metrics: From :func:`splet.metric_registry.load_metrics`: configured
            id to its module, state and spec.

    Returns:
        Result key (``<id><suffix>``) to ``(metric id, rule)``. The owner is
        recorded rather than recovered from the key, because ids may be
        prefixes of one another (``wer`` and ``wer_norm``).

    Raises:
        ValueError: If two metrics declare the same key, or a rule is malformed.
    """
    rules: Dict[str, Tuple[str, str]] = {}
    for metric_id, module in metrics.items():
        for suffix, rule in module["spec"].outputs.items():
            _parse_rule(rule)
            key = f"{metric_id}{suffix}"
            if key in rules:
                raise ValueError(
                    f"result key '{key}' is declared by both '{rules[key][0]}' "
                    f"and '{metric_id}'"
                )
            rules[key] = (metric_id, rule)
    return rules


def summarize(
    results: Sequence[Dict[str, Any]], metrics: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    """Reduce per-item results to corpus figures, by the metrics' own rules.

    Args:
        results: Per-item results from :func:`measure_utterances`, each with
            a ``key``.
        metrics: The loaded metrics the results came from, whose specs
            declare how every key is reduced.

    Returns:
        The corpus figures, plus ``num_utterances``.

    Raises:
        ValueError: If a result holds a key no loaded metric declared. A key
            without a rule has no defined corpus figure, and guessing one
            (an average, say) is how an error rate gets averaged.
    """
    summary: Dict[str, Any] = {"num_utterances": len(results)}
    if not results:
        return summary

    rules = declared_rules(metrics)
    seen = [key for key in results[0] if key != "key"]
    undeclared = [key for key in seen if key not in rules]
    if undeclared:
        raise ValueError(
            f"result keys {undeclared} are not declared by any loaded metric "
            f"({sorted(metrics)}); declare them in the metric's spec outputs"
        )

    for key in seen:
        metric_id, rule = rules[key]
        parsed = _parse_rule(rule)
        if parsed[0] == "text":
            continue
        values = [result[key] for result in results if key in result]
        if parsed[0] == "sum":
            summary[key] = sum(values)
        elif parsed[0] == "mean":
            summary[key] = sum(values) / len(values)
        else:
            _, errors_suffix, ref_len_suffix = parsed
            errors = sum(result[f"{metric_id}{errors_suffix}"] for result in results)
            ref_len = sum(result[f"{metric_id}{ref_len_suffix}"] for result in results)
            summary[key] = error_rate_from_counts(errors, ref_len)
    return summary
